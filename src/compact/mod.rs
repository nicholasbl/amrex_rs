//! Compact sparse AMR archive support.
//!
//! Compact archives store selected components in a chunked sparse layout that
//! can be reused by isosurface extraction without rereading AMReX FAB shards.

use std::io::Write;
use std::time::{Duration, Instant};

use crate::sparse_amr::{SparseAmr, SparseAmrLevel};
use crate::utility::{Aabb3u, SparseGrid3, SparseGridChunkView};
use crate::{BoundingBox, CoordinateSystem, Header, IndexDomain, PlotFile, Variable};

use anyhow::{Context, Result, ensure};
use glam::{DVec3, IVec3, UVec3};
use rkyv::{Archive, Deserialize, Serialize, rancor::Error};

const FORMAT_MAGIC: [u8; 8] = *b"AMRCMPCT";
const FORMAT_VERSION: u32 = 2;

/// An affine normalization applied to one component before it is stored.
///
/// Values are mapped from `[min, max]` to `[0, 1]` and clamped to that range.
/// The calculation happens in `f64`, before compact storage converts to `f32`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CompactNormalization {
    /// Plotfile component id to normalize.
    pub component_id: u32,
    /// Original value mapped to zero.
    pub min: f64,
    /// Original value mapped to one.
    pub max: f64,
}

/// Options controlling which data are written to a compact archive.
#[derive(Debug, Default)]
pub struct CompactOptions {
    /// Component ids to include. Empty means all plotfile variables.
    pub component_ids: Vec<u32>,
    /// Optional per-component normalization bounds.
    pub normalizations: Vec<CompactNormalization>,
}

/// Result from a compaction operation
#[derive(Debug, Default)]
pub struct WriteCompactResult {
    pub compact_time: Duration,
    pub archive_time: Duration,
    pub serialization_time: Duration,
}

/// Write selected plotfile components as a compact binary archive.
///
/// The archive format is intended for this crate's reader and may evolve while
/// the crate is pre-1.0.
pub fn write_compact(
    plot_file: &PlotFile,
    options: CompactOptions,
    dest: &mut impl Write,
) -> Result<WriteCompactResult> {
    let now = Instant::now();

    let component_ids = selected_component_ids(plot_file, &options)?;
    let normalizations = selected_normalizations(&component_ids, &options)?;
    let compact =
        CompactPlot::load_with_normalizations(plot_file, &component_ids, &normalizations)?;

    let compact_time = now.elapsed();
    let now = Instant::now();

    let archive = CompactArchive::from_plot(plot_file.header(), compact)?;
    let archive_time = now.elapsed();
    let now = Instant::now();

    let io_writer = rkyv::ser::writer::IoWriter::new(dest);
    rkyv::api::high::to_bytes_in::<_, Error>(&archive, io_writer)
        .context("serializing compact plot")?;

    let serialization_time = now.elapsed();

    Ok(WriteCompactResult {
        compact_time,
        archive_time,
        serialization_time,
    })
}

/// Read a compact binary archive produced by [`write_compact`].
pub fn read_compact(bytes: &[u8]) -> Result<CompactPlot> {
    let archive =
        rkyv::from_bytes::<CompactArchive, Error>(bytes).context("reading compact archive")?;
    archive.into_compact_plot()
}

/// Read only selected components from a compact archive without validating or
/// deserializing the unused component grids.
///
/// The returned plot preserves the requested component order. Active-cube
/// masks and archive metadata are still loaded for every AMR level.
///
/// # Safety
///
/// `bytes` must contain a valid, properly aligned `CompactArchive` produced for
/// this build's rkyv format, and must remain immutable for this call. Passing
/// malformed bytes can cause undefined behavior.
pub unsafe fn read_compact_selected_unchecked(
    bytes: &[u8],
    component_ids: &[u32],
) -> Result<CompactPlot> {
    ensure!(
        !component_ids.is_empty(),
        "at least one compact component must be selected"
    );
    // SAFETY: The caller guarantees a valid immutable compact archive.
    let archive = unsafe { rkyv::access_unchecked::<ArchivedCompactArchive>(bytes) };
    validate_archive_envelope(archive)?;

    let variable_count = archive.header.variables.len();
    let mut selected_ids = Vec::with_capacity(component_ids.len());
    let mut selected_slots = Vec::with_capacity(component_ids.len());
    for &id in component_ids {
        let id = usize::try_from(id).context("component id does not fit in usize")?;
        ensure!(id < variable_count, "component index {id} is out of range");
        ensure!(
            !selected_ids.contains(&id),
            "component index {id} was selected more than once"
        );
        let slot = archive
            .component_ids
            .iter()
            .position(|archived_id| archived_id.to_native() as usize == id)
            .with_context(|| format!("component index {id} is not stored in archive"))?;
        selected_ids.push(id);
        selected_slots.push(slot);
    }

    let variables = archive
        .header
        .variables
        .iter()
        .map(|variable| {
            Ok(Variable {
                name: variable.name.as_str().to_string(),
                index: usize::try_from(variable.index.to_native())
                    .context("variable index does not fit in usize")?,
            })
        })
        .collect::<Result<Vec<_>>>()?;

    let levels = archive
        .levels
        .iter()
        .map(|level| {
            ensure!(
                level.components.len() == archive.component_ids.len(),
                "archive level component count mismatch"
            );
            let components = selected_slots
                .iter()
                .map(|&slot| {
                    rkyv::deserialize::<ArchiveF32Grid, Error>(&level.components[slot])
                        .context("deserializing selected compact component")?
                        .into_grid()
                })
                .collect::<Result<Vec<_>>>()?;
            let active_cubes = rkyv::deserialize::<ArchiveMaskGrid, Error>(&level.active_cubes)
                .context("deserializing compact active-cube mask")?
                .into_grid();
            Ok(CompactLevel {
                level_index: usize::try_from(level.level_index.to_native())
                    .context("level index does not fit in usize")?,
                index_origin: IVec3::from_array(level.index_origin.map(|value| value.to_native())),
                physical_origin: DVec3::from_array(
                    level.physical_origin.map(|value| value.to_native()),
                ),
                cell_size: DVec3::from_array(level.cell_size.map(|value| value.to_native())),
                components,
                eligible_cubes: active_cubes,
            })
        })
        .collect::<Result<Vec<_>>>()?;

    let normalizations = archive
        .normalizations
        .iter()
        .filter_map(|normalization| {
            let component_id = normalization.component_id.to_native() as usize;
            selected_ids
                .contains(&component_id)
                .then_some(CompactNormalization {
                    component_id: component_id as u32,
                    min: normalization.min.to_native(),
                    max: normalization.max.to_native(),
                })
        })
        .collect();

    Ok(CompactPlot {
        simulation_time: archive.header.simulation_time.to_native(),
        variables,
        variable_count,
        refinement_ratios: archive
            .header
            .refinement_ratios
            .iter()
            .map(|ratio| ratio.to_native() as usize)
            .collect(),
        component_ids: selected_ids,
        normalizations,
        levels,
    })
}

/// A zero-copy, read-only view of compact archive metadata.
///
/// The view borrows the archive bytes and does not deserialize level grids,
/// chunks, masks, or component values.
pub struct CompactArchiveView<'a> {
    archive: &'a ArchivedCompactArchive,
}

/// Validate a compact archive and return a zero-copy view of its metadata.
///
/// Validation checks the complete archived object graph. It does not allocate
/// or deserialize the archived grids, but its initial cost is proportional to
/// the size of the archive.
pub fn view_compact(bytes: &[u8]) -> Result<CompactArchiveView<'_>> {
    let archive = rkyv::access::<ArchivedCompactArchive, Error>(bytes)
        .context("validating compact archive")?;
    validate_archive_envelope(archive)?;
    Ok(CompactArchiveView { archive })
}

/// Return a zero-copy metadata view without validating the archived object graph.
///
/// This still checks the compact format magic, version, and chunk size. Unlike
/// [`view_compact`], opening the view does not traverse the archive.
///
/// # Safety
///
/// `bytes` must contain a valid, properly aligned `CompactArchive` produced for
/// this build's rkyv format, and the bytes must not be mutated for the lifetime
/// of the returned view. Passing malformed bytes can cause undefined behavior.
pub unsafe fn view_compact_unchecked(bytes: &[u8]) -> Result<CompactArchiveView<'_>> {
    // SAFETY: The caller guarantees that `bytes` contain a valid archived
    // `CompactArchive` and remain immutable for the returned view's lifetime.
    let archive = unsafe { rkyv::access_unchecked::<ArchivedCompactArchive>(bytes) };
    validate_archive_envelope(archive)?;
    Ok(CompactArchiveView { archive })
}

impl CompactArchiveView<'_> {
    /// Simulation time from the original plotfile header.
    pub fn simulation_time(&self) -> f64 {
        self.archive.header.simulation_time.to_native()
    }

    /// Finest AMR level index in the original plotfile.
    pub fn finest_level(&self) -> usize {
        self.archive.header.finest_level.to_native() as usize
    }

    /// Physical domain bounds as `(minimum, maximum)`.
    pub fn domain(&self) -> ([f64; 3], [f64; 3]) {
        (
            self.archive
                .header
                .domain
                .min
                .map(|value| value.to_native()),
            self.archive
                .header
                .domain
                .max
                .map(|value| value.to_native()),
        )
    }

    /// Variables from the original plotfile as `(name, index)` pairs.
    pub fn variables(&self) -> impl ExactSizeIterator<Item = (&str, usize)> {
        self.archive
            .header
            .variables
            .iter()
            .map(|variable| (variable.name.as_str(), variable.index.to_native() as usize))
    }

    /// Refinement ratios between adjacent AMR levels.
    pub fn refinement_ratios(&self) -> impl ExactSizeIterator<Item = usize> + '_ {
        self.archive
            .header
            .refinement_ratios
            .iter()
            .map(|ratio| ratio.to_native() as usize)
    }

    /// Per-level index domains as `(minimum, maximum, index_type)` tuples.
    pub fn index_domains(
        &self,
    ) -> impl ExactSizeIterator<Item = ([i32; 3], [i32; 3], [i32; 3])> + '_ {
        self.archive.header.index_domains.iter().map(|domain| {
            (
                domain.min.map(|value| value.to_native()),
                domain.max.map(|value| value.to_native()),
                domain.index_type.map(|value| value.to_native()),
            )
        })
    }

    /// Per-level time step counters.
    pub fn level_steps(&self) -> impl ExactSizeIterator<Item = usize> + '_ {
        self.archive
            .header
            .level_steps
            .iter()
            .map(|step| step.to_native() as usize)
    }

    /// Per-level cell sizes.
    pub fn cell_sizes(&self) -> impl ExactSizeIterator<Item = [f64; 3]> + '_ {
        self.archive
            .header
            .cell_sizes
            .iter()
            .map(|size| size.map(|value| value.to_native()))
    }

    /// Coordinate-system identifier from the AMReX header.
    pub fn coordinate_system(&self) -> CoordinateSystem {
        match self.archive.header.coordinate_system {
            0 => CoordinateSystem::Cartesian,
            1 => CoordinateSystem::Cylindrical,
            2 => CoordinateSystem::Spherical,
            _ => unreachable!("compact archive coordinate system was validated"),
        }
    }

    /// Plotfile boundary width.
    pub fn boundary_width(&self) -> usize {
        self.archive.header.boundary_width.to_native() as usize
    }

    /// Original plotfile component ids stored in this archive.
    pub fn component_ids(&self) -> impl ExactSizeIterator<Item = usize> + '_ {
        self.archive
            .component_ids
            .iter()
            .map(|id| id.to_native() as usize)
    }

    /// Normalizations stored in the archive as `(component_id, min, max)`.
    pub fn normalizations(&self) -> impl ExactSizeIterator<Item = (usize, f64, f64)> + '_ {
        self.archive.normalizations.iter().map(|normalization| {
            (
                normalization.component_id.to_native() as usize,
                normalization.min.to_native(),
                normalization.max.to_native(),
            )
        })
    }

    /// Number of AMR levels stored in this archive.
    pub fn level_count(&self) -> usize {
        self.archive.levels.len()
    }
}

fn validate_archive_envelope(archive: &ArchivedCompactArchive) -> Result<()> {
    ensure!(
        archive.magic == FORMAT_MAGIC,
        "invalid compact archive magic {:?}",
        archive.magic
    );
    ensure!(
        archive.version == FORMAT_VERSION,
        "unsupported compact archive version {}",
        archive.version
    );
    ensure!(
        archive.chunk_bits == crate::utility::CHUNK_BITS,
        "unsupported compact chunk size"
    );
    ensure!(
        archive.header.coordinate_system <= 2,
        "invalid compact archive coordinate system {}",
        archive.header.coordinate_system
    );
    for (position, normalization) in archive.normalizations.iter().enumerate() {
        let component_id = normalization.component_id.to_native();
        ensure!(
            archive
                .component_ids
                .iter()
                .any(|stored| stored.to_native() == component_id),
            "normalization component {component_id} is not stored in archive"
        );
        ensure!(
            archive.normalizations[..position]
                .iter()
                .all(|previous| previous.component_id.to_native() != component_id),
            "component {component_id} has multiple normalizations"
        );
        validate_normalization_bounds(
            normalization.min.to_native(),
            normalization.max.to_native(),
        )?;
    }
    Ok(())
}

#[derive(Archive, Deserialize, Serialize)]
struct CompactArchive {
    magic: [u8; 8],
    version: u32,
    chunk_bits: u32,
    header: ArchiveHeader,
    component_ids: Vec<u32>,
    normalizations: Vec<ArchiveNormalization>,
    levels: Vec<ArchiveLevel>,
}

#[derive(Archive, Deserialize, Serialize)]
struct ArchiveNormalization {
    component_id: u32,
    min: f64,
    max: f64,
}

#[derive(Archive, Deserialize, Serialize)]
struct ArchiveHeader {
    simulation_time: f64,
    finest_level: u64,
    domain: ArchiveBoundingBox,
    variables: Vec<ArchiveVariable>,
    refinement_ratios: Vec<u64>,
    index_domains: Vec<ArchiveIndexDomain>,
    level_steps: Vec<u64>,
    cell_sizes: Vec<[f64; 3]>,
    coordinate_system: u8,
    boundary_width: u64,
}

#[derive(Archive, Deserialize, Serialize)]
struct ArchiveVariable {
    name: String,
    index: u64,
}

#[derive(Archive, Deserialize, Serialize)]
struct ArchiveBoundingBox {
    min: [f64; 3],
    max: [f64; 3],
}

#[derive(Archive, Deserialize, Serialize)]
struct ArchiveIndexDomain {
    min: [i32; 3],
    max: [i32; 3],
    index_type: [i32; 3],
}

#[derive(Archive, Deserialize, Serialize)]
struct ArchiveLevel {
    level_index: u64,
    index_origin: [i32; 3],
    physical_origin: [f64; 3],
    cell_size: [f64; 3],
    components: Vec<ArchiveF32Grid>,
    active_cubes: ArchiveMaskGrid,
}

#[derive(Archive, Deserialize, Serialize)]
struct ArchiveF32Grid {
    bounds: [u32; 3],
    chunks: Vec<ArchiveF32Chunk>,
}

#[derive(Archive, Deserialize, Serialize)]
enum ArchiveF32Chunk {
    Uniform {
        key: [u32; 3],
        value: f32,
        mask: Box<[u64; crate::utility::MASK_WORDS]>,
    },
    Dense {
        key: [u32; 3],
        values: Vec<f32>,
        mask: Box<[u64; crate::utility::MASK_WORDS]>,
    },
}

#[derive(Archive, Deserialize, Serialize)]
struct ArchiveMaskGrid {
    bounds: [u32; 3],
    chunks: Vec<ArchiveMaskChunk>,
}

#[derive(Archive, Deserialize, Serialize)]
struct ArchiveMaskChunk {
    key: [u32; 3],
    mask: Box<[u64; crate::utility::MASK_WORDS]>,
}

/// Sparse AMR data loaded from a plotfile or compact archive.
///
/// A `CompactPlot` stores selected components as sparse grids per AMR level and
/// precomputes dual-cell eligibility used by isosurface extraction.
pub struct CompactPlot {
    pub(crate) simulation_time: f64,
    pub(crate) variables: Vec<Variable>,
    pub(crate) variable_count: usize,
    pub(crate) refinement_ratios: Vec<usize>,
    pub(crate) component_ids: Vec<usize>,
    pub(crate) normalizations: Vec<CompactNormalization>,
    pub(crate) levels: Vec<CompactLevel>,
}

/// Summary of the values stored for one component in a compact archive.
///
/// `min` and `max` include infinite values but ignore NaNs. They are `None`
/// when the component contains no non-NaN values.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct CompactComponentStats {
    /// Number of present values across all AMR levels, including NaNs.
    pub value_count: usize,
    /// Number of present values that are NaN.
    pub nan_count: usize,
    /// Smallest non-NaN value across all AMR levels.
    pub min: Option<f32>,
    /// Largest non-NaN value across all AMR levels.
    pub max: Option<f32>,
}

impl CompactComponentStats {
    fn observe(&mut self, value: f32) {
        self.value_count += 1;
        if value.is_nan() {
            self.nan_count += 1;
            return;
        }

        self.min = Some(self.min.map_or(value, |current| current.min(value)));
        self.max = Some(self.max.map_or(value, |current| current.max(value)));
    }
}

pub(crate) struct CompactLevel {
    pub(crate) level_index: usize,
    pub(crate) index_origin: IVec3,
    pub(crate) physical_origin: DVec3,
    pub(crate) cell_size: DVec3,
    pub(crate) components: Vec<SparseGrid3<f32>>,
    pub(crate) eligible_cubes: SparseGrid3<()>,
}

impl CompactPlot {
    /// Load selected component ids from an AMReX plotfile into sparse AMR storage.
    ///
    /// `component_ids` are plotfile variable indices. At least one component is
    /// required because the first component defines active dual cubes.
    pub fn load(plot_file: &PlotFile, component_ids: &[usize]) -> Result<Self> {
        Self::load_with_normalizations(plot_file, component_ids, &vec![None; component_ids.len()])
    }

    fn load_with_normalizations(
        plot_file: &PlotFile,
        component_ids: &[usize],
        normalizations: &[Option<(f64, f64)>],
    ) -> Result<Self> {
        let sparse_amr = SparseAmr::load(plot_file, component_ids, normalizations)?;
        ensure!(
            sparse_amr.component_ids == component_ids,
            "sparse AMR component order changed while loading"
        );

        let mut levels: Vec<CompactLevel> = Vec::with_capacity(sparse_amr.levels.len());
        for sparse_level in sparse_amr.levels {
            let SparseAmrLevel {
                level_index,
                index_origin,
                index_type,
                physical_origin,
                cell_size,
                components,
                valid_regions,
            } = sparse_level;
            ensure!(
                index_type == IVec3::ZERO,
                "compact plot extraction requires cell-centered data"
            );
            let eligible_cubes = build_active_dual_cubes(
                components
                    .first()
                    .context("compact plot requires at least one component")?,
                &valid_regions,
            );
            levels.push(CompactLevel {
                level_index,
                index_origin,
                physical_origin,
                cell_size,
                components,
                eligible_cubes,
            });
        }

        Ok(Self {
            simulation_time: plot_file.header().simulation_time,
            variables: plot_file
                .variables()
                .iter()
                .map(|variable| Variable {
                    name: variable.name.clone(),
                    index: variable.index,
                })
                .collect(),
            variable_count: plot_file.variables().len(),
            refinement_ratios: plot_file.header().refinement_ratios.clone(),
            component_ids: component_ids.to_vec(),
            normalizations: component_ids
                .iter()
                .zip(normalizations)
                .filter_map(|(&component_id, &bounds)| {
                    bounds.map(|(min, max)| CompactNormalization {
                        component_id: component_id as u32,
                        min,
                        max,
                    })
                })
                .collect(),
            levels,
        })
    }

    /// Simulation time from the original plotfile header.
    pub fn simulation_time(&self) -> f64 {
        self.simulation_time
    }

    /// Variables from the original plotfile header.
    pub fn variables(&self) -> &[Variable] {
        &self.variables
    }

    /// Number of variables in the original plotfile.
    pub fn variable_count(&self) -> usize {
        self.variable_count
    }

    /// Original plotfile component ids stored in this compact plot.
    pub fn component_ids(&self) -> &[usize] {
        &self.component_ids
    }

    /// Return the original normalization bounds for a stored component.
    pub fn component_normalization(&self, component_id: usize) -> Option<(f64, f64)> {
        self.normalizations
            .iter()
            .find(|normalization| normalization.component_id as usize == component_id)
            .map(|normalization| (normalization.min, normalization.max))
    }

    /// Number of AMR levels stored in this compact plot.
    pub fn level_count(&self) -> usize {
        self.levels.len()
    }

    /// Compute value statistics for a stored component across all AMR levels.
    ///
    /// Returns `None` when `component_id` is not present in the compact plot.
    pub fn component_stats(&self, component_id: usize) -> Option<CompactComponentStats> {
        let slot = self.component_slot(component_id)?;
        let mut stats = CompactComponentStats::default();

        for level in &self.levels {
            let grid = &level.components[slot];
            grid.for_each_present_in_aabb(grid.bounds_aabb(), |_, value| stats.observe(value));
        }

        Some(stats)
    }

    pub(crate) fn component_slot(&self, component_id: usize) -> Option<usize> {
        self.component_ids
            .iter()
            .position(|&loaded_id| loaded_id == component_id)
    }
}

impl CompactArchive {
    fn from_plot(header: &Header, compact: CompactPlot) -> Result<Self> {
        Ok(Self {
            magic: FORMAT_MAGIC,
            version: FORMAT_VERSION,
            chunk_bits: crate::utility::CHUNK_BITS,
            header: ArchiveHeader::from_header(header),
            component_ids: compact
                .component_ids
                .into_iter()
                .map(|id| u32::try_from(id).context("component id does not fit in u32"))
                .collect::<Result<Vec<_>>>()?,
            normalizations: compact
                .normalizations
                .into_iter()
                .map(|normalization| ArchiveNormalization {
                    component_id: normalization.component_id,
                    min: normalization.min,
                    max: normalization.max,
                })
                .collect(),
            levels: compact
                .levels
                .into_iter()
                .map(ArchiveLevel::from_level)
                .collect::<Result<Vec<_>>>()?,
        })
    }

    fn into_compact_plot(self) -> Result<CompactPlot> {
        ensure!(self.magic == FORMAT_MAGIC, "invalid compact archive magic");
        ensure!(
            self.version == FORMAT_VERSION,
            "unsupported compact archive version {}",
            self.version
        );
        ensure!(
            self.chunk_bits == crate::utility::CHUNK_BITS,
            "unsupported compact chunk size"
        );

        let variables = self
            .header
            .variables
            .into_iter()
            .map(Variable::try_from)
            .collect::<Result<Vec<_>>>()?;
        let variable_count = variables.len();
        let component_ids = self
            .component_ids
            .into_iter()
            .map(|id| {
                let id = usize::try_from(id).context("component id does not fit in usize")?;
                ensure!(
                    id < variable_count,
                    "component index {id} is out of range for archive"
                );
                Ok(id)
            })
            .collect::<Result<Vec<_>>>()?;

        let mut normalizations = Vec::with_capacity(self.normalizations.len());
        for normalization in self.normalizations {
            let component_id = usize::try_from(normalization.component_id)
                .context("normalization component id does not fit in usize")?;
            ensure!(
                component_ids.contains(&component_id),
                "normalization component {component_id} is not stored in archive"
            );
            ensure!(
                normalizations
                    .iter()
                    .all(|existing: &CompactNormalization| {
                        existing.component_id != normalization.component_id
                    }),
                "component {component_id} has multiple normalizations"
            );
            validate_normalization_bounds(normalization.min, normalization.max)?;
            normalizations.push(CompactNormalization {
                component_id: normalization.component_id,
                min: normalization.min,
                max: normalization.max,
            });
        }

        let levels = self
            .levels
            .into_iter()
            .map(|level| level.into_compact_level(component_ids.len()))
            .collect::<Result<Vec<_>>>()?;

        Ok(CompactPlot {
            simulation_time: self.header.simulation_time,
            variables,
            variable_count,
            refinement_ratios: self
                .header
                .refinement_ratios
                .into_iter()
                .map(|ratio| {
                    usize::try_from(ratio).context("refinement ratio does not fit in usize")
                })
                .collect::<Result<Vec<_>>>()?,
            component_ids,
            normalizations,
            levels,
        })
    }
}

impl ArchiveHeader {
    fn from_header(header: &Header) -> Self {
        Self {
            simulation_time: header.simulation_time,
            finest_level: header.finest_level as u64,
            domain: ArchiveBoundingBox::from_bounding_box(&header.domain),
            variables: header
                .variables
                .iter()
                .map(ArchiveVariable::from_variable)
                .collect(),
            refinement_ratios: header.refinement_ratios.iter().map(|&x| x as u64).collect(),
            index_domains: header
                .index_domains
                .iter()
                .map(ArchiveIndexDomain::from_index_domain)
                .collect(),
            level_steps: header.level_steps.iter().map(|&x| x as u64).collect(),
            cell_sizes: header
                .cell_sizes
                .iter()
                .map(|&v| dvec3_to_array(v))
                .collect(),
            coordinate_system: match header.coordinate_system {
                CoordinateSystem::Cartesian => 0,
                CoordinateSystem::Cylindrical => 1,
                CoordinateSystem::Spherical => 2,
            },
            boundary_width: header.boundary_width as u64,
        }
    }
}

impl ArchiveVariable {
    fn from_variable(variable: &Variable) -> Self {
        Self {
            name: variable.name.clone(),
            index: variable.index as u64,
        }
    }
}

impl TryFrom<ArchiveVariable> for Variable {
    type Error = anyhow::Error;

    fn try_from(variable: ArchiveVariable) -> Result<Self> {
        Ok(Self {
            name: variable.name,
            index: usize::try_from(variable.index)
                .context("variable index does not fit in usize")?,
        })
    }
}

impl ArchiveBoundingBox {
    fn from_bounding_box(bounds: &BoundingBox) -> Self {
        Self {
            min: dvec3_to_array(bounds.min),
            max: dvec3_to_array(bounds.max),
        }
    }
}

impl ArchiveIndexDomain {
    fn from_index_domain(domain: &IndexDomain) -> Self {
        Self {
            min: ivec3_to_array(domain.min),
            max: ivec3_to_array(domain.max),
            index_type: ivec3_to_array(domain.index_type),
        }
    }
}

impl ArchiveLevel {
    fn from_level(level: CompactLevel) -> Result<Self> {
        Ok(Self {
            level_index: level.level_index as u64,
            index_origin: ivec3_to_array(level.index_origin),
            physical_origin: dvec3_to_array(level.physical_origin),
            cell_size: dvec3_to_array(level.cell_size),
            components: level
                .components
                .iter()
                .map(ArchiveF32Grid::from_grid)
                .collect::<Result<Vec<_>>>()?,
            active_cubes: ArchiveMaskGrid::from_grid(&level.eligible_cubes),
        })
    }

    fn into_compact_level(self, component_count: usize) -> Result<CompactLevel> {
        ensure!(
            self.components.len() == component_count,
            "archive level component count mismatch"
        );
        Ok(CompactLevel {
            level_index: usize::try_from(self.level_index)
                .context("level index does not fit in usize")?,
            index_origin: array_to_ivec3(self.index_origin),
            physical_origin: array_to_dvec3(self.physical_origin),
            cell_size: array_to_dvec3(self.cell_size),
            components: self
                .components
                .into_iter()
                .map(ArchiveF32Grid::into_grid)
                .collect::<Result<Vec<_>>>()?,
            eligible_cubes: self.active_cubes.into_grid(),
        })
    }
}

impl ArchiveF32Grid {
    fn from_grid(grid: &SparseGrid3<f32>) -> Result<Self> {
        let mut chunks = Vec::with_capacity(grid.chunk_count());
        grid.for_each_chunk(|chunk| match chunk {
            SparseGridChunkView::Uniform { key, value, mask } => {
                chunks.push(ArchiveF32Chunk::Uniform {
                    key: uvec3_to_array(key),
                    value,
                    mask: Box::new(mask),
                });
            }
            SparseGridChunkView::Dense { key, values, mask } => {
                chunks.push(ArchiveF32Chunk::Dense {
                    key: uvec3_to_array(key),
                    values: values.to_vec(),
                    mask: Box::new(mask),
                });
            }
        });
        chunks.sort_by_key(ArchiveF32Chunk::key);
        Ok(Self {
            bounds: uvec3_to_array(grid.bounds()),
            chunks,
        })
    }

    fn into_grid(self) -> Result<SparseGrid3<f32>> {
        let mut grid =
            SparseGrid3::with_chunk_capacity(array_to_uvec3(self.bounds), self.chunks.len());
        for chunk in self.chunks {
            match chunk {
                ArchiveF32Chunk::Uniform { key, value, mask } => {
                    grid.insert_uniform_chunk(array_to_uvec3(key), value, *mask);
                }
                ArchiveF32Chunk::Dense { key, values, mask } => {
                    ensure!(
                        values.len() == crate::utility::CHUNK_VOLUME,
                        "dense chunk has {} values; expected {}",
                        values.len(),
                        crate::utility::CHUNK_VOLUME
                    );
                    grid.insert_dense_chunk(array_to_uvec3(key), values.into_boxed_slice(), *mask);
                }
            }
        }
        Ok(grid)
    }
}

impl ArchiveF32Chunk {
    fn key(&self) -> [u32; 3] {
        match self {
            Self::Uniform { key, .. } | Self::Dense { key, .. } => *key,
        }
    }
}

impl ArchiveMaskGrid {
    fn from_grid(grid: &SparseGrid3<()>) -> Self {
        let mut chunks = Vec::with_capacity(grid.chunk_count());
        grid.for_each_chunk(|chunk| match chunk {
            SparseGridChunkView::Uniform { key, mask, .. }
            | SparseGridChunkView::Dense { key, mask, .. } => {
                chunks.push(ArchiveMaskChunk {
                    key: uvec3_to_array(key),
                    mask: Box::new(mask),
                });
            }
        });
        chunks.sort_by_key(|chunk| chunk.key);
        Self {
            bounds: uvec3_to_array(grid.bounds()),
            chunks,
        }
    }

    fn into_grid(self) -> SparseGrid3<()> {
        let mut grid =
            SparseGrid3::with_chunk_capacity(array_to_uvec3(self.bounds), self.chunks.len());
        for chunk in self.chunks {
            grid.insert_uniform_chunk(array_to_uvec3(chunk.key), (), *chunk.mask);
        }
        grid
    }
}

fn selected_component_ids(plot_file: &PlotFile, options: &CompactOptions) -> Result<Vec<usize>> {
    if options.component_ids.is_empty() {
        return Ok((0..plot_file.variables().len()).collect());
    }

    let component_ids = options
        .component_ids
        .iter()
        .map(|&id| {
            let id = usize::try_from(id).context("component id does not fit in usize")?;
            ensure!(
                id < plot_file.variables().len(),
                "component index {id} is out of range"
            );
            Ok(id)
        })
        .collect::<Result<Vec<_>>>()?;
    for (position, &id) in component_ids.iter().enumerate() {
        ensure!(
            !component_ids[..position].contains(&id),
            "component index {id} was selected more than once"
        );
    }
    Ok(component_ids)
}

fn selected_normalizations(
    component_ids: &[usize],
    options: &CompactOptions,
) -> Result<Vec<Option<(f64, f64)>>> {
    let mut result = vec![None; component_ids.len()];
    for normalization in &options.normalizations {
        validate_normalization_bounds(normalization.min, normalization.max)?;
        let component_id = usize::try_from(normalization.component_id)
            .context("normalization component id does not fit in usize")?;
        let slot = component_ids
            .iter()
            .position(|&id| id == component_id)
            .with_context(|| {
                format!("normalization component {component_id} is not selected for compaction")
            })?;
        ensure!(
            result[slot].is_none(),
            "component {component_id} has multiple normalizations"
        );
        result[slot] = Some((normalization.min, normalization.max));
    }
    Ok(result)
}

fn validate_normalization_bounds(min: f64, max: f64) -> Result<()> {
    ensure!(min.is_finite(), "normalization minimum must be finite");
    ensure!(max.is_finite(), "normalization maximum must be finite");
    ensure!(
        max > min,
        "normalization maximum must be greater than minimum"
    );
    ensure!(
        (max - min).is_finite(),
        "normalization range must be finite"
    );
    Ok(())
}

fn uvec3_to_array(v: UVec3) -> [u32; 3] {
    [v.x, v.y, v.z]
}

fn array_to_uvec3(v: [u32; 3]) -> UVec3 {
    UVec3::new(v[0], v[1], v[2])
}

fn ivec3_to_array(v: IVec3) -> [i32; 3] {
    [v.x, v.y, v.z]
}

fn array_to_ivec3(v: [i32; 3]) -> IVec3 {
    IVec3::new(v[0], v[1], v[2])
}

fn dvec3_to_array(v: DVec3) -> [f64; 3] {
    [v.x, v.y, v.z]
}

fn array_to_dvec3(v: [f64; 3]) -> DVec3 {
    DVec3::new(v[0], v[1], v[2])
}

pub(crate) fn build_active_dual_cubes(
    samples: &SparseGrid3<f32>,
    sample_regions: &[Aabb3u],
) -> SparseGrid3<()> {
    let sample_bounds = samples.bounds();
    let cube_bounds = UVec3::new(
        sample_bounds.x.saturating_sub(1),
        sample_bounds.y.saturating_sub(1),
        sample_bounds.z.saturating_sub(1),
    );
    let mut candidates = SparseGrid3::with_chunk_capacity(cube_bounds, sample_regions.len());

    // A sample at p may contribute to cubes anchored at p and p - 1.
    for region in sample_regions {
        let candidate_region = Aabb3u::new(
            UVec3::new(
                region.min.x.saturating_sub(1),
                region.min.y.saturating_sub(1),
                region.min.z.saturating_sub(1),
            ),
            region.max,
        );
        candidates.fill_aabb(candidate_region, ());
    }

    let mut active = SparseGrid3::with_chunk_capacity(cube_bounds, candidates.chunk_count());
    let mut accessor = samples.accessor();
    candidates.for_each_present_in_aabb(candidates.bounds_aabb(), |anchor, ()| {
        let mut complete = true;
        for dz in 0..=1 {
            for dy in 0..=1 {
                for dx in 0..=1 {
                    complete &= accessor.contains(anchor + UVec3::new(dx, dy, dz));
                }
            }
        }
        if complete {
            active.set(anchor, ());
        }
    });
    active
}

#[cfg(test)]
mod tests {
    use std::{fs, fs::File, io::Write, path::Path};

    use super::*;

    fn write_single_cube_plotfile(root: &Path) -> Result<()> {
        write_single_cube_plotfile_values(root, [0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0])
    }

    fn write_single_cube_plotfile_values(root: &Path, values: [f64; 8]) -> Result<()> {
        fs::create_dir_all(root.join("Level_0"))?;
        fs::write(
            root.join("Header"),
            "HyperCLaw-V1.1\n1\ndensity\n3\n0\n0\n0 0 0\n2 2 2\n\n\
             ((0,0,0) (1,1,1) (0,0,0))\n0\n1 1 1\n0\n0\n0 1 0\n0\n\
             0 2\n0 2\n0 2\nLevel_0/Cell\n",
        )?;
        fs::write(
            root.join("Level_0/Cell_H"),
            "1\n1\n1\n0\n(1 0\n((0,0,0) (1,1,1) (0,0,0))\n)\n1\n\
             FabOnDisk: Cell_D_00000 0\n1,1\n0\n1,1\n1\n",
        )?;

        let mut shard = File::create(root.join("Level_0/Cell_D_00000"))?;
        shard.write_all(
            b"FAB ((8, (64 11 52 0 1 12 0 1023)),(8, (8 7 6 5 4 3 2 1)))((0,0,0) (1,1,1) (0,0,0)) 1\n",
        )?;
        for value in values {
            shard.write_all(&value.to_le_bytes())?;
        }
        Ok(())
    }

    #[test]
    fn normalizes_in_f64_before_compact_storage() -> Result<()> {
        let root = std::env::temp_dir().join(format!(
            "amrex_rs_compact_normalization_test_{}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&root);
        let min = 1.0e20;
        let step = 1.0e10;
        let max = min + 7.0 * step;
        write_single_cube_plotfile_values(
            &root,
            std::array::from_fn(|index| min + index as f64 * step),
        )?;

        let result = (|| -> Result<()> {
            let plotfile = PlotFile::open(&root)?;
            let mut bytes = Vec::new();
            write_compact(
                &plotfile,
                CompactOptions {
                    component_ids: vec![0],
                    normalizations: vec![CompactNormalization {
                        component_id: 0,
                        min,
                        max,
                    }],
                },
                &mut bytes,
            )?;

            let view = view_compact(&bytes)?;
            ensure!(
                view.normalizations().eq([(0, min, max)]),
                "normalization metadata was not preserved"
            );
            let compact = read_compact(&bytes)?;
            ensure!(
                compact.component_normalization(0) == Some((min, max)),
                "loaded compact plot lost normalization metadata"
            );
            let stats = compact.component_stats(0).context("missing component")?;
            ensure!(stats.min == Some(0.0), "unexpected normalized minimum");
            ensure!(stats.max == Some(1.0), "unexpected normalized maximum");
            ensure!(
                stats.value_count == 8,
                "normalization changed the value count"
            );
            Ok(())
        })();

        let _ = fs::remove_dir_all(&root);
        result
    }

    #[test]
    fn writes_compact_archive() -> Result<()> {
        let root =
            std::env::temp_dir().join(format!("amrex_rs_compact_test_{}", std::process::id()));
        let _ = fs::remove_dir_all(&root);
        write_single_cube_plotfile(&root)?;

        let result = (|| -> Result<()> {
            let plotfile = PlotFile::open(&root)?;
            let mut bytes = Vec::new();
            write_compact(&plotfile, CompactOptions::default(), &mut bytes)?;
            ensure!(!bytes.is_empty(), "compact archive is empty");

            let view = view_compact(&bytes)?;
            ensure!(
                view.simulation_time() == plotfile.header().simulation_time,
                "compact metadata view changed simulation time"
            );
            ensure!(
                view.variables()
                    .zip(plotfile.variables())
                    .all(|((name, index), expected)| name == expected.name
                        && index == expected.index),
                "compact metadata view changed variables"
            );
            ensure!(
                view.component_ids().eq(0..plotfile.variables().len()),
                "compact metadata view changed component ids"
            );
            ensure!(
                view.level_count() == plotfile.header().finest_level + 1,
                "compact metadata view changed level count"
            );

            // SAFETY: `bytes` were just produced by `write_compact` and remain
            // immutable while the unchecked view is used.
            let unchecked_view = unsafe { view_compact_unchecked(&bytes)? };
            ensure!(
                unchecked_view.simulation_time() == view.simulation_time(),
                "unchecked metadata view changed simulation time"
            );
            ensure!(
                unchecked_view.variables().eq(view.variables()),
                "unchecked metadata view changed variables"
            );

            let compact = read_compact(&bytes)?;
            ensure!(
                compact.simulation_time() == plotfile.header().simulation_time,
                "compact archive did not preserve simulation time"
            );
            ensure!(
                compact.variables().len() == plotfile.variables().len(),
                "compact archive did not preserve variable count"
            );
            for (actual, expected) in compact.variables().iter().zip(plotfile.variables()) {
                ensure!(
                    actual.name == expected.name,
                    "compact archive changed variable name"
                );
                ensure!(
                    actual.index == expected.index,
                    "compact archive changed variable index"
                );
            }
            let stats = compact
                .component_stats(0)
                .context("stored density component is missing")?;
            ensure!(stats.value_count == 8, "unexpected compact value count");
            ensure!(stats.nan_count == 0, "unexpected NaNs in compact values");
            ensure!(stats.min == Some(0.0), "unexpected compact minimum");
            ensure!(stats.max == Some(1.0), "unexpected compact maximum");
            ensure!(
                compact.component_stats(1).is_none(),
                "unstored component unexpectedly had statistics"
            );
            let (mesh, _) = crate::isosurface_compact(
                &compact,
                crate::IsosurfaceOptions {
                    surface: crate::Surface { id: 0, value: 0.5 },
                    sampled_quantities: Vec::new(),
                    levels: None,
                    flip_winding: false,
                },
            )?;
            ensure!(!mesh.positions.is_empty(), "expected extracted vertices");
            ensure!(!mesh.indices.is_empty(), "expected extracted faces");

            // SAFETY: `bytes` were produced immediately above and remain
            // immutable for the duration of the selective read.
            let selected = unsafe { read_compact_selected_unchecked(&bytes, &[0])? };
            let (selected_mesh, _) = crate::isosurface_compact(
                &selected,
                crate::IsosurfaceOptions {
                    surface: crate::Surface { id: 0, value: 0.5 },
                    sampled_quantities: Vec::new(),
                    levels: None,
                    flip_winding: false,
                },
            )?;
            ensure!(
                selected_mesh.positions == mesh.positions,
                "selective read changed extracted vertices"
            );
            ensure!(
                selected_mesh.indices == mesh.indices,
                "selective read changed extracted faces"
            );
            Ok(())
        })();

        let _ = fs::remove_dir_all(&root);
        result
    }
}
