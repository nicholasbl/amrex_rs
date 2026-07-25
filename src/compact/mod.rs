//! Compact sparse AMR archive support.
//!
//! Compact archives store selected components in a chunked sparse layout that
//! can be reused by isosurface extraction without rereading AMReX FAB shards.

use std::io::Write;

use crate::sparse_amr::{SparseAmr, SparseAmrLevel};
use crate::utility::{Aabb3u, SparseGrid3, SparseGridChunkView};
use crate::{BoundingBox, CoordinateSystem, Header, IndexDomain, PlotFile, Variable};

use anyhow::{Context, Result, ensure};
use glam::{DVec3, IVec3, UVec3};
use rkyv::{Archive, Deserialize, Serialize, rancor::Error};

const FORMAT_MAGIC: [u8; 8] = *b"AMRCMPCT";
const FORMAT_VERSION: u32 = 1;

/// Options controlling which data are written to a compact archive.
#[derive(Debug, Default)]
pub struct CompactOptions {
    /// Component ids to include. Empty means all plotfile variables.
    pub component_ids: Vec<u32>,
}

/// Write selected plotfile components as a compact binary archive.
///
/// The archive format is intended for this crate's reader and may evolve while
/// the crate is pre-1.0.
pub fn write_compact(
    plot_file: &PlotFile,
    options: CompactOptions,
    dest: &mut impl Write,
) -> Result<()> {
    let component_ids = selected_component_ids(plot_file, &options)?;
    let compact = CompactPlot::load(plot_file, &component_ids)?;
    let archive = CompactArchive::from_plot(plot_file.header(), compact)?;
    let bytes = rkyv::to_bytes::<Error>(&archive).context("serializing compact plot")?;
    dest.write_all(&bytes).context("writing compact plot")
}

/// Read a compact binary archive produced by [`write_compact`].
pub fn read_compact(bytes: &[u8]) -> Result<CompactPlot> {
    let archive =
        rkyv::from_bytes::<CompactArchive, Error>(bytes).context("reading compact archive")?;
    archive.into_compact_plot()
}

#[derive(Archive, Deserialize, Serialize)]
struct CompactArchive {
    magic: [u8; 8],
    version: u32,
    chunk_bits: u32,
    header: ArchiveHeader,
    component_ids: Vec<u32>,
    levels: Vec<ArchiveLevel>,
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
    pub(crate) variable_count: usize,
    pub(crate) refinement_ratios: Vec<usize>,
    pub(crate) component_ids: Vec<usize>,
    pub(crate) levels: Vec<CompactLevel>,
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
        let sparse_amr = SparseAmr::load(plot_file, component_ids)?;
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
            variable_count: plot_file.variables().len(),
            refinement_ratios: plot_file.header().refinement_ratios.clone(),
            component_ids: component_ids.to_vec(),
            levels,
        })
    }

    /// Number of variables in the original plotfile.
    pub fn variable_count(&self) -> usize {
        self.variable_count
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

        let variable_count = self.header.variables.len();
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

        let levels = self
            .levels
            .into_iter()
            .map(|level| level.into_compact_level(component_ids.len()))
            .collect::<Result<Vec<_>>>()?;

        Ok(CompactPlot {
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

    options
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
        .collect()
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
        for value in [0.0_f64, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0] {
            shard.write_all(&value.to_le_bytes())?;
        }
        Ok(())
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

            let compact = read_compact(&bytes)?;
            let mesh = crate::isosurface_compact(
                &compact,
                crate::IsosurfaceOptions {
                    surface: crate::Surface { id: 0, value: 0.5 },
                    sampled_quantities: Vec::new(),
                    levels: None,
                    flip_winding: false,
                },
            )?;
            ensure!(!mesh.positions.is_empty(), "expected extracted vertices");
            ensure!(!mesh.faces.is_empty(), "expected extracted faces");
            Ok(())
        })();

        let _ = fs::remove_dir_all(&root);
        result
    }
}
