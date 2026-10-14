//! Compact sparse AMR archive support.
//!
//! Compact archives store selected components in a chunked sparse layout that
//! can be reused by isosurface extraction without rereading AMReX FAB shards.

mod archive;
mod chunk_codec;
pub(crate) mod scalar;

pub use scalar::{
    CompactChunkCodec, CompactChunkCodecSelection, CompactComponentCodec,
    CompactComponentScalarType, CompactErrorTolerance, CompactScalarType, NonFinitePolicy,
    NonFiniteReplacement,
};

use std::io::Write;
use std::time::{Duration, Instant};

use crate::sparse_amr::{SparseAmr, SparseAmrLevel};
use crate::utility::{Aabb3u, SparseGrid3};
use crate::{CoordinateSystem, PlotFile, Variable};

use anyhow::{Context, Result, ensure};
use glam::{DVec3, IVec3, UVec3};
use rkyv::rancor::Error;

use archive::*;
pub(crate) use scalar::{CompactScalarAccessor, CompactScalarGrid};

/// Options controlling which data are written to a compact archive.
#[derive(Debug, Default)]
pub struct CompactOptions {
    /// Component ids to include. Empty means all plotfile variables.
    pub component_ids: Vec<u32>,
    /// Optional reconstructed scalar-type overrides. Unlisted components use f32.
    pub scalar_types: Vec<CompactComponentScalarType>,
    /// Optional per-component chunk-codec policies. Unlisted components use
    /// `AutoLossless`.
    pub codecs: Vec<CompactComponentCodec>,
    /// How non-finite source samples are handled.
    pub non_finite_policy: NonFinitePolicy,
}

#[derive(Debug, Clone, Copy)]
struct ResolvedComponentCodec {
    component_id: u32,
    selection: CompactChunkCodecSelection,
    requested_tolerance: Option<CompactErrorTolerance>,
    max_abs_error: Option<f64>,
}

/// Result from a compaction operation
#[derive(Debug, Default)]
pub struct WriteCompactResult {
    /// Time spent loading and indexing the sparse AMR hierarchy.
    pub compact_time: Duration,
    /// Time spent building archive blocks and metadata.
    pub archive_time: Duration,
    /// Time spent encoding and writing archive bytes.
    pub serialization_time: Duration,
    /// Aggregate non-finite or unrepresentable replacements, omitting zero counts.
    pub non_finite_replacements: Vec<NonFiniteReplacement>,
}

/// Options for rewriting an existing compact archive with new chunk codecs.
#[derive(Debug, Default)]
pub struct TranscodeCompactOptions {
    /// Per-component codec policies. Unlisted stored components use
    /// `AutoLossless`. Reconstructed scalar types are preserved.
    pub codecs: Vec<CompactComponentCodec>,
}

/// Timings from transcoding an existing compact archive.
#[derive(Debug, Default)]
pub struct TranscodeCompactResult {
    /// Time spent validating and decoding the source archive.
    pub decode_time: Duration,
    /// Time spent choosing codecs and building archive metadata.
    pub archive_time: Duration,
    /// Time spent serializing and writing archive bytes.
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
    let scalar_types = selected_scalar_types(&component_ids, &options)?;
    let codec_requests = selected_codecs(&component_ids, &scalar_types, &options.codecs)?;
    let (compact, replacement_counts) = CompactPlot::load_with_scalar_types(
        plot_file,
        &component_ids,
        &scalar_types,
        options.non_finite_policy,
    )?;

    let compact_time = now.elapsed();
    let codecs = resolve_codecs(&compact, &codec_requests)?;
    let (archive_time, serialization_time) = write_archive(
        ArchiveHeader::from_header(plot_file.header()),
        compact,
        &codecs,
        dest,
    )?;
    let non_finite_replacements = component_ids
        .iter()
        .zip(replacement_counts)
        .filter_map(|(&component_id, count)| {
            (count != 0).then_some(NonFiniteReplacement {
                component_id: component_id as u32,
                count,
            })
        })
        .collect();

    Ok(WriteCompactResult {
        compact_time,
        archive_time,
        serialization_time,
        non_finite_replacements,
    })
}

/// Rewrite a compact archive using new per-component chunk-codec policies.
///
/// The source archive's reconstructed scalar types and AMR metadata are
/// preserved. Transcoding a lossy source operates on its decoded values and
/// cannot recover precision discarded by the source codec.
pub fn transcode_compact(
    bytes: &[u8],
    options: TranscodeCompactOptions,
    dest: &mut impl Write,
) -> Result<TranscodeCompactResult> {
    let start = Instant::now();
    let view = view_compact(bytes).context("validating source compact archive")?;
    let source_header = rkyv::deserialize::<CompactArchiveHeader, Error>(view.archive)
        .context("deserializing source compact archive header")?
        .header;
    let mut compact = read_compact_impl(bytes, None).context("decoding source compact archive")?;
    compact.rebuild_value_ranges();
    let decode_time = start.elapsed();

    let requests = selected_codecs(
        &compact.component_ids,
        &compact.scalar_types,
        &options.codecs,
    )?;
    let codecs = resolve_codecs(&compact, &requests)?;
    let (archive_time, serialization_time) = write_archive(source_header, compact, &codecs, dest)?;

    Ok(TranscodeCompactResult {
        decode_time,
        archive_time,
        serialization_time,
    })
}

/// Read a compact binary archive produced by [`write_compact`].
pub fn read_compact(bytes: &[u8]) -> Result<CompactPlot> {
    read_compact_impl(bytes, None)
}

/// Read only selected components from a compact archive without deserializing
/// unused component grids.
///
/// The returned plot preserves the requested component order. Active-cube
/// masks and archive metadata are still loaded for every AMR level.
///
pub fn read_compact_selected(bytes: &[u8], component_ids: &[u32]) -> Result<CompactPlot> {
    ensure!(
        !component_ids.is_empty(),
        "at least one compact component must be selected"
    );
    read_compact_impl(bytes, Some(component_ids))
}

/// Deprecated alias for [`read_compact_selected`].
///
/// # Safety
///
/// This function no longer performs unchecked operations. It remains unsafe
/// only to avoid silently changing the contract of existing callers.
#[deprecated(note = "use the safe read_compact_selected function")]
pub unsafe fn read_compact_selected_unchecked(
    bytes: &[u8],
    component_ids: &[u32],
) -> Result<CompactPlot> {
    read_compact_selected(bytes, component_ids)
}

/// A zero-copy, read-only view of compact archive metadata.
///
/// The view borrows the archive bytes and does not deserialize level grids,
/// chunks, masks, or component values.
pub struct CompactArchiveView<'a> {
    archive: &'a ArchivedCompactArchiveHeader,
    bytes: &'a [u8],
}

/// Validate a compact archive and return a zero-copy view of its metadata.
///
/// Validation checks the metadata header object graph. Scalar and mask blocks
/// are independently validated when [`read_compact`] loads them.
pub fn view_compact(bytes: &[u8]) -> Result<CompactArchiveView<'_>> {
    let header = archive_header_bytes(bytes)?;
    let archive = rkyv::access::<ArchivedCompactArchiveHeader, Error>(header)
        .context("validating compact archive header")?;
    validate_archive_header(archive, bytes)?;
    Ok(CompactArchiveView { archive, bytes })
}

/// Return a zero-copy metadata view without validating the archived object graph.
///
/// This still checks the compact format magic, version, and chunk size. Unlike
/// [`view_compact`], opening the view does not traverse the archive.
///
/// # Safety
///
/// `bytes` must contain a valid, properly aligned compact metadata header
/// produced for this build's rkyv format, and the bytes must not be mutated for
/// the lifetime of the returned view. Passing malformed bytes can cause
/// undefined behavior.
pub unsafe fn view_compact_unchecked(bytes: &[u8]) -> Result<CompactArchiveView<'_>> {
    // SAFETY: The caller guarantees that `bytes` contain a valid archived
    // `CompactArchive` and remain immutable for the returned view's lifetime.
    let header = archive_header_bytes(bytes)?;
    let archive = unsafe { rkyv::access_unchecked::<ArchivedCompactArchiveHeader>(header) };
    validate_archive_header(archive, bytes)?;
    Ok(CompactArchiveView { archive, bytes })
}

/// Aggregate storage information for one chunk codec used by one component.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CompactChunkCodecStats {
    /// Original plotfile component id.
    pub component_id: usize,
    /// Stable codec identifier stored in the archive.
    pub codec_id: u32,
    /// Version of this codec's metadata and payload schema.
    pub codec_version: u16,
    /// Number of chunks using this codec across all AMR levels.
    pub chunk_count: usize,
    /// Codec metadata plus aligned payload bytes, excluding chunk-directory overhead.
    pub stored_bytes: u64,
    /// Largest verified pointwise error recorded for these chunks.
    pub max_abs_error: f64,
}

/// Error policy recorded for one stored component.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CompactComponentErrorTolerance {
    /// Original plotfile component id.
    pub component_id: usize,
    /// User-requested tolerance, or `None` for lossless storage.
    pub requested: Option<CompactErrorTolerance>,
    /// Absolute pointwise bound resolved in the reconstructed scalar domain.
    pub resolved_max_abs_error: f64,
}

impl CompactChunkCodecStats {
    /// Stable name for a built-in codec. Unknown future/private codecs return `None`.
    pub fn codec_name(self) -> Option<&'static str> {
        match self.codec_id {
            chunk_codec::CODEC_UNIFORM => Some("uniform:f64"),
            chunk_codec::CODEC_DENSE_F32 => Some("dense:f32"),
            chunk_codec::CODEC_DENSE_F64 => Some("dense:f64"),
            chunk_codec::CODEC_DENSE_U32 => Some("dense:u32"),
            chunk_codec::CODEC_QUANTIZED_U8 => Some("quantized:u8"),
            chunk_codec::CODEC_QUANTIZED_U16 => Some("quantized:u16"),
            chunk_codec::CODEC_QUANTIZED_U32 => Some("quantized:u32"),
            chunk_codec::CODEC_AFFINE_F64 => Some("affine:f64"),
            chunk_codec::CODEC_AFFINE_RESIDUAL_U8 => Some("affine-residual:u8"),
            chunk_codec::CODEC_AFFINE_RESIDUAL_U16 => Some("affine-residual:u16"),
            chunk_codec::CODEC_AFFINE_RESIDUAL_U32 => Some("affine-residual:u32"),
            _ => None,
        }
    }
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
            .components
            .iter()
            .map(|component| component.component_id.to_native() as usize)
    }

    /// Normalizations stored in the archive as `(component_id, min, max)`.
    pub fn normalizations(&self) -> impl Iterator<Item = (usize, f64, f64)> + '_ {
        self.archive
            .components
            .iter()
            .filter(|component| component.scalar_type == 2)
            .map(|component| {
                (
                    component.component_id.to_native() as usize,
                    component.min.to_native(),
                    component.max.to_native(),
                )
            })
    }

    /// Reconstructed scalar types stored as `(component_id, scalar_type)`.
    pub fn component_scalar_types(&self) -> impl Iterator<Item = (usize, CompactScalarType)> + '_ {
        self.archive.components.iter().map(|component| {
            let id = component.component_id.to_native() as usize;
            let scalar_type = archived_scalar_type(
                component.scalar_type,
                component.min.to_native(),
                component.max.to_native(),
            )
            .expect("compact archive scalar type was validated");
            (id, scalar_type)
        })
    }

    /// Error tolerances stored for each component.
    pub fn component_error_tolerances(
        &self,
    ) -> impl ExactSizeIterator<Item = CompactComponentErrorTolerance> + '_ {
        self.archive.components.iter().map(|component| {
            let requested = archived_error_tolerance(
                component.error_tolerance_kind,
                component.error_tolerance_value.to_native(),
            )
            .expect("compact archive error tolerance was validated");
            CompactComponentErrorTolerance {
                component_id: component.component_id.to_native() as usize,
                requested,
                resolved_max_abs_error: component.resolved_max_abs_error.to_native(),
            }
        })
    }

    /// Number of AMR levels stored in this archive.
    pub fn level_count(&self) -> usize {
        self.archive.levels.len()
    }

    /// Inspect adaptive chunk codecs without decoding their scalar values.
    pub fn chunk_codec_stats(&self) -> Result<Vec<CompactChunkCodecStats>> {
        let header_start =
            self.bytes.len() - FOOTER_LENGTH as usize - archive_header_bytes(self.bytes)?.len();
        let mut stats = Vec::<CompactChunkCodecStats>::new();
        for level in self.archive.levels.iter() {
            for (slot, component) in level.components.iter().enumerate() {
                let offset = component.data.offset.to_native();
                let length = component.data.stored_length.to_native();
                let start = usize::try_from(offset).context("component offset exceeds usize")?;
                let end = start
                    .checked_add(usize::try_from(length).context("component length exceeds usize")?)
                    .context("component extent overflow")?;
                ensure!(end <= header_start, "component block is out of bounds");
                let grid =
                    rkyv::access::<ArchivedArchiveScalarGrid, Error>(&self.bytes[start..end])
                        .context("validating adaptive chunk directory")?;
                let component_id = self.archive.components[slot].component_id.to_native() as usize;
                for chunk in grid.chunks.iter() {
                    let payload_length = chunk.payload_length.to_native();
                    ensure!(
                        u64::try_from(chunk.payload.len()).ok() == Some(payload_length.div_ceil(8)),
                        "invalid aligned chunk payload length"
                    );
                    let metadata_bytes = u64::try_from(chunk.metadata.len())
                        .context("chunk metadata size exceeds u64")?;
                    let payload_bytes = u64::try_from(chunk.payload.len())
                        .context("chunk payload size exceeds u64")?
                        .checked_mul(8)
                        .context("chunk payload size overflow")?;
                    let stored_bytes = metadata_bytes
                        .checked_add(payload_bytes)
                        .context("chunk storage size overflow")?;
                    let codec_id = chunk.codec_id.to_native();
                    let codec_version = chunk.codec_version.to_native();
                    let max_abs_error = chunk.max_abs_error.to_native();
                    ensure!(
                        max_abs_error.is_finite() && max_abs_error >= 0.0,
                        "invalid chunk error bound"
                    );
                    if let Some(existing) = stats.iter_mut().find(|entry| {
                        entry.component_id == component_id
                            && entry.codec_id == codec_id
                            && entry.codec_version == codec_version
                    }) {
                        existing.chunk_count += 1;
                        existing.stored_bytes = existing
                            .stored_bytes
                            .checked_add(stored_bytes)
                            .context("codec storage size overflow")?;
                        existing.max_abs_error = existing.max_abs_error.max(max_abs_error);
                    } else {
                        stats.push(CompactChunkCodecStats {
                            component_id,
                            codec_id,
                            codec_version,
                            chunk_count: 1,
                            stored_bytes,
                            max_abs_error,
                        });
                    }
                }
            }
        }
        stats.sort_by_key(|entry| (entry.component_id, entry.codec_id, entry.codec_version));
        Ok(stats)
    }
}

fn validate_archive_header(archive: &ArchivedCompactArchiveHeader, bytes: &[u8]) -> Result<()> {
    ensure!(
        archive.chunk_bits == crate::utility::CHUNK_BITS,
        "unsupported compact chunk size"
    );
    ensure!(
        archive.header.coordinate_system <= 2,
        "invalid compact archive coordinate system {}",
        archive.header.coordinate_system
    );
    for (index, component) in archive.components.iter().enumerate() {
        archived_scalar_type(
            component.scalar_type,
            component.min.to_native(),
            component.max.to_native(),
        )?;
        let tolerance = archived_error_tolerance(
            component.error_tolerance_kind,
            component.error_tolerance_value.to_native(),
        )?;
        validate_error_tolerance(tolerance)?;
        let resolved_max_abs_error = component.resolved_max_abs_error.to_native();
        ensure!(
            resolved_max_abs_error.is_finite() && resolved_max_abs_error >= 0.0,
            "compact component has invalid resolved error bound"
        );
        ensure!(
            tolerance.is_some() || resolved_max_abs_error == 0.0,
            "lossless compact component has a nonzero resolved error bound"
        );
        let component_id = component.component_id.to_native() as usize;
        ensure!(
            component_id < archive.header.variables.len(),
            "compact component index {component_id} is out of range"
        );
        ensure!(
            !archive.components[..index]
                .iter()
                .any(|other| other.component_id == component.component_id),
            "compact component index {component_id} is duplicated"
        );
    }
    let expected_level_count = usize::try_from(archive.header.finest_level.to_native())
        .context("compact finest level exceeds usize")?
        .checked_add(1)
        .context("compact level count overflow")?;
    ensure!(
        archive.levels.len() == expected_level_count,
        "compact level count does not match finest level"
    );

    let header_start = bytes.len() - FOOTER_LENGTH as usize - archive_header_bytes(bytes)?.len();
    let mut previous_end = PREFIX_LENGTH;
    for (expected_level, level) in archive.levels.iter().enumerate() {
        ensure!(
            usize::try_from(level.level_index.to_native()).ok() == Some(expected_level),
            "compact levels are not ordered coarse-to-fine"
        );
        ensure!(
            level.components.len() == archive.components.len(),
            "compact level component count mismatch"
        );
        for component in level.components.iter() {
            previous_end = validate_archived_block(&component.data, previous_end, header_start)?;
            for range in component.ranges.iter() {
                let min = range.min.map(|value| value.to_native());
                let max = range.max.map(|value| value.to_native());
                let value_min = range.value_min.to_native();
                let value_max = range.value_max.to_native();
                let max_abs_error = range.max_abs_error.to_native();
                ensure!(
                    min.iter().zip(max).all(|(&min, max)| min <= max),
                    "compact chunk range has invalid bounds"
                );
                ensure!(
                    value_min.is_finite() && value_max.is_finite() && value_min <= value_max,
                    "compact chunk range has invalid scalar bounds"
                );
                ensure!(
                    max_abs_error.is_finite() && max_abs_error >= 0.0,
                    "compact chunk range has invalid error bound"
                );
            }
        }
        previous_end = validate_archived_block(&level.active_cubes, previous_end, header_start)?;
    }
    Ok(())
}

fn validate_archived_block(
    block: &ArchivedArchiveBlock,
    previous_end: u64,
    header_start: usize,
) -> Result<u64> {
    ensure!(
        matches!(block.compression, ArchivedArchiveCompression::None),
        "unsupported compact block compression"
    );
    let offset = block.offset.to_native();
    let stored_length = block.stored_length.to_native();
    ensure!(
        offset.is_multiple_of(8),
        "compact block is not 8-byte aligned"
    );
    ensure!(
        stored_length == block.decoded_length.to_native(),
        "invalid uncompressed block lengths"
    );
    let end = offset
        .checked_add(stored_length)
        .context("compact block extent overflow")?;
    ensure!(
        offset >= previous_end && end <= header_start as u64,
        "compact block is out of order or out of bounds"
    );
    Ok(end)
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
    pub(crate) scalar_types: Vec<CompactScalarType>,
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
    pub min: Option<f64>,
    /// Largest non-NaN value across all AMR levels.
    pub max: Option<f64>,
}

impl CompactComponentStats {
    fn observe(&mut self, value: f64) {
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
    pub(crate) components: Vec<CompactScalarGrid>,
    pub(crate) value_ranges: Vec<Vec<ChunkValueRange>>,
    pub(crate) eligible_cubes: SparseGrid3<()>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct ChunkValueRange {
    pub(crate) aabb: Aabb3u,
    pub(crate) min: f64,
    pub(crate) max: f64,
}

impl CompactPlot {
    /// Load selected component ids from an AMReX plotfile into sparse AMR storage.
    ///
    /// `component_ids` are plotfile variable indices. At least one component is
    /// required because the first component defines active dual cubes.
    pub fn load(plot_file: &PlotFile, component_ids: &[usize]) -> Result<Self> {
        let scalar_types = vec![CompactScalarType::F32; component_ids.len()];
        Self::load_with_scalar_types(
            plot_file,
            component_ids,
            &scalar_types,
            NonFinitePolicy::ReplaceWithZero,
        )
        .map(|(plot, _)| plot)
    }

    fn load_with_scalar_types(
        plot_file: &PlotFile,
        component_ids: &[usize],
        scalar_types: &[CompactScalarType],
        non_finite_policy: NonFinitePolicy,
    ) -> Result<(Self, Vec<usize>)> {
        let sparse_amr =
            SparseAmr::load(plot_file, component_ids, scalar_types, non_finite_policy)?;
        ensure!(
            sparse_amr.component_ids == component_ids,
            "sparse AMR component order changed while loading"
        );

        let non_finite_replacements = sparse_amr.non_finite_replacements;
        let mut levels: Vec<CompactLevel> = Vec::with_capacity(sparse_amr.levels.len());
        for sparse_level in sparse_amr.levels {
            let SparseAmrLevel {
                level_index,
                index_origin,
                index_type,
                physical_origin,
                cell_size,
                components,
                non_finite_replacements: _,
                valid_regions,
            } = sparse_level;
            ensure!(
                index_type == IVec3::ZERO,
                "compact plot extraction requires cell-centered data"
            );
            let eligible_cubes = build_active_dual_cubes_for_scalar(
                components
                    .first()
                    .context("compact plot requires at least one component")?,
                &valid_regions,
            );
            let value_ranges = components
                .iter()
                .map(|component| build_chunk_value_ranges(component, &eligible_cubes))
                .collect();
            levels.push(CompactLevel {
                level_index,
                index_origin,
                physical_origin,
                cell_size,
                components,
                value_ranges,
                eligible_cubes,
            });
        }

        Ok((
            Self {
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
                scalar_types: scalar_types.to_vec(),
                levels,
            },
            non_finite_replacements,
        ))
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
        match self.component_scalar_type(component_id)? {
            CompactScalarType::UNorm32 { min, max } => Some((min, max)),
            CompactScalarType::F32 | CompactScalarType::F64 => None,
        }
    }

    /// Scalar representation used for a stored component.
    pub fn component_scalar_type(&self, component_id: usize) -> Option<CompactScalarType> {
        self.component_slot(component_id)
            .and_then(|slot| self.scalar_types.get(slot).copied())
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
            grid.for_each_value_in_aabb(grid.bounds_aabb(), |_, value| stats.observe(value));
        }

        Some(stats)
    }

    pub(crate) fn component_slot(&self, component_id: usize) -> Option<usize> {
        self.component_ids
            .iter()
            .position(|&loaded_id| loaded_id == component_id)
    }

    fn rebuild_value_ranges(&mut self) {
        for level in &mut self.levels {
            level.value_ranges = level
                .components
                .iter()
                .map(|component| build_chunk_value_ranges(component, &level.eligible_cubes))
                .collect();
        }
    }
}

fn write_archive(
    source_header: ArchiveHeader,
    compact: CompactPlot,
    codecs: &[ResolvedComponentCodec],
    dest: &mut impl Write,
) -> Result<(Duration, Duration)> {
    let mut archive_time = Duration::ZERO;
    let mut serialization_time = Duration::ZERO;
    let write_start = Instant::now();
    dest.write_all(&FORMAT_MAGIC.to_le_bytes())?;
    dest.write_all(&FORMAT_VERSION.to_le_bytes())?;
    serialization_time += write_start.elapsed();
    let mut offset = PREFIX_LENGTH;

    let CompactPlot {
        scalar_types,
        component_ids,
        levels,
        ..
    } = compact;
    ensure!(
        codecs.len() == component_ids.len(),
        "compact component codec metadata mismatch"
    );
    let mut archive_levels = Vec::with_capacity(levels.len());

    for level in levels {
        ensure!(
            level.components.len() == component_ids.len()
                && level.value_ranges.len() == component_ids.len(),
            "compact level component metadata mismatch"
        );
        let mut component_blocks = Vec::with_capacity(level.components.len());
        for ((component, ranges), &codec) in
            level.components.iter().zip(&level.value_ranges).zip(codecs)
        {
            let bytes = match component {
                CompactScalarGrid::F32(grid) => {
                    let start = Instant::now();
                    let grid = chunk_codec::encode_f32(grid, codec)?;
                    archive_time += start.elapsed();
                    let start = Instant::now();
                    let bytes = rkyv::to_bytes::<Error>(&grid)
                        .context("serializing f32 compact component")?;
                    serialization_time += start.elapsed();
                    bytes
                }
                CompactScalarGrid::F64(grid) => {
                    let start = Instant::now();
                    let grid = chunk_codec::encode_f64(grid, codec)?;
                    archive_time += start.elapsed();
                    let start = Instant::now();
                    let bytes = rkyv::to_bytes::<Error>(&grid)
                        .context("serializing f64 compact component")?;
                    serialization_time += start.elapsed();
                    bytes
                }
                CompactScalarGrid::UNorm32 { values, .. } => {
                    let start = Instant::now();
                    let grid = chunk_codec::encode_unorm32(values, codec)?;
                    archive_time += start.elapsed();
                    let start = Instant::now();
                    let bytes = rkyv::to_bytes::<Error>(&grid)
                        .context("serializing UNorm32 compact component")?;
                    serialization_time += start.elapsed();
                    bytes
                }
            };
            let write_start = Instant::now();
            let data = write_block(dest, &mut offset, &bytes)?;
            serialization_time += write_start.elapsed();
            let archive_start = Instant::now();
            component_blocks.push(ArchiveComponentBlock {
                data,
                ranges: ranges
                    .iter()
                    .map(|range| ArchiveChunkRange {
                        min: uvec3_to_array(range.aabb.min),
                        max: uvec3_to_array(range.aabb.max),
                        value_min: range.min,
                        value_max: range.max,
                        max_abs_error: codec.max_abs_error.unwrap_or(0.0),
                    })
                    .collect(),
            });
            archive_time += archive_start.elapsed();
        }

        let start = Instant::now();
        let mask = ArchiveMaskGrid::from_grid(&level.eligible_cubes);
        archive_time += start.elapsed();
        let start = Instant::now();
        let mask_bytes =
            rkyv::to_bytes::<Error>(&mask).context("serializing compact active-cube mask")?;
        serialization_time += start.elapsed();
        let write_start = Instant::now();
        let active_cubes = write_block(dest, &mut offset, &mask_bytes)?;
        serialization_time += write_start.elapsed();
        let archive_start = Instant::now();
        archive_levels.push(ArchiveLevel {
            level_index: level.level_index as u64,
            index_origin: ivec3_to_array(level.index_origin),
            physical_origin: dvec3_to_array(level.physical_origin),
            cell_size: dvec3_to_array(level.cell_size),
            components: component_blocks,
            active_cubes,
        });
        archive_time += archive_start.elapsed();
    }

    let archive_start = Instant::now();
    let components = component_ids
        .into_iter()
        .zip(scalar_types)
        .zip(codecs)
        .map(|((component_id, scalar_type), codec)| {
            ensure!(
                usize::try_from(codec.component_id).ok() == Some(component_id),
                "resolved component codec order changed"
            );
            let (scalar_type, min, max) = scalar_type_fields(scalar_type);
            let (error_tolerance_kind, error_tolerance_value) =
                error_tolerance_fields(codec.requested_tolerance);
            Ok(ArchiveStoredComponent {
                component_id: u32::try_from(component_id)
                    .context("component id does not fit in u32")?,
                scalar_type,
                min,
                max,
                error_tolerance_kind,
                error_tolerance_value,
                resolved_max_abs_error: codec.max_abs_error.unwrap_or(0.0),
            })
        })
        .collect::<Result<Vec<_>>>()?;
    let archive = CompactArchiveHeader {
        chunk_bits: crate::utility::CHUNK_BITS,
        header: source_header,
        components,
        levels: archive_levels,
    };
    archive_time += archive_start.elapsed();
    let serialization_start = Instant::now();
    let header_bytes = rkyv::to_bytes::<Error>(&archive).context("serializing compact header")?;
    serialization_time += serialization_start.elapsed();

    let write_start = Instant::now();
    align_writer(dest, &mut offset)?;
    let header_offset = offset;
    dest.write_all(&header_bytes)?;
    offset = offset
        .checked_add(header_bytes.len() as u64)
        .context("compact file length overflow")?;
    dest.write_all(&header_offset.to_le_bytes())?;
    let _final_length = offset
        .checked_add(FOOTER_LENGTH)
        .context("compact file length overflow")?;
    serialization_time += write_start.elapsed();

    Ok((archive_time, serialization_time))
}

fn write_block(dest: &mut impl Write, offset: &mut u64, bytes: &[u8]) -> Result<ArchiveBlock> {
    align_writer(dest, offset)?;
    let block_offset = *offset;
    dest.write_all(bytes)?;
    *offset = offset
        .checked_add(bytes.len() as u64)
        .context("compact file length overflow")?;
    Ok(ArchiveBlock {
        offset: block_offset,
        stored_length: bytes.len() as u64,
        decoded_length: bytes.len() as u64,
        compression: ArchiveCompression::None,
    })
}

fn align_writer(dest: &mut impl Write, offset: &mut u64) -> Result<()> {
    let padding = (8 - (*offset % 8)) % 8;
    if padding != 0 {
        dest.write_all(&[0; 7][..padding as usize])?;
        *offset += padding;
    }
    Ok(())
}

fn archive_header_bytes(bytes: &[u8]) -> Result<&[u8]> {
    ensure!(
        bytes.len() >= (PREFIX_LENGTH + FOOTER_LENGTH) as usize,
        "compact file is truncated"
    );
    let magic = u32::from_le_bytes(bytes[0..4].try_into().unwrap());
    ensure!(magic == FORMAT_MAGIC, "invalid compact archive magic");
    let version = u32::from_le_bytes(bytes[4..8].try_into().unwrap());
    ensure!(
        version == FORMAT_VERSION,
        "unsupported compact archive version {version}; regenerate the archive"
    );
    let footer_start = bytes.len() - FOOTER_LENGTH as usize;
    let header_offset = u64::from_le_bytes(bytes[footer_start..].try_into().unwrap());
    ensure!(
        header_offset.is_multiple_of(8),
        "compact header is not 8-byte aligned"
    );
    let header_offset = usize::try_from(header_offset).context("header offset exceeds usize")?;
    ensure!(
        header_offset >= PREFIX_LENGTH as usize && header_offset < footer_start,
        "compact header offset is out of bounds"
    );
    Ok(&bytes[header_offset..footer_start])
}

fn read_compact_impl(bytes: &[u8], requested: Option<&[u32]>) -> Result<CompactPlot> {
    let view = view_compact(bytes).context("validating compact archive metadata")?;
    let archive = rkyv::deserialize::<CompactArchiveHeader, Error>(view.archive)
        .context("deserializing compact archive header")?;
    ensure!(
        archive.chunk_bits == crate::utility::CHUNK_BITS,
        "unsupported compact chunk size"
    );

    let variables = archive
        .header
        .variables
        .iter()
        .cloned()
        .map(Variable::try_from)
        .collect::<Result<Vec<_>>>()?;
    let variable_count = variables.len();
    let requested_ids = requested
        .map(|ids| ids.iter().map(|&id| id as usize).collect::<Vec<_>>())
        .unwrap_or_else(|| {
            archive
                .components
                .iter()
                .map(|component| component.component_id as usize)
                .collect()
        });
    let mut selected_slots = Vec::with_capacity(requested_ids.len());
    for (position, &id) in requested_ids.iter().enumerate() {
        ensure!(id < variable_count, "component index {id} is out of range");
        ensure!(
            !requested_ids[..position].contains(&id),
            "component index {id} was selected more than once"
        );
        selected_slots.push(
            archive
                .components
                .iter()
                .position(|component| component.component_id as usize == id)
                .with_context(|| format!("component index {id} is not stored in archive"))?,
        );
    }
    let scalar_types = selected_slots
        .iter()
        .map(|&slot| {
            let component = &archive.components[slot];
            archived_scalar_type(component.scalar_type, component.min, component.max)
        })
        .collect::<Result<Vec<_>>>()?;

    let levels = archive
        .levels
        .iter()
        .map(|level| {
            ensure!(
                level.components.len() == archive.components.len(),
                "archive level component count mismatch"
            );
            let components = selected_slots
                .iter()
                .zip(&scalar_types)
                .map(|(&slot, &scalar_type)| {
                    read_scalar_block(bytes, &level.components[slot].data, scalar_type)
                })
                .collect::<Result<Vec<_>>>()?;
            let value_ranges = selected_slots
                .iter()
                .map(|&slot| {
                    level.components[slot]
                        .ranges
                        .iter()
                        .map(|range| ChunkValueRange {
                            aabb: Aabb3u::new(array_to_uvec3(range.min), array_to_uvec3(range.max)),
                            min: (range.value_min - range.max_abs_error).max(-f64::MAX),
                            max: (range.value_max + range.max_abs_error).min(f64::MAX),
                        })
                        .collect()
                })
                .collect();
            let mask_slice = block_slice(bytes, &level.active_cubes)?;
            let active_cubes = rkyv::from_bytes::<ArchiveMaskGrid, Error>(mask_slice)
                .context("reading compact active-cube block")?
                .into_grid();
            Ok(CompactLevel {
                level_index: usize::try_from(level.level_index)
                    .context("level index does not fit in usize")?,
                index_origin: array_to_ivec3(level.index_origin),
                physical_origin: array_to_dvec3(level.physical_origin),
                cell_size: array_to_dvec3(level.cell_size),
                components,
                value_ranges,
                eligible_cubes: active_cubes,
            })
        })
        .collect::<Result<Vec<_>>>()?;

    Ok(CompactPlot {
        simulation_time: archive.header.simulation_time,
        variables,
        variable_count,
        refinement_ratios: archive
            .header
            .refinement_ratios
            .iter()
            .map(|&ratio| usize::try_from(ratio).context("refinement ratio does not fit in usize"))
            .collect::<Result<Vec<_>>>()?,
        component_ids: requested_ids,
        scalar_types,
        levels,
    })
}

fn block_slice<'a>(bytes: &'a [u8], block: &ArchiveBlock) -> Result<&'a [u8]> {
    ensure!(
        matches!(block.compression, ArchiveCompression::None),
        "unsupported compact block compression"
    );
    ensure!(
        block.stored_length == block.decoded_length,
        "invalid uncompressed block lengths"
    );
    ensure!(
        block.offset.is_multiple_of(8),
        "compact block is not 8-byte aligned"
    );
    let start = usize::try_from(block.offset).context("block offset exceeds usize")?;
    let length = usize::try_from(block.stored_length).context("block length exceeds usize")?;
    let end = start.checked_add(length).context("block extent overflow")?;
    let header_start = bytes.len() - FOOTER_LENGTH as usize - archive_header_bytes(bytes)?.len();
    ensure!(
        start >= PREFIX_LENGTH as usize && end <= header_start,
        "compact block is out of bounds"
    );
    Ok(&bytes[start..end])
}

fn read_scalar_block(
    bytes: &[u8],
    block: &ArchiveBlock,
    scalar_type: CompactScalarType,
) -> Result<CompactScalarGrid> {
    let block = block_slice(bytes, block)?;
    let archive = rkyv::from_bytes::<ArchiveScalarGrid, Error>(block)
        .context("reading adaptive compact scalar grid")?;
    match scalar_type {
        CompactScalarType::F32 => Ok(CompactScalarGrid::F32(
            chunk_codec::decode_f32(archive).context("decoding f32 compact component")?,
        )),
        CompactScalarType::F64 => Ok(CompactScalarGrid::F64(
            chunk_codec::decode_f64(archive).context("decoding f64 compact component")?,
        )),
        CompactScalarType::UNorm32 { min, max } => Ok(CompactScalarGrid::UNorm32 {
            values: chunk_codec::decode_unorm32(archive)
                .context("decoding UNorm32 compact component")?,
            min,
            max,
        }),
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

fn selected_scalar_types(
    component_ids: &[usize],
    options: &CompactOptions,
) -> Result<Vec<CompactScalarType>> {
    let mut result = vec![CompactScalarType::F32; component_ids.len()];
    let mut assigned = vec![false; component_ids.len()];
    for requested in &options.scalar_types {
        requested.scalar_type.validate()?;
        let component_id = usize::try_from(requested.component_id)
            .context("scalar-type component id does not fit in usize")?;
        let slot = component_ids
            .iter()
            .position(|&id| id == component_id)
            .with_context(|| {
                format!("scalar-type component {component_id} is not selected for compaction")
            })?;
        ensure!(
            !assigned[slot],
            "component {component_id} has multiple scalar-type overrides"
        );
        assigned[slot] = true;
        result[slot] = requested.scalar_type;
    }
    Ok(result)
}

fn selected_codecs(
    component_ids: &[usize],
    scalar_types: &[CompactScalarType],
    requested_codecs: &[CompactComponentCodec],
) -> Result<Vec<CompactComponentCodec>> {
    ensure!(
        component_ids.len() == scalar_types.len(),
        "component scalar-type count mismatch"
    );
    let mut result = component_ids
        .iter()
        .map(|&component_id| CompactComponentCodec {
            component_id: component_id as u32,
            selection: CompactChunkCodecSelection::AutoLossless,
            error_tolerance: None,
        })
        .collect::<Vec<_>>();
    let mut assigned = vec![false; component_ids.len()];
    for &requested in requested_codecs {
        validate_error_tolerance(requested.error_tolerance)?;
        let component_id = usize::try_from(requested.component_id)
            .context("codec component id does not fit in usize")?;
        let slot = component_ids
            .iter()
            .position(|&id| id == component_id)
            .with_context(|| {
                format!("codec component {component_id} is not selected for compaction")
            })?;
        ensure!(
            !assigned[slot],
            "component {component_id} has multiple codec policies"
        );
        assigned[slot] = true;
        validate_codec_policy(scalar_types[slot], requested)?;
        result[slot] = requested;
    }
    Ok(result)
}

fn validate_codec_policy(
    scalar_type: CompactScalarType,
    policy: CompactComponentCodec,
) -> Result<()> {
    let requires_error = match policy.selection {
        CompactChunkCodecSelection::AutoLossless => false,
        CompactChunkCodecSelection::AutoSmallest => true,
        CompactChunkCodecSelection::Forced(codec) => match codec {
            CompactChunkCodec::DenseF32 if scalar_type == CompactScalarType::F32 => false,
            CompactChunkCodec::DenseF64 if scalar_type == CompactScalarType::F64 => false,
            CompactChunkCodec::DenseU32
                if matches!(scalar_type, CompactScalarType::UNorm32 { .. }) =>
            {
                false
            }
            CompactChunkCodec::DenseF64 | CompactChunkCodec::DenseU32 => {
                anyhow::bail!(
                    "codec {} is incompatible with scalar type {scalar_type:?}",
                    chunk_codec::codec_name(codec)
                )
            }
            _ => true,
        },
    };
    ensure!(
        policy.error_tolerance.is_some() == requires_error,
        if requires_error {
            "lossy codec policy requires an error tolerance"
        } else {
            "lossless codec policy does not accept an error tolerance"
        }
    );
    Ok(())
}

fn validate_error_tolerance(tolerance: Option<CompactErrorTolerance>) -> Result<()> {
    match tolerance {
        None => Ok(()),
        Some(CompactErrorTolerance::Absolute(value)) => {
            ensure!(
                value.is_finite() && value >= 0.0,
                "absolute error tolerance must be finite and non-negative"
            );
            Ok(())
        }
        Some(CompactErrorTolerance::RangeFraction(value)) => {
            ensure!(
                value.is_finite() && (0.0..=1.0).contains(&value),
                "range-relative error tolerance must be a finite fraction from 0 to 1"
            );
            Ok(())
        }
    }
}

fn resolve_codecs(
    compact: &CompactPlot,
    requests: &[CompactComponentCodec],
) -> Result<Vec<ResolvedComponentCodec>> {
    requests
        .iter()
        .map(|&request| {
            let max_abs_error = match request.error_tolerance {
                None => None,
                Some(CompactErrorTolerance::Absolute(value)) => Some(value),
                Some(CompactErrorTolerance::RangeFraction(fraction)) => {
                    let component_id = request.component_id as usize;
                    let stats = compact
                        .component_stats(component_id)
                        .with_context(|| format!("component {component_id} is not loaded"))?;
                    let range = match (stats.min, stats.max) {
                        (Some(min), Some(max)) => max - min,
                        (None, None) => 0.0,
                        _ => anyhow::bail!("component {component_id} has incomplete value bounds"),
                    };
                    ensure!(
                        range.is_finite() && range >= 0.0,
                        "component {component_id} global range is not finite"
                    );
                    let resolved = range * fraction;
                    ensure!(
                        resolved.is_finite(),
                        "component {component_id} resolved error tolerance is not finite"
                    );
                    Some(resolved)
                }
            };
            Ok(ResolvedComponentCodec {
                component_id: request.component_id,
                selection: request.selection,
                requested_tolerance: request.error_tolerance,
                max_abs_error,
            })
        })
        .collect()
}

fn build_active_dual_cubes_for_scalar(
    samples: &CompactScalarGrid,
    sample_regions: &[Aabb3u],
) -> SparseGrid3<()> {
    match samples {
        CompactScalarGrid::F32(grid) => build_active_dual_cubes(grid, sample_regions),
        CompactScalarGrid::F64(grid) => build_active_dual_cubes(grid, sample_regions),
        CompactScalarGrid::UNorm32 { values, .. } => {
            build_active_dual_cubes(values, sample_regions)
        }
    }
}

pub(crate) fn build_active_dual_cubes<T>(
    samples: &SparseGrid3<T>,
    sample_regions: &[Aabb3u],
) -> SparseGrid3<()>
where
    T: Copy + PartialEq,
{
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

pub(crate) fn build_chunk_value_ranges(
    samples: &CompactScalarGrid,
    eligible_cubes: &SparseGrid3<()>,
) -> Vec<ChunkValueRange> {
    eligible_cubes
        .chunk_aabbs()
        .into_iter()
        .filter_map(|aabb| {
            let sample_max = UVec3::new(
                aabb.max.x.saturating_add(1).min(samples.bounds().x),
                aabb.max.y.saturating_add(1).min(samples.bounds().y),
                aabb.max.z.saturating_add(1).min(samples.bounds().z),
            );
            let sample_aabb = Aabb3u::new(aabb.min, sample_max);
            let mut min = f64::INFINITY;
            let mut max = f64::NEG_INFINITY;
            samples.for_each_value_in_aabb(sample_aabb, |_, value| {
                min = min.min(value);
                max = max.max(value);
            });
            (min <= max).then_some(ChunkValueRange { aabb, min, max })
        })
        .collect()
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
                    scalar_types: vec![CompactComponentScalarType {
                        component_id: 0,
                        scalar_type: CompactScalarType::UNorm32 { min, max },
                    }],
                    ..CompactOptions::default()
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
    fn f64_archive_preserves_large_offset_detail_for_isosurfacing() -> Result<()> {
        let root =
            std::env::temp_dir().join(format!("amrex_rs_compact_f64_test_{}", std::process::id()));
        let _ = fs::remove_dir_all(&root);
        let base = 1.0e20;
        let step = 1.0e10;
        write_single_cube_plotfile_values(
            &root,
            std::array::from_fn(|index| base + (index % 2) as f64 * step),
        )?;

        let result = (|| -> Result<()> {
            let plotfile = PlotFile::open(&root)?;
            let mut bytes = Vec::new();
            write_compact(
                &plotfile,
                CompactOptions {
                    component_ids: vec![0],
                    scalar_types: vec![CompactComponentScalarType {
                        component_id: 0,
                        scalar_type: CompactScalarType::F64,
                    }],
                    ..CompactOptions::default()
                },
                &mut bytes,
            )?;

            let compact = read_compact(&bytes)?;
            ensure!(
                compact.component_scalar_type(0) == Some(CompactScalarType::F64),
                "f64 scalar-type metadata was not preserved"
            );
            let stats = compact.component_stats(0).context("missing component")?;
            ensure!(stats.min == Some(base), "f64 minimum lost precision");
            ensure!(stats.max == Some(base + step), "f64 maximum lost precision");

            let (mesh, _) = crate::isosurface_compact(
                &compact,
                crate::IsosurfaceOptions {
                    surface: crate::Surface {
                        id: 0,
                        value: base + 0.5 * step,
                    },
                    ..crate::IsosurfaceOptions::default()
                },
            )?;
            ensure!(
                !mesh.indices.is_empty(),
                "f64 detail did not produce a surface"
            );
            Ok(())
        })();

        let _ = fs::remove_dir_all(&root);
        result
    }

    #[test]
    fn transcode_preserves_scalar_type_and_reencodes_chunks() -> Result<()> {
        let root = std::env::temp_dir().join(format!(
            "amrex_rs_amr_transcode_test_{}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&root);
        write_single_cube_plotfile_values(&root, [0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0])?;

        let result = (|| -> Result<()> {
            let plotfile = PlotFile::open(&root)?;
            let mut source = Vec::new();
            write_compact(
                &plotfile,
                CompactOptions {
                    component_ids: vec![0],
                    scalar_types: vec![CompactComponentScalarType {
                        component_id: 0,
                        scalar_type: CompactScalarType::F64,
                    }],
                    ..CompactOptions::default()
                },
                &mut source,
            )?;

            let mut transcoded = Vec::new();
            transcode_compact(
                &source,
                TranscodeCompactOptions {
                    codecs: vec![CompactComponentCodec {
                        component_id: 0,
                        selection: CompactChunkCodecSelection::AutoSmallest,
                        error_tolerance: Some(CompactErrorTolerance::RangeFraction(0.5)),
                    }],
                },
                &mut transcoded,
            )?;

            let view = view_compact(&transcoded)?;
            ensure!(
                view.component_scalar_types()
                    .eq([(0, CompactScalarType::F64)]),
                "transcode changed the reconstructed scalar type"
            );
            let codecs = view.chunk_codec_stats()?;
            ensure!(
                codecs.len() == 1 && codecs[0].codec_id == chunk_codec::CODEC_UNIFORM,
                "transcode did not apply the requested codec policy"
            );
            let compact = read_compact(&transcoded)?;
            let stats = compact.component_stats(0).context("missing component")?;
            ensure!(stats.min == Some(0.5) && stats.max == Some(0.5));
            Ok(())
        })();

        let _ = fs::remove_dir_all(&root);
        result
    }

    #[test]
    fn adaptive_archive_reports_codec_and_respects_error() -> Result<()> {
        let root = std::env::temp_dir().join(format!(
            "amrex_rs_compact_adaptive_test_{}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&root);
        let source = [0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0];
        write_single_cube_plotfile_values(&root, source)?;

        let result = (|| -> Result<()> {
            let plotfile = PlotFile::open(&root)?;
            let mut bytes = Vec::new();
            write_compact(
                &plotfile,
                CompactOptions {
                    component_ids: vec![0],
                    scalar_types: vec![CompactComponentScalarType {
                        component_id: 0,
                        scalar_type: CompactScalarType::F64,
                    }],
                    codecs: vec![CompactComponentCodec {
                        component_id: 0,
                        selection: CompactChunkCodecSelection::AutoSmallest,
                        error_tolerance: Some(CompactErrorTolerance::RangeFraction(0.5)),
                    }],
                    ..CompactOptions::default()
                },
                &mut bytes,
            )?;

            let view = view_compact(&bytes)?;
            let codecs = view.chunk_codec_stats()?;
            ensure!(codecs.len() == 1, "unexpected adaptive codec count");
            ensure!(
                codecs[0].codec_id == chunk_codec::CODEC_UNIFORM,
                "expected approximate uniform chunk"
            );
            ensure!(codecs[0].max_abs_error <= 0.5);

            let tolerances = view.component_error_tolerances().collect::<Vec<_>>();
            ensure!(tolerances.len() == 1, "unexpected tolerance count");
            ensure!(
                tolerances[0].requested == Some(CompactErrorTolerance::RangeFraction(0.5))
                    && tolerances[0].resolved_max_abs_error == 0.5,
                "range-relative tolerance metadata changed"
            );

            let compact = read_compact(&bytes)?;
            let stats = compact.component_stats(0).context("missing component")?;
            ensure!(stats.min == Some(0.5) && stats.max == Some(0.5));
            let range = compact.levels[0].value_ranges[0][0];
            ensure!(
                range.min <= -0.5 && range.max >= 1.5,
                "isosurface range was not expanded by the error budget"
            );
            Ok(())
        })();

        let _ = fs::remove_dir_all(&root);
        result
    }

    #[test]
    fn chunk_value_ranges_include_positive_sample_halo() {
        let mut samples = SparseGrid3::new(UVec3::new(34, 2, 2));
        samples.fill_aabb(samples.bounds_aabb(), 0.0_f64);
        samples.fill_aabb(Aabb3u::new(UVec3::new(32, 0, 0), samples.bounds()), 1.0);
        let mut cubes = SparseGrid3::new(UVec3::new(33, 1, 1));
        cubes.fill_aabb(cubes.bounds_aabb(), ());
        let ranges = build_chunk_value_ranges(&CompactScalarGrid::F64(samples), &cubes);

        assert_eq!(ranges.len(), 2);
        assert_eq!((ranges[0].min, ranges[0].max), (0.0, 1.0));
        assert_eq!((ranges[1].min, ranges[1].max), (1.0, 1.0));
    }

    #[test]
    fn compaction_reports_and_flattens_non_finite_values() -> Result<()> {
        let root = std::env::temp_dir().join(format!(
            "amrex_rs_compact_nonfinite_test_{}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&root);
        write_single_cube_plotfile_values(
            &root,
            [
                f64::NAN,
                f64::INFINITY,
                f64::NEG_INFINITY,
                3.0,
                4.0,
                5.0,
                6.0,
                7.0,
            ],
        )?;

        let result = (|| -> Result<()> {
            let plotfile = PlotFile::open(&root)?;
            let mut bytes = Vec::new();
            let written = write_compact(
                &plotfile,
                CompactOptions {
                    component_ids: vec![0],
                    scalar_types: vec![CompactComponentScalarType {
                        component_id: 0,
                        scalar_type: CompactScalarType::F64,
                    }],
                    ..CompactOptions::default()
                },
                &mut bytes,
            )?;
            ensure!(
                written.non_finite_replacements
                    == vec![NonFiniteReplacement {
                        component_id: 0,
                        count: 3,
                    }],
                "unexpected non-finite replacement report"
            );
            let stats = read_compact(&bytes)?
                .component_stats(0)
                .context("missing component")?;
            ensure!(stats.nan_count == 0, "non-finite value survived compaction");
            ensure!(stats.min == Some(0.0) && stats.max == Some(7.0));
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
            ensure!(
                bytes[..4] == FORMAT_MAGIC.to_le_bytes()
                    && bytes[4..8] == FORMAT_VERSION.to_le_bytes(),
                "compact archive prefix changed"
            );
            let header_offset = u64::from_le_bytes(bytes[bytes.len() - 8..].try_into().unwrap());
            ensure!(
                header_offset.is_multiple_of(8),
                "compact header is not aligned"
            );

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

            let selected = read_compact_selected(&bytes, &[0])?;
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
