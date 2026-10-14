use anyhow::{Result, ensure};
use glam::{I64Vec3, UVec3};

use crate::utility::{Aabb3u, GridAccessor, SparseGrid3};

/// Scalar type reconstructed when a compact quantity is loaded.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub enum CompactScalarType {
    /// Reconstruct IEEE-754 single-precision values.
    #[default]
    F32,
    /// Reconstruct IEEE-754 double-precision values.
    F64,
    /// Quantize one global physical range uniformly over all `u32` codes.
    UNorm32 { min: f64, max: f64 },
}

/// Per-component reconstructed scalar-type override.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CompactComponentScalarType {
    /// Plotfile component id.
    pub component_id: u32,
    /// Scalar type used by every AMR level for this component.
    pub scalar_type: CompactScalarType,
}

/// A concrete on-disk chunk codec.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompactChunkCodec {
    UniformF64,
    DenseF32,
    DenseF64,
    DenseU32,
    QuantizedU8,
    QuantizedU16,
    QuantizedU32,
    AffineF64,
    AffineResidualU8,
    AffineResidualU16,
    AffineResidualU32,
}

/// How the writer chooses a codec for dense chunks.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum CompactChunkCodecSelection {
    /// Use exact uniform chunks and the dense codec matching the scalar type.
    #[default]
    AutoLossless,
    /// Select the smallest codec satisfying the configured error tolerance.
    AutoSmallest,
    /// Require one codec for every non-uniform chunk.
    Forced(CompactChunkCodec),
}

/// Maximum pointwise error accepted from a lossy chunk codec.
///
/// For every physical source value `x` and reconstructed value `x_hat`, the
/// codec must satisfy
/// `abs(x_hat - x) <= absolute + relative_percent / 100 * abs(x)`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CompactErrorTolerance {
    /// Absolute error floor in the quantity's original physical units.
    pub absolute: f64,
    /// Error permitted relative to each source value, expressed as a percentage.
    pub relative_percent: f64,
}

/// Per-component chunk-codec policy.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CompactComponentCodec {
    /// Plotfile component id.
    pub component_id: u32,
    /// Automatic or forced codec selection.
    pub selection: CompactChunkCodecSelection,
    /// Required for lossy automatic or forced codecs.
    pub error_tolerance: Option<CompactErrorTolerance>,
}

/// Policy for non-finite plotfile samples encountered during compaction.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum NonFinitePolicy {
    /// Replace non-finite values with physical zero and report aggregate counts.
    #[default]
    ReplaceWithZero,
    /// Skip finiteness checks. The caller guarantees that all values are finite
    /// and representable by their selected scalar types.
    AssumeFinite,
}

/// Aggregate count of non-finite or unrepresentable values replaced for one component.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct NonFiniteReplacement {
    pub component_id: u32,
    pub count: usize,
}

#[derive(Debug, Clone)]
pub(crate) enum CompactScalarGrid {
    F32(SparseGrid3<f32>),
    F64(SparseGrid3<f64>),
    UNorm32 {
        values: SparseGrid3<u32>,
        min: f64,
        max: f64,
    },
}

pub(crate) enum CompactScalarAccessor<'a> {
    F32(GridAccessor<'a, f32>),
    F64(GridAccessor<'a, f64>),
    UNorm32(GridAccessor<'a, u32>),
}

impl CompactScalarType {
    pub(crate) fn validate(self) -> Result<()> {
        if let Self::UNorm32 { min, max } = self {
            ensure!(min.is_finite(), "UNorm32 minimum must be finite");
            ensure!(max.is_finite(), "UNorm32 maximum must be finite");
            ensure!(max > min, "UNorm32 maximum must exceed its minimum");
            ensure!((max - min).is_finite(), "UNorm32 range must be finite");
        }
        Ok(())
    }
}

impl CompactScalarGrid {
    pub(crate) fn new(
        bounds: UVec3,
        chunk_capacity: usize,
        scalar_type: CompactScalarType,
    ) -> Self {
        match scalar_type {
            CompactScalarType::F32 => {
                Self::F32(SparseGrid3::with_chunk_capacity(bounds, chunk_capacity))
            }
            CompactScalarType::F64 => {
                Self::F64(SparseGrid3::with_chunk_capacity(bounds, chunk_capacity))
            }
            CompactScalarType::UNorm32 { min, max } => Self::UNorm32 {
                values: SparseGrid3::with_chunk_capacity(bounds, chunk_capacity),
                min,
                max,
            },
        }
    }

    pub(crate) fn bounds(&self) -> UVec3 {
        match self {
            Self::F32(grid) => grid.bounds(),
            Self::F64(grid) => grid.bounds(),
            Self::UNorm32 { values, .. } => values.bounds(),
        }
    }

    pub(crate) fn bounds_aabb(&self) -> Aabb3u {
        Aabb3u::full_grid(self.bounds())
    }

    pub(crate) fn chunk_count(&self) -> usize {
        match self {
            Self::F32(grid) => grid.chunk_count(),
            Self::F64(grid) => grid.chunk_count(),
            Self::UNorm32 { values, .. } => values.chunk_count(),
        }
    }

    pub(crate) fn accessor(&self) -> CompactScalarAccessor<'_> {
        match self {
            Self::F32(grid) => CompactScalarAccessor::F32(grid.accessor()),
            Self::F64(grid) => CompactScalarAccessor::F64(grid.accessor()),
            Self::UNorm32 { values, .. } => CompactScalarAccessor::UNorm32(values.accessor()),
        }
    }

    pub(crate) fn mask_out_by_presence_scaled(
        &self,
        coarse_mask: &mut SparseGrid3<()>,
        scale: u32,
        translation: I64Vec3,
    ) {
        match self {
            Self::F32(grid) => {
                coarse_mask.mask_out_by_presence_scaled(grid, scale, translation);
            }
            Self::F64(grid) => {
                coarse_mask.mask_out_by_presence_scaled(grid, scale, translation);
            }
            Self::UNorm32 { values, .. } => {
                coarse_mask.mask_out_by_presence_scaled(values, scale, translation);
            }
        }
    }

    pub(crate) fn write_physical_values(
        &mut self,
        aabb: Aabb3u,
        source: &[f64],
        policy: NonFinitePolicy,
    ) -> Result<usize> {
        let mut replaced = 0;
        match self {
            Self::F32(grid) => {
                let values = source
                    .iter()
                    .map(|&value| {
                        let value = sanitize(value, policy, &mut replaced);
                        let value = value as f32;
                        if policy == NonFinitePolicy::ReplaceWithZero && !value.is_finite() {
                            replaced += 1;
                            0.0
                        } else {
                            value
                        }
                    })
                    .collect::<Vec<_>>();
                grid.write_aabb_values(aabb, &values)
                    .map_err(|error| anyhow::anyhow!("writing f32 sparse values: {error:?}"))?;
            }
            Self::F64(grid) => {
                let values = source
                    .iter()
                    .map(|&value| sanitize(value, policy, &mut replaced))
                    .collect::<Vec<_>>();
                grid.write_aabb_values(aabb, &values)
                    .map_err(|error| anyhow::anyhow!("writing f64 sparse values: {error:?}"))?;
            }
            Self::UNorm32 { values, min, max } => {
                let scale = f64::from(u32::MAX);
                let quantized = source
                    .iter()
                    .map(|&value| {
                        let value = sanitize(value, policy, &mut replaced);
                        (((value - *min) / (*max - *min)).clamp(0.0, 1.0) * scale).round() as u32
                    })
                    .collect::<Vec<_>>();
                values
                    .write_aabb_values(aabb, &quantized)
                    .map_err(|error| anyhow::anyhow!("writing UNorm32 sparse values: {error:?}"))?;
            }
        }
        Ok(replaced)
    }

    pub(crate) fn for_each_value_in_aabb(&self, aabb: Aabb3u, mut visitor: impl FnMut(UVec3, f64)) {
        match self {
            Self::F32(grid) => grid
                .for_each_present_in_aabb(aabb, |position, value| visitor(position, value.into())),
            Self::F64(grid) => grid.for_each_present_in_aabb(aabb, &mut visitor),
            Self::UNorm32 { values, .. } => values
                .for_each_present_in_aabb(aabb, |position, value| {
                    visitor(position, f64::from(value) / f64::from(u32::MAX))
                }),
        }
    }
}

impl CompactScalarAccessor<'_> {
    #[inline]
    pub(crate) fn get(&mut self, position: UVec3) -> Option<f64> {
        match self {
            Self::F32(accessor) => accessor.get(position).map(f64::from),
            Self::F64(accessor) => accessor.get(position),
            Self::UNorm32(accessor) => accessor
                .get(position)
                .map(|value| f64::from(value) / f64::from(u32::MAX)),
        }
    }
}

fn sanitize(value: f64, policy: NonFinitePolicy, replaced: &mut usize) -> f64 {
    if policy == NonFinitePolicy::ReplaceWithZero && !value.is_finite() {
        *replaced += 1;
        0.0
    } else {
        value
    }
}
