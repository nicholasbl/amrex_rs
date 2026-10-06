//! Triangle-mesh decimation backed by meshoptimizer.
//!
//! This module keeps the native meshoptimizer interface private and exposes
//! operations directly on [`Mesh3D`].

use std::{
    mem, ptr,
    time::{Duration, Instant},
};

use anyhow::{Context, Result, ensure};
use meshopt::SimplifyOptions;

use crate::{
    DedupMeshOptions, Mesh3D, RemoveDegenerateTrianglesOptions, dedup_mesh_vertices,
    remove_degenerate_triangles,
};

/// Desired final mesh size.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum DecimateTarget {
    /// Attempt to reduce the mesh to this many triangles.
    FaceCount(usize),
    /// Attempt to retain this fraction of the input triangles, in `(0, 1]`.
    FaceRatio(f32),
}

/// Interpretation of [`DecimateOptions::max_error`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DecimateErrorMode {
    /// Error relative to the extent of the mesh. `0.01` permits roughly 1%.
    Relative,
    /// Error in the same physical-space units as vertex positions.
    Absolute,
}

/// Options controlling mesh decimation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DecimateOptions {
    /// Requested final mesh size.
    pub target: DecimateTarget,
    /// Maximum meshoptimizer simplification error.
    ///
    /// This is a quadric-derived error estimate, not a strict Hausdorff bound.
    pub max_error: f32,
    /// Whether `max_error` is relative to mesh extent or in physical units.
    pub error_mode: DecimateErrorMode,
    /// Relative importance of the two UV/scientific quantity channels.
    ///
    /// A zero weight excludes that channel from simplification and prevents
    /// meshoptimizer from updating it. Ignored when the mesh has no UV buffer.
    pub uv_weights: [f32; 2],
    /// Keep vertices on open mesh boundaries fixed.
    pub lock_boundaries: bool,
    /// Prefer more regular triangle shapes at some geometric-quality cost.
    pub regularize: bool,
}

impl Default for DecimateOptions {
    fn default() -> Self {
        Self {
            target: DecimateTarget::FaceRatio(0.5),
            max_error: 0.01,
            error_mode: DecimateErrorMode::Relative,
            uv_weights: [1.0, 1.0],
            lock_boundaries: true,
            regularize: false,
        }
    }
}

/// Summary of a decimation operation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DecimateResult {
    /// Triangle count before decimation.
    pub original_face_count: usize,
    /// Triangle count after decimation.
    pub final_face_count: usize,
    /// Vertex count before decimation.
    pub original_vertex_count: usize,
    /// Referenced vertex count after decimation and compaction.
    pub final_vertex_count: usize,
    /// Error reported by meshoptimizer, in the requested error mode.
    pub error: f32,
    /// True when the requested face count was reached.
    pub reached_target: bool,
    /// Time spent in each stage of this decimation call.
    pub timings: DecimateTimings,
}

/// Stage timings for [`decimate_mesh`].
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct DecimateTimings {
    /// Validation of the input mesh and decimation options.
    pub input_validation: Duration,
    /// The meshoptimizer simplification call and result-size checks.
    ///
    /// This is zero when the input is empty or already meets the target.
    pub simplification: Duration,
    /// Validation of meshoptimizer's output.
    ///
    /// This is zero when meshoptimizer was not called.
    pub output_validation: Duration,
    /// Removal and remapping of unreferenced vertices after simplification.
    pub compaction: Duration,
    /// Wall-clock duration of the complete decimation call.
    pub total: Duration,
}

/// Options for welding, cleaning, and decimating a mesh in one call.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DecimatePipelineOptions {
    /// Tolerances used to weld vertices before simplification.
    pub dedup: DedupMeshOptions,
    /// Threshold used to discard degenerate faces after welding.
    pub remove_degenerate: RemoveDegenerateTrianglesOptions,
    /// Options passed to the decimation stage.
    pub decimate: DecimateOptions,
}

impl Default for DecimatePipelineOptions {
    fn default() -> Self {
        Self {
            dedup: DedupMeshOptions {
                position_epsilon: 1e-6,
                uv_epsilon: 1e-6,
            },
            remove_degenerate: RemoveDegenerateTrianglesOptions { area_epsilon: 0.0 },
            decimate: DecimateOptions::default(),
        }
    }
}

/// Summary of the complete cleanup and decimation pipeline.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DecimatePipelineResult {
    /// Triangle count before any cleanup.
    pub original_face_count: usize,
    /// Vertex count before any cleanup.
    pub original_vertex_count: usize,
    /// Number of vertices removed by welding.
    pub removed_duplicate_vertices: usize,
    /// Number of faces removed after welding because they were degenerate.
    pub removed_degenerate_faces: usize,
    /// Number of unreferenced vertices removed before decimation.
    pub removed_unreferenced_vertices: usize,
    /// Result of the final decimation stage.
    pub decimation: DecimateResult,
    /// Time spent in each stage of the complete pipeline.
    pub timings: DecimatePipelineTimings,
}

/// Stage timings for [`decimate_mesh_pipeline`].
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct DecimatePipelineTimings {
    /// Approximate vertex welding by position and UV.
    pub deduplication: Duration,
    /// Removal of repeated-index and small-area triangles.
    pub degenerate_removal: Duration,
    /// Removal and remapping of unreferenced vertices before decimation.
    pub pre_decimation_compaction: Duration,
    /// The complete [`decimate_mesh`] call.
    pub decimation: Duration,
    /// Wall-clock duration of the complete cleanup and decimation pipeline.
    pub total: Duration,
}

/// Decimate an indexed triangle mesh in place.
///
/// Meshoptimizer attempts to preserve topology and may stop before the target
/// when constrained by topology or `max_error`. Surviving vertices are compacted
/// after simplification, and UV values are updated alongside positions when
/// their corresponding weight is nonzero.
pub fn decimate_mesh(mesh: &mut Mesh3D, options: DecimateOptions) -> Result<DecimateResult> {
    let total_start = Instant::now();
    let stage_start = Instant::now();
    validate_mesh(mesh)?;
    validate_options(options)?;
    let input_validation = stage_start.elapsed();

    let original_face_count = mesh.indices.len();
    let original_vertex_count = mesh.positions.len();
    if original_face_count == 0 {
        let stage_start = Instant::now();
        compact_vertices(mesh)?;
        let compaction = stage_start.elapsed();
        let final_vertex_count = mesh.positions.len();
        return Ok(DecimateResult {
            original_face_count,
            final_face_count: 0,
            original_vertex_count,
            final_vertex_count,
            error: 0.0,
            reached_target: true,
            timings: DecimateTimings {
                input_validation,
                compaction,
                total: total_start.elapsed(),
                ..DecimateTimings::default()
            },
        });
    }

    let target_face_count = resolve_target(options.target, original_face_count)?;
    if target_face_count >= original_face_count {
        let stage_start = Instant::now();
        compact_vertices(mesh)?;
        let compaction = stage_start.elapsed();
        return Ok(DecimateResult {
            original_face_count,
            final_face_count: original_face_count,
            original_vertex_count,
            final_vertex_count: mesh.positions.len(),
            error: 0.0,
            reached_target: true,
            timings: DecimateTimings {
                input_validation,
                compaction,
                total: total_start.elapsed(),
                ..DecimateTimings::default()
            },
        });
    }

    let mut simplify_options = SimplifyOptions::empty();
    if options.lock_boundaries {
        simplify_options.insert(SimplifyOptions::LockBorder);
    }
    if options.regularize {
        simplify_options.insert(SimplifyOptions::Regularize);
    }
    if options.error_mode == DecimateErrorMode::Absolute {
        simplify_options.insert(SimplifyOptions::ErrorAbsolute);
    }

    let use_uv = !mesh.uv.is_empty();
    let (attributes, attribute_stride, weights, attribute_count) = if use_uv {
        (
            mesh.uv.as_mut_ptr().cast::<f32>(),
            mem::size_of::<[f32; 2]>(),
            options.uv_weights.as_ptr(),
            options.uv_weights.len(),
        )
    } else {
        (ptr::null_mut(), 0, ptr::null(), 0)
    };

    let mut result_error = 0.0;
    let index_count = mesh.indices.len() * 3;
    let target_index_count = target_face_count
        .checked_mul(3)
        .context("target index count overflowed usize")?;

    // SAFETY: `validate_mesh` ensures that every index is in bounds and that
    // position/UV buffers have valid lengths and finite values. Arrays of
    // `[u32; 3]`, `[f32; 3]`, and `[f32; 2]` are contiguous, and all pointers
    // remain valid for the duration of this call. Counts and strides exactly
    // match their respective buffers. The optional lock pointer is null.
    let stage_start = Instant::now();
    let simplified_index_count = unsafe {
        meshopt::ffi::meshopt_simplifyWithUpdate(
            mesh.indices.as_mut_ptr().cast::<u32>(),
            index_count,
            mesh.positions.as_mut_ptr().cast::<f32>(),
            mesh.positions.len(),
            mem::size_of::<[f32; 3]>(),
            attributes,
            attribute_stride,
            weights,
            attribute_count,
            ptr::null(),
            target_index_count,
            options.max_error,
            simplify_options.bits(),
            &mut result_error,
        )
    };

    ensure!(
        simplified_index_count <= index_count && simplified_index_count % 3 == 0,
        "meshoptimizer returned an invalid index count {simplified_index_count}"
    );
    mesh.indices.truncate(simplified_index_count / 3);
    let simplification = stage_start.elapsed();

    let stage_start = Instant::now();
    validate_mesh(mesh).context("validating meshoptimizer output")?;
    let output_validation = stage_start.elapsed();

    let stage_start = Instant::now();
    compact_vertices(mesh)?;
    let compaction = stage_start.elapsed();

    let final_face_count = mesh.indices.len();
    Ok(DecimateResult {
        original_face_count,
        final_face_count,
        original_vertex_count,
        final_vertex_count: mesh.positions.len(),
        error: result_error,
        reached_target: final_face_count <= target_face_count,
        timings: DecimateTimings {
            input_validation,
            simplification,
            output_validation,
            compaction,
            total: total_start.elapsed(),
        },
    })
}

/// Weld, clean, and decimate an indexed triangle mesh in place.
///
/// The pipeline performs vertex deduplication, removes degenerate triangles,
/// removes vertices no longer referenced by a triangle, and finally decimates
/// and compacts the mesh.
pub fn decimate_mesh_pipeline(
    mesh: &mut Mesh3D,
    options: DecimatePipelineOptions,
) -> Result<DecimatePipelineResult> {
    let total_start = Instant::now();
    let original_face_count = mesh.indices.len();
    let original_vertex_count = mesh.positions.len();

    let stage_start = Instant::now();
    let removed_duplicate_vertices =
        dedup_mesh_vertices(mesh, options.dedup).context("deduplicating mesh before decimation")?;
    let deduplication = stage_start.elapsed();

    let stage_start = Instant::now();
    let removed_degenerate_faces = remove_degenerate_triangles(mesh, options.remove_degenerate)
        .context("removing degenerate triangles before decimation")?;
    let degenerate_removal = stage_start.elapsed();

    let stage_start = Instant::now();
    let removed_unreferenced_vertices = compact_vertices(mesh)?;
    let pre_decimation_compaction = stage_start.elapsed();

    let stage_start = Instant::now();
    let decimation = decimate_mesh(mesh, options.decimate)?;
    let decimation_time = stage_start.elapsed();

    Ok(DecimatePipelineResult {
        original_face_count,
        original_vertex_count,
        removed_duplicate_vertices,
        removed_degenerate_faces,
        removed_unreferenced_vertices,
        decimation,
        timings: DecimatePipelineTimings {
            deduplication,
            degenerate_removal,
            pre_decimation_compaction,
            decimation: decimation_time,
            total: total_start.elapsed(),
        },
    })
}

fn validate_mesh(mesh: &Mesh3D) -> Result<()> {
    ensure!(
        mesh.uv.is_empty() || mesh.uv.len() == mesh.positions.len(),
        "mesh uv buffer must be empty or match position count"
    );
    for (index, position) in mesh.positions.iter().enumerate() {
        ensure!(
            position.iter().all(|value| value.is_finite()),
            "mesh position {index} contains a non-finite component"
        );
    }
    for (index, uv) in mesh.uv.iter().enumerate() {
        ensure!(
            uv.iter().all(|value| value.is_finite()),
            "mesh uv {index} contains a non-finite component"
        );
    }
    for (face_index, face) in mesh.indices.iter().enumerate() {
        for &vertex_index in face {
            ensure!(
                (vertex_index as usize) < mesh.positions.len(),
                "mesh face {face_index} index {vertex_index} is out of bounds"
            );
        }
    }
    Ok(())
}

fn validate_options(options: DecimateOptions) -> Result<()> {
    ensure!(
        options.max_error.is_finite() && options.max_error >= 0.0,
        "maximum decimation error must be finite and non-negative"
    );
    if options.error_mode == DecimateErrorMode::Relative {
        ensure!(
            options.max_error <= 1.0,
            "relative maximum decimation error must not exceed 1"
        );
    }
    ensure!(
        options
            .uv_weights
            .iter()
            .all(|weight| weight.is_finite() && *weight >= 0.0),
        "UV weights must be finite and non-negative"
    );
    match options.target {
        DecimateTarget::FaceCount(count) => {
            ensure!(count > 0, "target face count must be positive");
        }
        DecimateTarget::FaceRatio(ratio) => ensure!(
            ratio.is_finite() && ratio > 0.0 && ratio <= 1.0,
            "target face ratio must be finite and in (0, 1]"
        ),
    }
    Ok(())
}

fn resolve_target(target: DecimateTarget, face_count: usize) -> Result<usize> {
    match target {
        DecimateTarget::FaceCount(count) => Ok(count),
        DecimateTarget::FaceRatio(ratio) => {
            let count = ((face_count as f64) * f64::from(ratio)).floor();
            ensure!(count <= usize::MAX as f64, "target face count is too large");
            Ok((count as usize).max(1))
        }
    }
}

fn compact_vertices(mesh: &mut Mesh3D) -> Result<usize> {
    let old_vertex_count = mesh.positions.len();
    if mesh.indices.is_empty() {
        mesh.positions.clear();
        mesh.uv.clear();
        return Ok(old_vertex_count);
    }

    let use_uv = !mesh.uv.is_empty();
    let mut remap = vec![u32::MAX; old_vertex_count];
    let mut positions = Vec::new();
    let mut uv = Vec::new();

    for (face_index, face) in mesh.indices.iter_mut().enumerate() {
        for vertex_index in face {
            let old_index = *vertex_index as usize;
            let mapped = remap.get_mut(old_index).with_context(|| {
                format!("mesh face {face_index} index {old_index} is out of bounds")
            })?;
            if *mapped == u32::MAX {
                *mapped =
                    u32::try_from(positions.len()).context("mesh vertex count exceeds u32")?;
                positions.push(mesh.positions[old_index]);
                if use_uv {
                    uv.push(mesh.uv[old_index]);
                }
            }
            *vertex_index = *mapped;
        }
    }

    let removed = old_vertex_count - positions.len();
    mesh.positions = positions;
    mesh.uv = uv;
    Ok(removed)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn planar_grid() -> Mesh3D {
        let mut positions = Vec::new();
        let mut uv = Vec::new();
        for y in 0..=4 {
            for x in 0..=4 {
                positions.push([x as f32, y as f32, 0.0]);
                uv.push([x as f32 / 4.0, y as f32 / 4.0]);
            }
        }

        let mut indices = Vec::new();
        for y in 0..4 {
            for x in 0..4 {
                let a = y * 5 + x;
                let b = a + 1;
                let c = a + 5;
                let d = c + 1;
                indices.push([a, b, d]);
                indices.push([a, d, c]);
            }
        }

        Mesh3D {
            positions,
            uv,
            indices,
        }
    }

    #[test]
    fn decimates_and_compacts_mesh() -> Result<()> {
        let mut mesh = planar_grid();
        let result = decimate_mesh(
            &mut mesh,
            DecimateOptions {
                target: DecimateTarget::FaceRatio(0.5),
                max_error: 1.0,
                lock_boundaries: false,
                ..DecimateOptions::default()
            },
        )?;

        ensure!(result.final_face_count < result.original_face_count);
        ensure!(result.final_vertex_count <= result.original_vertex_count);
        ensure!(result.timings.total >= result.timings.input_validation);
        ensure!(result.timings.total >= result.timings.simplification);
        ensure!(result.timings.total >= result.timings.output_validation);
        ensure!(result.timings.total >= result.timings.compaction);
        ensure!(mesh.uv.len() == mesh.positions.len());
        ensure!(mesh.indices.iter().flatten().all(|&index| {
            usize::try_from(index).is_ok_and(|index| index < mesh.positions.len())
        }));
        Ok(())
    }

    #[test]
    fn pipeline_welds_and_removes_degenerate_faces() -> Result<()> {
        let mut mesh = Mesh3D {
            positions: vec![
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0],
            ],
            uv: vec![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 0.0]],
            indices: vec![[0, 1, 2], [0, 3, 1]],
        };

        let result = decimate_mesh_pipeline(
            &mut mesh,
            DecimatePipelineOptions {
                dedup: DedupMeshOptions {
                    position_epsilon: 1e-6,
                    uv_epsilon: 1e-6,
                },
                remove_degenerate: RemoveDegenerateTrianglesOptions { area_epsilon: 0.0 },
                decimate: DecimateOptions {
                    target: DecimateTarget::FaceCount(1),
                    ..DecimateOptions::default()
                },
            },
        )?;

        ensure!(result.original_face_count == 2);
        ensure!(result.original_vertex_count == 4);
        ensure!(result.removed_duplicate_vertices == 1);
        ensure!(result.removed_degenerate_faces == 1);
        ensure!(result.timings.total >= result.timings.deduplication);
        ensure!(result.timings.total >= result.timings.degenerate_removal);
        ensure!(result.timings.total >= result.timings.pre_decimation_compaction);
        ensure!(result.timings.total >= result.timings.decimation);
        ensure!(result.timings.decimation >= result.decimation.timings.total);
        ensure!(mesh.positions.len() == 3);
        ensure!(mesh.indices == vec![[0, 1, 2]]);
        Ok(())
    }

    #[test]
    fn decimates_mesh_without_uvs() -> Result<()> {
        let mut mesh = planar_grid();
        mesh.uv.clear();
        decimate_mesh(
            &mut mesh,
            DecimateOptions {
                target: DecimateTarget::FaceCount(8),
                max_error: 1.0,
                lock_boundaries: false,
                ..DecimateOptions::default()
            },
        )?;

        ensure!(mesh.uv.is_empty());
        ensure!(mesh.indices.len() <= 8);
        Ok(())
    }

    #[test]
    fn empty_mesh_is_compacted_without_calling_meshoptimizer() -> Result<()> {
        let mut mesh = Mesh3D {
            positions: vec![[0.0, 0.0, 0.0]],
            uv: vec![[0.0, 0.0]],
            indices: Vec::new(),
        };
        let result = decimate_mesh(&mut mesh, DecimateOptions::default())?;

        ensure!(mesh.positions.is_empty());
        ensure!(mesh.uv.is_empty());
        ensure!(result.original_vertex_count == 1);
        ensure!(result.final_vertex_count == 0);
        ensure!(result.timings.simplification == Duration::ZERO);
        ensure!(result.timings.output_validation == Duration::ZERO);
        Ok(())
    }
}
