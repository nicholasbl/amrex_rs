//! MC33 isosurface extraction for AMReX plotfiles and compact archives.
//!
//! The extractor operates on cell-centered scalar data. Optional sampled
//! quantities are interpolated onto vertices and emitted as normalized texture
//! coordinate channels.

mod dual_grid;
mod mc33;
mod sampling;

use std::collections::HashMap;
use std::ops::RangeInclusive;

use anyhow::{Context, Result, ensure};
use rayon::prelude::*;

use crate::PlotFile;
use crate::compact::CompactPlot;
use crate::utility::Aabb3u;

use dual_grid::{dual_grid_levels_from_compact, load_compact_for_isosurface};

/// Scalar component and threshold used to extract a surface.
#[derive(Debug, Default)]
pub struct Surface {
    /// Component id in the plotfile variable list.
    pub id: u32,
    /// Isovalue in the component's native units.
    pub value: f64,
}

/// Auxiliary scalar component to sample onto generated vertices.
#[derive(Debug)]
pub struct Sample {
    /// Component id in the plotfile variable list.
    pub id: u32,
    /// Value range mapped to `[0, 1]` for texture coordinates.
    pub range: RangeInclusive<f64>,
}

/// Options controlling isosurface extraction.
#[derive(Debug, Default)]
pub struct IsosurfaceOptions {
    /// Surface component and threshold.
    pub surface: Surface,
    /// Up to two additional components sampled into vertex `uv`.
    pub sampled_quantities: Vec<Sample>,
    /// Optional inclusive AMR level range to extract.
    pub levels: Option<RangeInclusive<usize>>,
    /// Flip all emitted triangle winding if set.
    pub flip_winding: bool,
}

/// Triangle mesh produced by isosurface extraction.
#[derive(Debug, Default)]
pub struct Mesh3D {
    /// Physical-space vertex positions.
    pub positions: Vec<[f32; 3]>,
    /// Per-vertex texture coordinates. Indices match `positions`.
    pub uv: Vec<[f32; 2]>,
    /// Triangle vertex indices.
    pub indices: Vec<[u32; 3]>,
}

/// Thresholds used when merging duplicate mesh vertices.
#[derive(Debug, Clone, Copy)]
pub struct DedupMeshOptions {
    /// Maximum Euclidean distance between positions for vertices to merge.
    pub position_epsilon: f32,
    /// Maximum Euclidean distance between UVs for vertices to merge.
    ///
    /// This is ignored when `mesh.uv` is empty.
    pub uv_epsilon: f32,
}

/// Options controlling removal of degenerate triangles.
#[derive(Debug, Clone, Copy)]
pub struct RemoveDegenerateTrianglesOptions {
    /// Minimum triangle area to keep.
    ///
    /// Triangles with repeated indices are always removed. Triangles with area
    /// less than or equal to this threshold are removed after validating their
    /// indices and positions.
    pub area_epsilon: f32,
}

/// Merge vertices whose positions and UVs are within the configured thresholds.
///
/// The operation is in-place and returns the number of removed vertices. Meshes
/// with an empty UV buffer are deduplicated by position only. Meshes with a
/// non-empty UV buffer must have one UV per position.
pub fn dedup_mesh_vertices(mesh: &mut Mesh3D, options: DedupMeshOptions) -> Result<usize> {
    ensure!(
        options.position_epsilon.is_finite() && options.position_epsilon > 0.0,
        "position_epsilon must be finite and positive"
    );
    ensure!(
        options.uv_epsilon.is_finite() && options.uv_epsilon > 0.0,
        "uv_epsilon must be finite and positive"
    );
    ensure!(
        mesh.uv.is_empty() || mesh.uv.len() == mesh.positions.len(),
        "mesh uv buffer must be empty or match position count"
    );

    if mesh.positions.is_empty() {
        return Ok(0);
    }

    let use_uv = !mesh.uv.is_empty();
    let mut remap = vec![0u32; mesh.positions.len()];
    let mut positions = Vec::with_capacity(mesh.positions.len());
    let mut uv = Vec::with_capacity(mesh.uv.len());
    let mut buckets: HashMap<[i64; 5], Vec<u32>> = HashMap::new();
    let position_epsilon_squared = options.position_epsilon * options.position_epsilon;
    let uv_epsilon_squared = options.uv_epsilon * options.uv_epsilon;

    for (old_index, &position) in mesh.positions.iter().enumerate() {
        let old_uv = use_uv.then(|| mesh.uv[old_index]);
        ensure!(
            position.iter().all(|component| component.is_finite()),
            "mesh position {old_index} contains a non-finite component"
        );
        ensure!(
            old_uv.is_none_or(|uv| uv.iter().all(|component| component.is_finite())),
            "mesh uv {old_index} contains a non-finite component"
        );
        let key = dedup_bucket_key(position, old_uv, options);
        let mut merged = None;

        visit_neighboring_dedup_keys(key, use_uv, |neighbor_key| {
            let Some(candidates) = buckets.get(&neighbor_key) else {
                return true;
            };
            for &candidate in candidates {
                let candidate_index = candidate as usize;
                if position_distance_squared(position, positions[candidate_index])
                    <= position_epsilon_squared
                    && (!use_uv
                        || uv_distance_squared(old_uv.unwrap(), uv[candidate_index])
                            <= uv_epsilon_squared)
                {
                    merged = Some(candidate);
                    return false;
                }
            }
            true
        });

        let new_index = match merged {
            Some(index) => index,
            None => {
                let index =
                    u32::try_from(positions.len()).context("mesh vertex count exceeds u32")?;
                positions.push(position);
                if let Some(uv_value) = old_uv {
                    uv.push(uv_value);
                }
                buckets.entry(key).or_default().push(index);
                index
            }
        };
        remap[old_index] = new_index;
    }

    mesh.indices
        .par_iter_mut()
        .try_for_each(|face| -> Result<()> {
            for index in face {
                let old_index =
                    usize::try_from(*index).context("mesh index does not fit in usize")?;
                *index = *remap
                    .get(old_index)
                    .with_context(|| format!("mesh index {old_index} is out of bounds"))?;
            }
            Ok(())
        })?;

    let removed = mesh.positions.len() - positions.len();
    mesh.positions = positions;
    mesh.uv = uv;
    Ok(removed)
}

/// Remove triangles with repeated indices or near-zero physical area.
///
/// The operation is in-place and returns the number of removed triangles.
pub fn remove_degenerate_triangles(
    mesh: &mut Mesh3D,
    options: RemoveDegenerateTrianglesOptions,
) -> Result<usize> {
    ensure!(
        options.area_epsilon.is_finite() && options.area_epsilon >= 0.0,
        "area_epsilon must be finite and non-negative"
    );

    let area_epsilon_squared = options.area_epsilon * options.area_epsilon;
    let mut kept = Vec::with_capacity(mesh.indices.len());
    for (face_index, &face) in mesh.indices.iter().enumerate() {
        if has_repeated_index(face) {
            continue;
        }

        let a = position_for_face_vertex(mesh, face_index, face[0])?;
        let b = position_for_face_vertex(mesh, face_index, face[1])?;
        let c = position_for_face_vertex(mesh, face_index, face[2])?;
        if triangle_area_squared(a, b, c) <= area_epsilon_squared {
            continue;
        }
        kept.push(face);
    }

    let removed = mesh.indices.len() - kept.len();
    mesh.indices = kept;
    Ok(removed)
}

fn dedup_bucket_key(
    position: [f32; 3],
    uv: Option<[f32; 2]>,
    options: DedupMeshOptions,
) -> [i64; 5] {
    let uv = uv.unwrap_or([0.0, 0.0]);
    [
        bucket_coord(position[0], options.position_epsilon),
        bucket_coord(position[1], options.position_epsilon),
        bucket_coord(position[2], options.position_epsilon),
        bucket_coord(uv[0], options.uv_epsilon),
        bucket_coord(uv[1], options.uv_epsilon),
    ]
}

fn bucket_coord(value: f32, epsilon: f32) -> i64 {
    (value / epsilon).floor() as i64
}

fn visit_neighboring_dedup_keys<F>(key: [i64; 5], use_uv: bool, mut visitor: F)
where
    F: FnMut([i64; 5]) -> bool,
{
    for dz in -1..=1 {
        for dy in -1..=1 {
            for dx in -1..=1 {
                if use_uv {
                    for dv in -1..=1 {
                        for du in -1..=1 {
                            if !visitor([
                                key[0] + dx,
                                key[1] + dy,
                                key[2] + dz,
                                key[3] + du,
                                key[4] + dv,
                            ]) {
                                return;
                            }
                        }
                    }
                } else if !visitor([key[0] + dx, key[1] + dy, key[2] + dz, key[3], key[4]]) {
                    return;
                }
            }
        }
    }
}

fn position_distance_squared(a: [f32; 3], b: [f32; 3]) -> f32 {
    let dx = a[0] - b[0];
    let dy = a[1] - b[1];
    let dz = a[2] - b[2];
    dx.mul_add(dx, dy.mul_add(dy, dz * dz))
}

fn uv_distance_squared(a: [f32; 2], b: [f32; 2]) -> f32 {
    let du = a[0] - b[0];
    let dv = a[1] - b[1];
    du.mul_add(du, dv * dv)
}

fn has_repeated_index(face: [u32; 3]) -> bool {
    face[0] == face[1] || face[0] == face[2] || face[1] == face[2]
}

fn position_for_face_vertex(
    mesh: &Mesh3D,
    face_index: usize,
    vertex_index: u32,
) -> Result<[f32; 3]> {
    let vertex_index = usize::try_from(vertex_index).context("mesh index does not fit in usize")?;
    let position = *mesh
        .positions
        .get(vertex_index)
        .with_context(|| format!("mesh face {face_index} index {vertex_index} is out of bounds"))?;
    ensure!(
        position.iter().all(|component| component.is_finite()),
        "mesh face {face_index} references a non-finite position"
    );
    Ok(position)
}

fn triangle_area_squared(a: [f32; 3], b: [f32; 3], c: [f32; 3]) -> f32 {
    let ab = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
    let ac = [c[0] - a[0], c[1] - a[1], c[2] - a[2]];
    let cross = [
        ab[1].mul_add(ac[2], -ab[2] * ac[1]),
        ab[2].mul_add(ac[0], -ab[0] * ac[2]),
        ab[0].mul_add(ac[1], -ab[1] * ac[0]),
    ];
    0.25 * position_distance_squared(cross, [0.0, 0.0, 0.0])
}

#[derive(Debug, Clone, Copy)]
struct SampleRange {
    min: f64,
    max: f64,
}

#[derive(Debug, Clone, Copy)]
struct SampleSpec {
    component: usize,
    range: SampleRange,
}

fn validate_sample_specs_for_len(
    variable_count: usize,
    samples: &[Sample],
) -> Result<Vec<SampleSpec>> {
    ensure!(
        samples.len() <= 2,
        "at most two sampled quantities are supported"
    );

    samples
        .iter()
        .map(|sample| {
            let component =
                usize::try_from(sample.id).context("sample component id is too large")?;
            ensure!(
                component < variable_count,
                "sample component index {component} is out of range"
            );

            let min = *sample.range.start();
            let max = *sample.range.end();
            ensure!(
                min.is_finite() && max.is_finite(),
                "sample range bounds must be finite"
            );
            ensure!(max > min, "sample range maximum must exceed its minimum");

            Ok(SampleSpec {
                component,
                range: SampleRange { min, max },
            })
        })
        .collect()
}

fn validate_isosurface_options(
    variable_count: usize,
    options: &IsosurfaceOptions,
) -> Result<(usize, f32, Vec<SampleSpec>)> {
    let component = usize::try_from(options.surface.id).context("surface id is too large")?;
    ensure!(
        component < variable_count,
        "surface component index {component} is out of range"
    );
    ensure!(options.surface.value.is_finite(), "isovalue must be finite");
    let isovalue = options.surface.value as f32;
    ensure!(isovalue.is_finite(), "isovalue does not fit in f32");
    if let Some(levels) = &options.levels {
        ensure!(
            levels.start() <= levels.end(),
            "level range start must not exceed its end"
        );
    }

    let sample_specs = validate_sample_specs_for_len(variable_count, &options.sampled_quantities)?;
    Ok((component, isovalue, sample_specs))
}

/// Load required data from a plotfile and extract an isosurface.
///
/// This is the convenience entry point for direct plotfile use. It loads the
/// requested surface component and sampled components into the compact sparse
/// representation before meshing.
pub fn isosurface(plot_file: &PlotFile, options: IsosurfaceOptions) -> Result<Mesh3D> {
    let (component, _, sample_specs) =
        validate_isosurface_options(plot_file.variables().len(), &options)?;
    let sampled_components = sample_specs
        .iter()
        .map(|sample| sample.component)
        .collect::<Vec<_>>();
    let compact_plot = load_compact_for_isosurface(plot_file, component, &sampled_components)?;
    isosurface_compact(&compact_plot, options)
}

/// Extract an isosurface from an already-loaded compact plot.
///
/// Use this when a compact archive has already been read, or when multiple
/// operations should share the same compact representation.
pub fn isosurface_compact(
    compact_plot: &CompactPlot,
    options: IsosurfaceOptions,
) -> Result<Mesh3D> {
    let (component, isovalue, sample_specs) =
        validate_isosurface_options(compact_plot.variable_count(), &options)?;
    let sampled_components = sample_specs
        .iter()
        .map(|sample| sample.component)
        .collect::<Vec<_>>();
    let levels = dual_grid_levels_from_compact(
        compact_plot,
        component,
        &sampled_components,
        options.levels.as_ref(),
    )?;
    let ranges = sample_specs
        .iter()
        .map(|sample| sample.range)
        .collect::<Vec<_>>();
    let mut mesh = Mesh3D::default();

    for level in &levels {
        let index_start = mesh.indices.len();
        let level_mesh = mesh_level_parallel(level, &ranges, isovalue)
            .with_context(|| format!("extracting level {}", level.level_index))?;
        merge_mesh(&mut mesh, level_mesh)?;

        if options.flip_winding {
            flip_face_winding(&mut mesh.indices[index_start..]);
        }
    }

    Ok(mesh)
}

fn mesh_level_parallel(
    level: &dual_grid::DualGridLevel<'_>,
    ranges: &[SampleRange],
    isovalue: f32,
) -> Result<Mesh3D> {
    let chunk_aabbs = level.active_cubes.chunk_aabbs();
    let mut chunks = chunk_aabbs
        .into_par_iter()
        .map(|active_aabb| {
            let mut mesh = Mesh3D::default();
            mc33::mesh_level_aabb(level, active_aabb, ranges, isovalue, &mut mesh)?;
            Ok((active_aabb, mesh))
        })
        .collect::<Result<Vec<_>>>()?;
    chunks.sort_by_key(|(aabb, _)| chunk_sort_key(*aabb));

    let mut mesh = Mesh3D::default();
    for (_, chunk_mesh) in chunks {
        merge_mesh(&mut mesh, chunk_mesh)?;
    }
    Ok(mesh)
}

fn merge_mesh(dest: &mut Mesh3D, src: Mesh3D) -> Result<()> {
    let base = u32::try_from(dest.positions.len()).context("mesh vertex count exceeds u32")?;
    ensure!(
        src.positions.len() == src.uv.len(),
        "source mesh position and uv buffers have different lengths"
    );
    ensure!(
        src.indices.iter().all(|face| {
            face[0] <= u32::MAX - base && face[1] <= u32::MAX - base && face[2] <= u32::MAX - base
        }),
        "mesh face index exceeds u32 after merge"
    );

    dest.positions.extend(src.positions);
    dest.uv.extend(src.uv);
    dest.indices.extend(
        src.indices
            .into_iter()
            .map(|face| [face[0] + base, face[1] + base, face[2] + base]),
    );
    Ok(())
}

fn chunk_sort_key(aabb: Aabb3u) -> (u32, u32, u32) {
    (aabb.min.z, aabb.min.y, aabb.min.x)
}

fn flip_face_winding(faces: &mut [[u32; 3]]) {
    for face in faces {
        face.swap(1, 2);
    }
}

#[cfg(test)]
mod tests;
