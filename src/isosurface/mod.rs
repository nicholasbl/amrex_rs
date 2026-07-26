//! MC33 isosurface extraction for AMReX plotfiles and compact archives.
//!
//! The extractor operates on cell-centered scalar data. Optional sampled
//! quantities are interpolated onto vertices and emitted as normalized texture
//! coordinate channels.

mod dual_grid;
mod mc33;
mod sampling;

use std::ops::RangeInclusive;

use anyhow::{Context, Result, ensure};
use glam::{UVec3, Vec3};
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
    pub positions: Vec<Vec3>,
    /// Per-vertex texture coordinates. Indices match `positions`.
    pub uv: Vec<Vec3>,
    /// Triangle vertex indices.
    pub indices: Vec<UVec3>,
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
            face.x <= u32::MAX - base && face.y <= u32::MAX - base && face.z <= u32::MAX - base
        }),
        "mesh face index exceeds u32 after merge"
    );

    dest.positions.extend(src.positions);
    dest.uv.extend(src.uv);
    dest.indices.extend(
        src.indices
            .into_iter()
            .map(|face| face + UVec3::splat(base)),
    );
    Ok(())
}

fn chunk_sort_key(aabb: Aabb3u) -> (u32, u32, u32) {
    (aabb.min.z, aabb.min.y, aabb.min.x)
}

fn flip_face_winding(faces: &mut [UVec3]) {
    for face in faces {
        std::mem::swap(&mut face.y, &mut face.z);
    }
}

#[cfg(test)]
mod tests;
