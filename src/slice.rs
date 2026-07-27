//! Axis-aligned volume slice mesh extraction.

use std::collections::HashMap;
use std::ops::RangeInclusive;

use anyhow::{Context, Result, ensure};
use glam::{DVec3, IVec3, UVec3};

use crate::PlotFile;
use crate::compact::CompactPlot;
use crate::isosurface::{Mesh3D, Sample};
use crate::sparse_amr::level_translation;
use crate::utility::{GridAccessor, SparseGrid3};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SliceAxis {
    X,
    Y,
    Z,
}

#[derive(Debug, Clone, Copy)]
pub struct SlicePlane {
    /// Cardinal axis normal to the slice plane.
    pub axis: SliceAxis,
    /// Physical coordinate where the slice plane is placed.
    pub value: f64,
}

#[derive(Debug)]
pub struct SliceOptions {
    /// Axis-aligned physical slice plane.
    pub plane: SlicePlane,
    /// One or two scalar components sampled into mesh UV channels.
    pub sampled_quantities: Vec<Sample>,
    /// Optional inclusive AMR level range to extract.
    pub levels: Option<RangeInclusive<usize>>,
    /// Flip emitted triangle winding if set.
    pub flip_winding: bool,
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

struct SliceLevel<'a> {
    level_index: usize,
    index_origin: IVec3,
    physical_origin: DVec3,
    cell_size: DVec3,
    sampled_quantities: Vec<&'a SparseGrid3<f32>>,
    active_cubes: SparseGrid3<()>,
}

/// Load required data from a plotfile and extract an axis-aligned slice mesh.
pub fn slice(plot_file: &PlotFile, options: SliceOptions) -> Result<Mesh3D> {
    let sample_specs =
        validate_slice_options(plot_file.variables().len(), &options.sampled_quantities)?;
    let sampled_components = sample_specs
        .iter()
        .map(|sample| sample.component)
        .collect::<Vec<_>>();
    let compact_plot = CompactPlot::load(plot_file, &sampled_components)?;
    slice_compact(&compact_plot, options)
}

/// Extract an axis-aligned slice mesh from an already-loaded compact plot.
pub fn slice_compact(compact_plot: &CompactPlot, options: SliceOptions) -> Result<Mesh3D> {
    ensure!(
        options.plane.value.is_finite(),
        "slice plane value must be finite"
    );
    let sample_specs =
        validate_slice_options(compact_plot.variable_count(), &options.sampled_quantities)?;
    let sampled_components = sample_specs
        .iter()
        .map(|sample| sample.component)
        .collect::<Vec<_>>();
    let ranges = sample_specs
        .iter()
        .map(|sample| sample.range)
        .collect::<Vec<_>>();
    let levels =
        slice_levels_from_compact(compact_plot, &sampled_components, options.levels.as_ref())?;

    let mut mesh = Mesh3D::default();
    for level in &levels {
        let index_start = mesh.indices.len();
        let level_mesh = mesh_level(level, options.plane, &ranges)
            .with_context(|| format!("extracting slice for level {}", level.level_index))?;
        merge_mesh(&mut mesh, level_mesh)?;
        if options.flip_winding {
            flip_winding(&mut mesh.indices[index_start..]);
        }
    }
    Ok(mesh)
}

fn validate_slice_options(variable_count: usize, samples: &[Sample]) -> Result<Vec<SampleSpec>> {
    ensure!(
        !samples.is_empty(),
        "at least one sampled quantity is required for slice extraction"
    );
    ensure!(samples.len() <= 2, "at most two UV axes are supported");

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

fn slice_levels_from_compact<'a>(
    compact_plot: &'a CompactPlot,
    sampled_components: &[usize],
    level_range: Option<&RangeInclusive<usize>>,
) -> Result<Vec<SliceLevel<'a>>> {
    let sampled_slots = sampled_components
        .iter()
        .map(|&component| {
            compact_plot
                .component_slot(component)
                .with_context(|| format!("compact plot does not contain component {component}"))
        })
        .collect::<Result<Vec<_>>>()?;

    let mut levels: Vec<SliceLevel<'a>> = Vec::with_capacity(compact_plot.levels.len());
    for compact_level in &compact_plot.levels {
        if !level_in_range(compact_level.level_index, level_range) {
            continue;
        }

        if let Some(coarse) = levels.last_mut() {
            let ratio = compact_plot
                .refinement_ratios
                .get(compact_level.level_index - 1)
                .context("refinement ratio is absent for compact level")?;
            let ratio = u32::try_from(*ratio).context("refinement ratio does not fit in u32")?;
            ensure!(ratio > 0, "refinement ratio must be nonzero");

            let translation =
                level_translation(coarse.index_origin, compact_level.index_origin, ratio)?;
            let fine_coverage = compact_level
                .components
                .get(sampled_slots[0])
                .context("sample component slot is absent from compact level")?;
            coarse
                .active_cubes
                .mask_out_by_presence_scaled(fine_coverage, ratio, translation);
        }

        levels.push(SliceLevel {
            level_index: compact_level.level_index,
            index_origin: compact_level.index_origin,
            physical_origin: compact_level.physical_origin,
            cell_size: compact_level.cell_size,
            sampled_quantities: sampled_slots
                .iter()
                .map(|&slot| {
                    compact_level
                        .components
                        .get(slot)
                        .context("sample component slot is absent from compact level")
                })
                .collect::<Result<Vec<_>>>()?,
            active_cubes: compact_level.eligible_cubes.clone(),
        });
    }

    Ok(levels)
}

fn mesh_level(level: &SliceLevel<'_>, plane: SlicePlane, ranges: &[SampleRange]) -> Result<Mesh3D> {
    let axis = axis_index(plane.axis);
    let Some((anchor_axis, t)) = plane_anchor_and_t(level, plane)? else {
        return Ok(Mesh3D::default());
    };

    let mut mesh = Mesh3D::default();
    let mut vertex_ids = HashMap::new();
    let mut accessors = level
        .sampled_quantities
        .iter()
        .map(|grid| grid.accessor())
        .collect::<Vec<_>>();

    let mut error = None;
    level
        .active_cubes
        .for_each_present_in_aabb(level.active_cubes.bounds_aabb(), |anchor, ()| {
            if error.is_none() && component(anchor, axis) == anchor_axis {
                error = emit_quad(
                    &mut mesh,
                    &mut vertex_ids,
                    &mut accessors,
                    ranges,
                    level,
                    plane,
                    anchor,
                    t,
                )
                .err();
            }
        });

    match error {
        Some(error) => Err(error),
        None => Ok(mesh),
    }
}

fn plane_anchor_and_t(level: &SliceLevel<'_>, plane: SlicePlane) -> Result<Option<(u32, f32)>> {
    let axis = axis_index(plane.axis);
    let bounds = level.active_cubes.bounds();
    if component(bounds, axis) == 0 {
        return Ok(None);
    }

    let grid_position = (plane.value - component_d(level.physical_origin, axis))
        / component_d(level.cell_size, axis)
        - 0.5;
    if !grid_position.is_finite() {
        return Ok(None);
    }

    let lower = grid_position.floor();
    let mut anchor = lower as i64;
    let mut t = grid_position - lower;
    if t == 0.0 && anchor > 0 {
        anchor -= 1;
        t = 1.0;
    }
    if anchor < 0 || anchor >= i64::from(component(bounds, axis)) {
        return Ok(None);
    }
    Ok(Some((
        u32::try_from(anchor).context("slice anchor does not fit in u32")?,
        t as f32,
    )))
}

fn emit_quad(
    mesh: &mut Mesh3D,
    vertex_ids: &mut HashMap<UVec3, u32>,
    accessors: &mut [GridAccessor<'_, f32>],
    ranges: &[SampleRange],
    level: &SliceLevel<'_>,
    plane: SlicePlane,
    anchor: UVec3,
    t: f32,
) -> Result<()> {
    let corners = quad_corners(plane.axis, anchor);
    let a = vertex(
        mesh, vertex_ids, accessors, ranges, level, plane, corners[0], t,
    )?;
    let b = vertex(
        mesh, vertex_ids, accessors, ranges, level, plane, corners[1], t,
    )?;
    let c = vertex(
        mesh, vertex_ids, accessors, ranges, level, plane, corners[2], t,
    )?;
    let d = vertex(
        mesh, vertex_ids, accessors, ranges, level, plane, corners[3], t,
    )?;
    mesh.indices.push([a, b, c]);
    mesh.indices.push([a, c, d]);
    Ok(())
}

fn vertex(
    mesh: &mut Mesh3D,
    vertex_ids: &mut HashMap<UVec3, u32>,
    accessors: &mut [GridAccessor<'_, f32>],
    ranges: &[SampleRange],
    level: &SliceLevel<'_>,
    plane: SlicePlane,
    key: UVec3,
    t: f32,
) -> Result<u32> {
    if let Some(&id) = vertex_ids.get(&key) {
        return Ok(id);
    }

    let position = physical_position(level, plane, key);
    let uv = sample_uv(accessors, ranges, plane.axis, key, t)?;
    let id = u32::try_from(mesh.positions.len()).context("mesh vertex count exceeds u32")?;
    mesh.positions.push(position);
    mesh.uv.push(uv);
    vertex_ids.insert(key, id);
    Ok(id)
}

fn sample_uv(
    accessors: &mut [GridAccessor<'_, f32>],
    ranges: &[SampleRange],
    axis: SliceAxis,
    key: UVec3,
    t: f32,
) -> Result<[f32; 2]> {
    ensure!(
        accessors.len() == ranges.len(),
        "sample grid and range counts differ"
    );
    let mut uv = [0.0; 2];
    let lower = key;
    let upper = add_axis(key, axis, 1)?;
    for (channel, (accessor, range)) in accessors.iter_mut().zip(ranges).enumerate() {
        let a = accessor
            .get(lower)
            .with_context(|| format!("slice sample is absent at {lower:?}"))?;
        let b = accessor
            .get(upper)
            .with_context(|| format!("slice sample is absent at {upper:?}"))?;
        uv[channel] = normalize_sample((b - a).mul_add(t, a), *range);
    }
    Ok(uv)
}

fn normalize_sample(value: f32, range: SampleRange) -> f32 {
    ((f64::from(value) - range.min) / (range.max - range.min)).clamp(0.0, 1.0) as f32
}

fn physical_position(level: &SliceLevel<'_>, plane: SlicePlane, key: UVec3) -> [f32; 3] {
    let mut position = (level.physical_origin
        + (key.as_dvec3() + DVec3::splat(0.5)) * level.cell_size)
        .as_vec3()
        .to_array();
    position[axis_index(plane.axis)] = plane.value as f32;
    position
}

fn quad_corners(axis: SliceAxis, anchor: UVec3) -> [UVec3; 4] {
    match axis {
        SliceAxis::X => [
            UVec3::new(anchor.x, anchor.y, anchor.z),
            UVec3::new(anchor.x, anchor.y, anchor.z + 1),
            UVec3::new(anchor.x, anchor.y + 1, anchor.z + 1),
            UVec3::new(anchor.x, anchor.y + 1, anchor.z),
        ],
        SliceAxis::Y => [
            UVec3::new(anchor.x, anchor.y, anchor.z),
            UVec3::new(anchor.x + 1, anchor.y, anchor.z),
            UVec3::new(anchor.x + 1, anchor.y, anchor.z + 1),
            UVec3::new(anchor.x, anchor.y, anchor.z + 1),
        ],
        SliceAxis::Z => [
            UVec3::new(anchor.x, anchor.y, anchor.z),
            UVec3::new(anchor.x, anchor.y + 1, anchor.z),
            UVec3::new(anchor.x + 1, anchor.y + 1, anchor.z),
            UVec3::new(anchor.x + 1, anchor.y, anchor.z),
        ],
    }
}

fn add_axis(mut p: UVec3, axis: SliceAxis, amount: u32) -> Result<UVec3> {
    match axis {
        SliceAxis::X => p.x = p.x.checked_add(amount).context("x coordinate overflow")?,
        SliceAxis::Y => p.y = p.y.checked_add(amount).context("y coordinate overflow")?,
        SliceAxis::Z => p.z = p.z.checked_add(amount).context("z coordinate overflow")?,
    }
    Ok(p)
}

fn component(v: UVec3, axis: usize) -> u32 {
    [v.x, v.y, v.z][axis]
}

fn component_d(v: DVec3, axis: usize) -> f64 {
    [v.x, v.y, v.z][axis]
}

fn axis_index(axis: SliceAxis) -> usize {
    match axis {
        SliceAxis::X => 0,
        SliceAxis::Y => 1,
        SliceAxis::Z => 2,
    }
}

fn level_in_range(level_index: usize, level_range: Option<&RangeInclusive<usize>>) -> bool {
    level_range.is_none_or(|range| range.contains(&level_index))
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

fn flip_winding(faces: &mut [[u32; 3]]) {
    for face in faces {
        face.swap(1, 2);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compact::{CompactLevel, CompactPlot};
    use crate::utility::SparseGrid3;
    use glam::{IVec3, Vec3};

    fn compact_for_linear_field() -> CompactPlot {
        let mut u = SparseGrid3::new(UVec3::splat(3));
        let mut v = SparseGrid3::new(UVec3::splat(3));
        for z in 0..3 {
            for y in 0..3 {
                for x in 0..3 {
                    let p = UVec3::new(x, y, z);
                    u.set(p, x as f32);
                    v.set(p, (y + z) as f32);
                }
            }
        }
        let eligible_cubes = crate::compact::build_active_dual_cubes(&u, &[u.bounds_aabb()]);
        CompactPlot {
            variable_count: 2,
            refinement_ratios: Vec::new(),
            component_ids: vec![0, 1],
            levels: vec![CompactLevel {
                level_index: 0,
                index_origin: IVec3::ZERO,
                physical_origin: DVec3::ZERO,
                cell_size: DVec3::ONE,
                components: vec![u, v],
                eligible_cubes,
            }],
        }
    }

    #[test]
    fn x_slice_places_geometry_at_exact_plane_and_interpolates_uvs() -> Result<()> {
        let compact = compact_for_linear_field();
        let mesh = slice_compact(
            &compact,
            SliceOptions {
                plane: SlicePlane {
                    axis: SliceAxis::X,
                    value: 1.25,
                },
                sampled_quantities: vec![
                    Sample {
                        id: 0,
                        range: 0.0..=2.0,
                    },
                    Sample {
                        id: 1,
                        range: 0.0..=4.0,
                    },
                ],
                levels: None,
                flip_winding: false,
            },
        )?;

        assert_eq!(mesh.indices.len(), 8);
        assert_eq!(mesh.positions.len(), 9);
        assert!(mesh.positions.iter().all(|p| (p[0] - 1.25).abs() < 1.0e-6));
        let min_u = mesh.uv.iter().map(|uv| uv[0]).fold(f32::INFINITY, f32::min);
        assert!((min_u - 0.375).abs() < 1.0e-6);
        assert!(mesh.uv.iter().all(|uv| (0.0..=1.0).contains(&uv[1])));
        Ok(())
    }

    #[test]
    fn flip_winding_inverts_slice_orientation() -> Result<()> {
        let compact = compact_for_linear_field();
        let options = |flip_winding| SliceOptions {
            plane: SlicePlane {
                axis: SliceAxis::Z,
                value: 1.25,
            },
            sampled_quantities: vec![Sample {
                id: 0,
                range: 0.0..=2.0,
            }],
            levels: None,
            flip_winding,
        };
        let mesh = slice_compact(&compact, options(false))?;
        let flipped = slice_compact(&compact, options(true))?;

        assert!(normal(&mesh, mesh.indices[0]).dot(Vec3::NEG_Z) > 0.0);
        assert!(normal(&flipped, flipped.indices[0]).dot(Vec3::Z) > 0.0);
        Ok(())
    }

    fn normal(mesh: &Mesh3D, face: [u32; 3]) -> Vec3 {
        let a = Vec3::from_array(mesh.positions[face[0] as usize]);
        let b = Vec3::from_array(mesh.positions[face[1] as usize]);
        let c = Vec3::from_array(mesh.positions[face[2] as usize]);
        (b - a).cross(c - a)
    }
}
