use std::{
    collections::{HashMap, HashSet},
    ops::RangeInclusive,
};

use anyhow::{Context, Result, ensure};
use glam::{DVec3, I64Vec3, IVec3, U16Vec2, UVec3, Vec3};

use crate::{DataReader, IndexDomain, Level, Patch, PlotFile};

use super::sparse_grid3::{Aabb3u, SparseGrid3};

#[derive(Debug, Default)]
pub struct Surface {
    pub id: u32,
    pub value: f64,
}

#[derive(Debug)]
pub struct Sample {
    pub id: u32,
    pub range: RangeInclusive<f64>,
}

#[derive(Debug)]
pub struct IsosurfaceOptions {
    pub surface: Surface,
    pub sampled_quantities: Vec<Sample>,
    /// Fraction of an edge near either endpoint that is snapped to that endpoint.
    pub regularization: f32,
}

impl Default for IsosurfaceOptions {
    fn default() -> Self {
        Self {
            surface: Surface::default(),
            sampled_quantities: Vec::new(),
            regularization: 0.25,
        }
    }
}

#[derive(Debug)]
pub struct Mesh3D {
    pub positions: Vec<Vertex3D>,
    pub faces: Vec<UVec3>,
}

#[derive(Debug)]
pub struct Vertex3D {
    pub position: Vec3,
    pub sampled_values: U16Vec2,
}

/// Cell-centered samples and eligible dual-lattice cube anchors for one AMR level.
///
/// All sparse grids use coordinates local to `index_origin`. Samples covered by
/// a finer level remain available; only `active_cubes` is masked.
pub(crate) struct DualGridLevel {
    pub(crate) level_index: usize,
    pub(crate) index_origin: IVec3,
    pub(crate) physical_origin: DVec3,
    pub(crate) cell_size: DVec3,
    pub(crate) samples: SparseGrid3<f32>,
    pub(crate) sampled_quantities: Vec<SparseGrid3<f32>>,
    pub(crate) active_cubes: SparseGrid3<()>,
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

/// Load one scalar component into a sparse dual lattice for every AMR level.
///
/// Levels are loaded from coarse to fine. Valid fine-level sample coverage masks
/// eligible cube anchors on the immediately coarser level. Fine and coarse meshes
/// are intentionally not stitched at this stage.
pub(crate) fn load_dual_grid_levels(
    plot_file: &PlotFile,
    component: usize,
    sampled_components: &[usize],
) -> Result<Vec<DualGridLevel>> {
    ensure!(
        component < plot_file.variables().len(),
        "component index {component} is out of range"
    );
    ensure!(
        sampled_components.len() <= 2,
        "at most two sampled quantities are supported"
    );
    for &sampled_component in sampled_components {
        ensure!(
            sampled_component < plot_file.variables().len(),
            "sample component index {sampled_component} is out of range"
        );
    }

    let reader = DataReader::new(plot_file);
    let mut levels: Vec<DualGridLevel> = Vec::with_capacity(plot_file.header().finest_level + 1);

    for level in plot_file.levels() {
        let loaded = load_dual_grid_level(
            &reader,
            level,
            plot_file.header().domain.min,
            component,
            sampled_components,
        )
        .with_context(|| format!("loading dual grid for level {}", level.index()))?;

        if let Some(coarse) = levels.last_mut() {
            let ratio = u32::try_from(plot_file.header().refinement_ratios[level.index() - 1])
                .context("refinement ratio does not fit in u32")?;
            ensure!(ratio > 0, "refinement ratio must be nonzero");

            let translation = level_translation(coarse.index_origin, loaded.index_origin, ratio)?;
            coarse
                .active_cubes
                .mask_out_by_presence_scaled(&loaded.samples, ratio, translation);
        }

        levels.push(loaded);
    }

    Ok(levels)
}

fn load_dual_grid_level(
    reader: &DataReader<'_>,
    level: Level<'_>,
    physical_origin: DVec3,
    component: usize,
    sampled_components: &[usize],
) -> Result<DualGridLevel> {
    let domain = level.index_domain();
    ensure!(
        domain.index_type == IVec3::ZERO,
        "isosurface extraction requires cell-centered data"
    );

    let bounds = index_extent(domain)?;
    let mut samples = SparseGrid3::with_chunk_capacity(bounds, level.patch_count());
    let mut sampled_quantities = sampled_components
        .iter()
        .map(|_| SparseGrid3::with_chunk_capacity(bounds, level.patch_count()))
        .collect::<Vec<_>>();
    let mut patch_aabbs = Vec::with_capacity(level.patch_count());

    for patch in level.patches()? {
        let patch_box = patch.index_box();
        ensure!(
            patch_box.index_type == IVec3::ZERO,
            "patch {} is not cell-centered",
            patch.index()
        );

        let local_aabb = local_aabb(patch_box, domain)?;
        load_patch_component(reader, patch, component, local_aabb, &mut samples)?;
        for (&sampled_component, grid) in sampled_components.iter().zip(&mut sampled_quantities) {
            load_patch_component(reader, patch, sampled_component, local_aabb, grid)?;
        }
        patch_aabbs.push(local_aabb);
    }

    let active_cubes = build_active_dual_cubes(&samples, &patch_aabbs);

    Ok(DualGridLevel {
        level_index: level.index(),
        index_origin: domain.min,
        physical_origin,
        cell_size: level.cell_size(),
        samples,
        sampled_quantities,
        active_cubes,
    })
}

fn load_patch_component(
    reader: &DataReader<'_>,
    patch: Patch<'_>,
    component: usize,
    local_aabb: Aabb3u,
    destination: &mut SparseGrid3<f32>,
) -> Result<()> {
    let patch_box = patch.index_box();
    let view = reader.component(patch, component)?;
    let value_count = local_aabb
        .volume_usize()
        .context("patch sample count does not fit in usize")?;
    let mut values = Vec::with_capacity(value_count);

    // Read only the valid index box. ComponentView may also contain ghost cells.
    for z in patch_box.min.z..=patch_box.max.z {
        for y in patch_box.min.y..=patch_box.max.y {
            for x in patch_box.min.x..=patch_box.max.x {
                let index = IVec3::new(x, y, z);
                let value = view.get(index).with_context(|| {
                    format!(
                        "valid index {index:?} is absent from patch {} component {component}",
                        patch.index()
                    )
                })?;
                values.push(value as f32);
            }
        }
    }

    destination
        .write_aabb_values(local_aabb, &values)
        .map_err(|error| anyhow::anyhow!("writing patch component {component}: {error:?}"))
}

fn build_active_dual_cubes(
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

fn index_extent(domain: &IndexDomain) -> Result<UVec3> {
    let extent = domain.max.as_i64vec3() - domain.min.as_i64vec3() + I64Vec3::ONE;
    ensure!(extent.cmpgt(I64Vec3::ZERO).all(), "index domain is empty");
    Ok(UVec3::new(
        u32::try_from(extent.x).context("x index extent does not fit in u32")?,
        u32::try_from(extent.y).context("y index extent does not fit in u32")?,
        u32::try_from(extent.z).context("z index extent does not fit in u32")?,
    ))
}

fn local_aabb(index_box: &IndexDomain, domain: &IndexDomain) -> Result<Aabb3u> {
    ensure!(
        index_box.min.cmpge(domain.min).all() && index_box.max.cmple(domain.max).all(),
        "patch index box lies outside its level domain"
    );
    let min = index_box.min.as_i64vec3() - domain.min.as_i64vec3();
    let max = index_box.max.as_i64vec3() - domain.min.as_i64vec3() + I64Vec3::ONE;
    Ok(Aabb3u::new(
        UVec3::new(
            u32::try_from(min.x)?,
            u32::try_from(min.y)?,
            u32::try_from(min.z)?,
        ),
        UVec3::new(
            u32::try_from(max.x)?,
            u32::try_from(max.y)?,
            u32::try_from(max.z)?,
        ),
    ))
}

fn level_translation(coarse_origin: IVec3, fine_origin: IVec3, scale: u32) -> Result<I64Vec3> {
    fn axis(coarse: i32, fine: i32, scale: u32) -> Result<i64> {
        let value = i128::from(coarse) * i128::from(scale) - i128::from(fine);
        i64::try_from(value).context("level-index translation does not fit in i64")
    }

    Ok(I64Vec3::new(
        axis(coarse_origin.x, fine_origin.x, scale)?,
        axis(coarse_origin.y, fine_origin.y, scale)?,
        axis(coarse_origin.z, fine_origin.z, scale)?,
    ))
}

fn validate_sample_specs(plot_file: &PlotFile, samples: &[Sample]) -> Result<Vec<SampleSpec>> {
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
                component < plot_file.variables().len(),
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

/// Interpolate auxiliary quantities along the same edge and with the same
/// parameter used to place an isosurface vertex.
fn sampled_values_on_edge(
    level: &DualGridLevel,
    start: UVec3,
    end: UVec3,
    t: f32,
    ranges: &[SampleRange],
) -> Result<U16Vec2> {
    encode_sampled_values(level, ranges, |grid| {
        let start_value = grid
            .get(start)
            .with_context(|| format!("sample value is absent at edge start {start:?}"))?;
        let end_value = grid
            .get(end)
            .with_context(|| format!("sample value is absent at edge end {end:?}"))?;
        Ok((end_value - start_value).mul_add(t, start_value))
    })
}

/// Read auxiliary quantities at a lattice point used by an RMT-snapped vertex.
fn sampled_values_at(
    level: &DualGridLevel,
    position: UVec3,
    ranges: &[SampleRange],
) -> Result<U16Vec2> {
    encode_sampled_values(level, ranges, |grid| {
        grid.get(position)
            .with_context(|| format!("sample value is absent at snapped vertex {position:?}"))
    })
}

fn encode_sampled_values<F>(
    level: &DualGridLevel,
    ranges: &[SampleRange],
    mut value: F,
) -> Result<U16Vec2>
where
    F: FnMut(&SparseGrid3<f32>) -> Result<f32>,
{
    ensure!(
        level.sampled_quantities.len() == ranges.len(),
        "sample grid and range counts differ"
    );
    ensure!(ranges.len() <= 2, "at most two UV axes are supported");

    let mut encoded = [0_u16; 2];
    for (axis, (grid, range)) in level.sampled_quantities.iter().zip(ranges).enumerate() {
        encoded[axis] = encode_unorm16(value(grid)?, *range);
    }
    Ok(U16Vec2::new(encoded[0], encoded[1]))
}

fn encode_unorm16(value: f32, range: SampleRange) -> u16 {
    let normalized = ((f64::from(value) - range.min) / (range.max - range.min)).clamp(0.0, 1.0);
    (normalized * f64::from(u16::MAX)).round() as u16
}

const CUBE_CORNERS: [UVec3; 8] = [
    UVec3::new(0, 0, 0),
    UVec3::new(1, 0, 0),
    UVec3::new(0, 1, 0),
    UVec3::new(1, 1, 0),
    UVec3::new(0, 0, 1),
    UVec3::new(1, 0, 1),
    UVec3::new(0, 1, 1),
    UVec3::new(1, 1, 1),
];

// Six tetrahedra around the 0-7 body diagonal. This produces matching face
// diagonals in adjacent dual cubes.
const CUBE_TETRAHEDRA: [[usize; 4]; 6] = [
    [0, 1, 3, 7],
    [0, 3, 2, 7],
    [0, 2, 6, 7],
    [0, 6, 4, 7],
    [0, 4, 5, 7],
    [0, 5, 1, 7],
];

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum VertexKey {
    Grid(UVec3),
    Edge(UVec3, UVec3),
}

struct LevelMesher<'a> {
    level: &'a DualGridLevel,
    ranges: &'a [SampleRange],
    isovalue: f32,
    regularization: f32,
    vertices: &'a mut Vec<Vertex3D>,
    faces: &'a mut Vec<UVec3>,
    vertex_ids: HashMap<VertexKey, u32>,
    face_ids: HashSet<[u32; 3]>,
}

impl LevelMesher<'_> {
    fn march_cube(&mut self, anchor: UVec3) -> Result<()> {
        let corners = CUBE_CORNERS.map(|offset| anchor + offset);
        let mut values = [0.0_f32; 8];
        for (slot, corner) in values.iter_mut().zip(corners) {
            *slot = self
                .level
                .samples
                .get(corner)
                .with_context(|| format!("surface sample is absent at {corner:?}"))?;
        }

        for tetrahedron in CUBE_TETRAHEDRA {
            self.march_tetrahedron(
                tetrahedron.map(|index| corners[index]),
                tetrahedron.map(|index| values[index]),
            )?;
        }
        Ok(())
    }

    fn march_tetrahedron(&mut self, corners: [UVec3; 4], values: [f32; 4]) -> Result<()> {
        let mut inside = Vec::with_capacity(3);
        let mut outside = Vec::with_capacity(3);
        for (index, value) in values.into_iter().enumerate() {
            if value >= self.isovalue {
                inside.push(index);
            } else {
                outside.push(index);
            }
        }

        match inside.len() {
            0 | 4 => {}
            1 => {
                let i = inside[0];
                let a = self.edge_vertex(corners[i], corners[outside[0]])?;
                let b = self.edge_vertex(corners[i], corners[outside[1]])?;
                let c = self.edge_vertex(corners[i], corners[outside[2]])?;
                self.emit_face(a, b, c, corners[i]);
            }
            3 => {
                let o = outside[0];
                let a = self.edge_vertex(corners[o], corners[inside[0]])?;
                let b = self.edge_vertex(corners[o], corners[inside[1]])?;
                let c = self.edge_vertex(corners[o], corners[inside[2]])?;
                self.emit_face(a, b, c, corners[inside[0]]);
            }
            2 => {
                let i0 = inside[0];
                let i1 = inside[1];
                let o0 = outside[0];
                let o1 = outside[1];
                let p00 = self.edge_vertex(corners[i0], corners[o0])?;
                let p01 = self.edge_vertex(corners[i0], corners[o1])?;
                let p10 = self.edge_vertex(corners[i1], corners[o0])?;
                let p11 = self.edge_vertex(corners[i1], corners[o1])?;
                self.emit_face(p00, p01, p11, corners[i0]);
                self.emit_face(p00, p11, p10, corners[i0]);
            }
            _ => unreachable!(),
        }
        Ok(())
    }

    fn edge_vertex(&mut self, first: UVec3, second: UVec3) -> Result<u32> {
        let (start, end) = if grid_point_le(first, second) {
            (first, second)
        } else {
            (second, first)
        };
        let start_value = self
            .level
            .samples
            .get(start)
            .context("missing edge start")?;
        let end_value = self.level.samples.get(end).context("missing edge end")?;
        let denominator = end_value - start_value;
        let t = if denominator == 0.0 {
            0.5
        } else {
            ((self.isovalue - start_value) / denominator).clamp(0.0, 1.0)
        };

        let (key, position, sampled_values) = if t <= self.regularization {
            (
                VertexKey::Grid(start),
                self.grid_position(start),
                sampled_values_at(self.level, start, self.ranges)?,
            )
        } else if t >= 1.0 - self.regularization {
            (
                VertexKey::Grid(end),
                self.grid_position(end),
                sampled_values_at(self.level, end, self.ranges)?,
            )
        } else {
            let start_position = self.grid_position(start);
            let end_position = self.grid_position(end);
            (
                VertexKey::Edge(start, end),
                start_position.lerp(end_position, t),
                sampled_values_on_edge(self.level, start, end, t, self.ranges)?,
            )
        };

        if let Some(&id) = self.vertex_ids.get(&key) {
            return Ok(id);
        }
        let id = u32::try_from(self.vertices.len()).context("OBJ vertex count exceeds u32")?;
        self.vertices.push(Vertex3D {
            position,
            sampled_values,
        });
        self.vertex_ids.insert(key, id);
        Ok(id)
    }

    fn grid_position(&self, position: UVec3) -> Vec3 {
        let local = DVec3::new(
            f64::from(position.x) + 0.5,
            f64::from(position.y) + 0.5,
            f64::from(position.z) + 0.5,
        );
        (self.level.physical_origin + local * self.level.cell_size).as_vec3()
    }

    fn emit_face(&mut self, mut a: u32, mut b: u32, c: u32, inside: UVec3) {
        if a == b || b == c || c == a {
            return;
        }

        let pa = self.vertices[a as usize].position;
        let pb = self.vertices[b as usize].position;
        let pc = self.vertices[c as usize].position;
        let normal = (pb - pa).cross(pc - pa);
        if normal.length_squared() == 0.0 {
            return;
        }
        let inside_direction = self.grid_position(inside) - (pa + pb + pc) / 3.0;
        if normal.dot(inside_direction) > 0.0 {
            std::mem::swap(&mut a, &mut b);
        }

        let mut key = [a, b, c];
        key.sort_unstable();
        if self.face_ids.insert(key) {
            self.faces.push(UVec3::new(a, b, c));
        }
    }
}

fn grid_point_le(a: UVec3, b: UVec3) -> bool {
    (a.x, a.y, a.z) <= (b.x, b.y, b.z)
}

pub fn isosurface(plot_file: &PlotFile, options: IsosurfaceOptions) -> Result<Mesh3D> {
    let component = usize::try_from(options.surface.id).context("surface id is too large")?;
    ensure!(options.surface.value.is_finite(), "isovalue must be finite");
    ensure!(
        options.regularization.is_finite()
            && options.regularization >= 0.0
            && options.regularization < 0.5,
        "regularization must be in [0, 0.5)"
    );
    let isovalue = options.surface.value as f32;
    ensure!(isovalue.is_finite(), "isovalue does not fit in f32");

    let sample_specs = validate_sample_specs(plot_file, &options.sampled_quantities)?;
    let sampled_components = sample_specs
        .iter()
        .map(|sample| sample.component)
        .collect::<Vec<_>>();
    let levels = load_dual_grid_levels(plot_file, component, &sampled_components)?;
    let ranges = sample_specs
        .iter()
        .map(|sample| sample.range)
        .collect::<Vec<_>>();
    let mut mesh = Mesh3D {
        positions: Vec::new(),
        faces: Vec::new(),
    };

    for level in &levels {
        let mut mesher = LevelMesher {
            level,
            ranges: &ranges,
            isovalue,
            regularization: options.regularization,
            vertices: &mut mesh.positions,
            faces: &mut mesh.faces,
            vertex_ids: HashMap::new(),
            face_ids: HashSet::new(),
        };
        let mut error = None;
        level.active_cubes.for_each_present_in_aabb(
            level.active_cubes.bounds_aabb(),
            |anchor, ()| {
                if error.is_none() {
                    error = mesher.march_cube(anchor).err();
                }
            },
        );
        if let Some(error) = error {
            return Err(error).with_context(|| format!("extracting level {}", level.level_index));
        }
    }

    Ok(mesh)
}

#[cfg(test)]
mod tests {
    use std::{fs, fs::File, io::Write, path::Path};

    use super::*;

    fn level_with_sampled_quantities(sampled_quantities: Vec<SparseGrid3<f32>>) -> DualGridLevel {
        DualGridLevel {
            level_index: 0,
            index_origin: IVec3::ZERO,
            physical_origin: DVec3::ZERO,
            cell_size: DVec3::ONE,
            samples: SparseGrid3::new(UVec3::new(2, 1, 1)),
            sampled_quantities,
            active_cubes: SparseGrid3::new(UVec3::ZERO),
        }
    }

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
        shard.write_all(b"FAB ((8, (64 11 52 0 1 12 0 1023)),(8, (8 7 6 5 4 3 2 1)))((0,0,0) (1,1,1) (0,0,0)) 1\n")?;
        for _z in 0..2 {
            for _y in 0..2 {
                for x in 0..2 {
                    shard.write_all(&(x as f64).to_le_bytes())?;
                }
            }
        }
        Ok(())
    }

    #[test]
    fn active_cubes_span_adjacent_sample_regions() {
        let mut samples = SparseGrid3::new(UVec3::new(3, 2, 2));
        let left = Aabb3u::new(UVec3::ZERO, UVec3::new(1, 2, 2));
        let right = Aabb3u::new(UVec3::new(1, 0, 0), UVec3::new(3, 2, 2));
        samples.fill_aabb(left, 1.0);
        samples.fill_aabb(right, 2.0);

        let active = build_active_dual_cubes(&samples, &[left, right]);
        assert_eq!(active.bounds(), UVec3::new(2, 1, 1));
        assert_eq!(active.present_voxel_count(), 2);
        assert!(active.contains(UVec3::new(0, 0, 0)));
        assert!(active.contains(UVec3::new(1, 0, 0)));
    }

    #[test]
    fn incomplete_dual_cube_is_not_active() {
        let mut samples = SparseGrid3::new(UVec3::new(2, 2, 2));
        let region = samples.bounds_aabb();
        samples.fill_aabb(region, 1.0);
        samples.clear(UVec3::new(1, 1, 1));

        let active = build_active_dual_cubes(&samples, &[region]);
        assert!(active.is_empty());
    }

    #[test]
    fn translation_relates_level_local_coordinates() {
        let translation =
            level_translation(IVec3::new(-4, 3, 10), IVec3::new(-7, 8, 20), 2).unwrap();
        assert_eq!(translation, I64Vec3::new(-1, -2, 0));
    }

    #[test]
    fn edge_samples_are_interpolated_normalized_and_encoded_as_uv() {
        let mut u = SparseGrid3::new(UVec3::new(2, 1, 1));
        u.set(UVec3::ZERO, 0.0);
        u.set(UVec3::X, 10.0);
        let mut v = SparseGrid3::new(UVec3::new(2, 1, 1));
        v.set(UVec3::ZERO, -5.0);
        v.set(UVec3::X, 15.0);
        let level = level_with_sampled_quantities(vec![u, v]);
        let ranges = [
            SampleRange {
                min: 0.0,
                max: 10.0,
            },
            SampleRange {
                min: 0.0,
                max: 10.0,
            },
        ];

        let uv = sampled_values_on_edge(&level, UVec3::ZERO, UVec3::X, 0.25, &ranges).unwrap();
        assert_eq!(uv, U16Vec2::new(16_384, 0));
    }

    #[test]
    fn snapped_samples_are_clamped_and_unused_axis_is_zero() {
        let mut u = SparseGrid3::new(UVec3::new(1, 1, 1));
        u.set(UVec3::ZERO, 12.0);
        let level = level_with_sampled_quantities(vec![u]);
        let ranges = [SampleRange {
            min: 0.0,
            max: 10.0,
        }];

        let uv = sampled_values_at(&level, UVec3::ZERO, &ranges).unwrap();
        assert_eq!(uv, U16Vec2::new(u16::MAX, 0));
    }

    #[test]
    fn marching_tetrahedra_extracts_a_plane() {
        let mut level = level_with_sampled_quantities(Vec::new());
        level.samples = SparseGrid3::new(UVec3::splat(2));
        for z in 0..2 {
            for y in 0..2 {
                for x in 0..2 {
                    level.samples.set(UVec3::new(x, y, z), x as f32);
                }
            }
        }
        level.active_cubes = SparseGrid3::new(UVec3::ONE);
        level.active_cubes.set(UVec3::ZERO, ());

        let mut vertices = Vec::new();
        let mut faces = Vec::new();
        let mut mesher = LevelMesher {
            level: &level,
            ranges: &[],
            isovalue: 0.5,
            regularization: 0.0,
            vertices: &mut vertices,
            faces: &mut faces,
            vertex_ids: HashMap::new(),
            face_ids: HashSet::new(),
        };
        mesher.march_cube(UVec3::ZERO).unwrap();

        assert!(!vertices.is_empty());
        assert!(!faces.is_empty());
        assert!(
            vertices
                .iter()
                .all(|vertex| (vertex.position.x - 1.0).abs() < 1.0e-6)
        );
        assert!(faces.iter().all(|face| {
            face.x < vertices.len() as u32
                && face.y < vertices.len() as u32
                && face.z < vertices.len() as u32
        }));
    }

    #[test]
    fn public_api_extracts_from_a_plotfile() -> Result<()> {
        let root =
            std::env::temp_dir().join(format!("amrex_rs_isosurface_test_{}", std::process::id()));
        let _ = fs::remove_dir_all(&root);
        write_single_cube_plotfile(&root)?;

        let result = (|| -> Result<()> {
            let plotfile = PlotFile::open(&root)?;
            let mesh = isosurface(
                &plotfile,
                IsosurfaceOptions {
                    surface: Surface { id: 0, value: 0.5 },
                    sampled_quantities: Vec::new(),
                    regularization: 0.0,
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
