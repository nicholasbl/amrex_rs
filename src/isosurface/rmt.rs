//! Regularized marching tetrahedra (RMT) isosurface extraction.

use std::collections::{HashMap, HashSet};

use anyhow::{Context, Result};
use glam::{DVec3, UVec3, Vec3};

use crate::utility::Aabb3u;

use super::dual_grid::DualGridLevel;
use super::sampling::{sampled_values_at, sampled_values_on_edge};
use super::{Mesh3D, SampleRange, Vertex3D};

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

struct RmtMesher<'a> {
    level: &'a DualGridLevel<'a>,
    ranges: &'a [SampleRange],
    isovalue: f32,
    regularization: f32,
    vertices: &'a mut Vec<Vertex3D>,
    faces: &'a mut Vec<UVec3>,
    vertex_ids: HashMap<VertexKey, u32>,
    face_ids: HashSet<[u32; 3]>,
}

impl RmtMesher<'_> {
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

pub(super) fn mesh_level_aabb(
    level: &DualGridLevel<'_>,
    active_aabb: Aabb3u,
    ranges: &[SampleRange],
    isovalue: f32,
    regularization: f32,
    mesh: &mut Mesh3D,
) -> Result<()> {
    let mut mesher = RmtMesher {
        level,
        ranges,
        isovalue,
        regularization,
        vertices: &mut mesh.positions,
        faces: &mut mesh.faces,
        vertex_ids: HashMap::new(),
        face_ids: HashSet::new(),
    };
    let mut error = None;
    level
        .active_cubes
        .for_each_present_in_aabb(active_aabb, |anchor, ()| {
            if error.is_none() {
                error = mesher.march_cube(anchor).err();
            }
        });
    match error {
        Some(error) => Err(error),
        None => Ok(()),
    }
}
