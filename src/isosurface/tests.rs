use std::{fs, fs::File, io::Write, path::Path};

use super::dual_grid::{DualGridLevel, build_active_dual_cubes};
use super::mc33::mesh_level_aabb as mesh_level_mc33_aabb;
use super::rmt::mesh_level_aabb;
use super::sampling::{sampled_values_at, sampled_values_on_edge};
use super::*;
use crate::sparse_amr::level_translation;
use crate::utility::{Aabb3u, SparseGrid3};
use glam::{DVec3, I64Vec3, IVec3, U16Vec2, UVec3};

fn level_with_grids<'a>(
    samples: &'a SparseGrid3<f32>,
    sampled_quantities: Vec<&'a SparseGrid3<f32>>,
    active_cubes: &'a SparseGrid3<()>,
) -> DualGridLevel<'a> {
    DualGridLevel {
        level_index: 0,
        index_origin: IVec3::ZERO,
        physical_origin: DVec3::ZERO,
        cell_size: DVec3::ONE,
        samples,
        sampled_quantities,
        active_cubes: active_cubes.clone(),
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
    shard.write_all(
        b"FAB ((8, (64 11 52 0 1 12 0 1023)),(8, (8 7 6 5 4 3 2 1)))((0,0,0) (1,1,1) (0,0,0)) 1\n",
    )?;
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
    let translation = level_translation(IVec3::new(-4, 3, 10), IVec3::new(-7, 8, 20), 2).unwrap();
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
    let samples = SparseGrid3::new(UVec3::new(2, 1, 1));
    let active_cubes = SparseGrid3::new(UVec3::ZERO);
    let level = level_with_grids(&samples, vec![&u, &v], &active_cubes);
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
    let samples = SparseGrid3::new(UVec3::new(1, 1, 1));
    let active_cubes = SparseGrid3::new(UVec3::ZERO);
    let level = level_with_grids(&samples, vec![&u], &active_cubes);
    let ranges = [SampleRange {
        min: 0.0,
        max: 10.0,
    }];

    let uv = sampled_values_at(&level, UVec3::ZERO, &ranges).unwrap();
    assert_eq!(uv, U16Vec2::new(u16::MAX, 0));
}

#[test]
fn rmt_extracts_a_plane() {
    let mut samples = SparseGrid3::new(UVec3::splat(2));
    for z in 0..2 {
        for y in 0..2 {
            for x in 0..2 {
                samples.set(UVec3::new(x, y, z), x as f32);
            }
        }
    }
    let mut active_cubes = SparseGrid3::new(UVec3::ONE);
    active_cubes.set(UVec3::ZERO, ());
    let level = level_with_grids(&samples, Vec::new(), &active_cubes);

    let mut mesh = Mesh3D {
        positions: Vec::new(),
        faces: Vec::new(),
    };
    mesh_level_aabb(
        &level,
        level.active_cubes.bounds_aabb(),
        &[],
        0.5,
        0.0,
        &mut mesh,
    )
    .unwrap();

    assert!(!mesh.positions.is_empty());
    assert!(!mesh.faces.is_empty());
    assert!(
        mesh.positions
            .iter()
            .all(|vertex| (vertex.position.x - 1.0).abs() < 1.0e-6)
    );
    assert!(mesh.faces.iter().all(|face| {
        face.x < mesh.positions.len() as u32
            && face.y < mesh.positions.len() as u32
            && face.z < mesh.positions.len() as u32
    }));
}

#[test]
fn mc33_extracts_a_plane_without_tetrahedral_diagonals() {
    let mut samples = SparseGrid3::new(UVec3::splat(2));
    for z in 0..2 {
        for y in 0..2 {
            for x in 0..2 {
                samples.set(UVec3::new(x, y, z), x as f32);
            }
        }
    }
    let mut active_cubes = SparseGrid3::new(UVec3::ONE);
    active_cubes.set(UVec3::ZERO, ());
    let level = level_with_grids(&samples, Vec::new(), &active_cubes);

    let mut mesh = Mesh3D {
        positions: Vec::new(),
        faces: Vec::new(),
    };
    mesh_level_mc33_aabb(
        &level,
        level.active_cubes.bounds_aabb(),
        &[],
        0.5,
        &mut mesh,
    )
    .unwrap();

    assert_eq!(mesh.positions.len(), 4);
    assert_eq!(mesh.faces.len(), 2);
    assert!(
        mesh.positions
            .iter()
            .all(|vertex| (vertex.position.x - 1.0).abs() < 1.0e-6)
    );
}

#[test]
fn public_mesher_extracts_across_active_chunks() -> Result<()> {
    let mut samples = SparseGrid3::new(UVec3::new(34, 2, 2));
    samples.fill_aabb(samples.bounds_aabb(), 0.0);
    for z in 0..2 {
        for y in 0..2 {
            for x in 17..34 {
                samples.set(UVec3::new(x, y, z), 1.0);
            }
        }
    }
    let mut active_cubes = SparseGrid3::new(UVec3::new(33, 1, 1));
    active_cubes.fill_aabb(active_cubes.bounds_aabb(), ());
    let level = level_with_grids(&samples, Vec::new(), &active_cubes);
    let compact = CompactPlot {
        variable_count: 1,
        refinement_ratios: Vec::new(),
        component_ids: vec![0],
        levels: vec![crate::compact::CompactLevel {
            level_index: 0,
            index_origin: IVec3::ZERO,
            physical_origin: level.physical_origin,
            cell_size: level.cell_size,
            components: vec![samples],
            eligible_cubes: active_cubes,
        }],
    };

    let mesh = isosurface_compact(
        &compact,
        IsosurfaceOptions {
            surface: Surface { id: 0, value: 0.5 },
            sampled_quantities: Vec::new(),
            method: IsosurfaceMethod::Mc33,
            levels: None,
            flip_winding: Vec::new(),
        },
    )?;

    ensure!(!mesh.positions.is_empty(), "expected extracted vertices");
    ensure!(!mesh.faces.is_empty(), "expected extracted faces");
    Ok(())
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
                method: IsosurfaceMethod::Mc33,
                levels: None,
                flip_winding: Vec::new(),
            },
        )?;
        ensure!(!mesh.positions.is_empty(), "expected extracted vertices");
        ensure!(!mesh.faces.is_empty(), "expected extracted faces");

        let compact = CompactPlot::load(&plotfile, &[0])?;
        let mesh = isosurface_compact(
            &compact,
            IsosurfaceOptions {
                surface: Surface { id: 0, value: 0.5 },
                sampled_quantities: Vec::new(),
                method: IsosurfaceMethod::Mc33,
                levels: None,
                flip_winding: Vec::new(),
            },
        )?;
        ensure!(
            !mesh.positions.is_empty(),
            "expected compact extracted vertices"
        );
        ensure!(!mesh.faces.is_empty(), "expected compact extracted faces");

        let mesh = isosurface_compact(
            &compact,
            IsosurfaceOptions {
                surface: Surface { id: 0, value: 0.5 },
                sampled_quantities: Vec::new(),
                method: IsosurfaceMethod::Mc33,
                levels: Some(0..=0),
                flip_winding: Vec::new(),
            },
        )?;
        ensure!(
            !mesh.positions.is_empty(),
            "expected selected level extracted vertices"
        );

        let mesh = isosurface_compact(
            &compact,
            IsosurfaceOptions {
                surface: Surface { id: 0, value: 0.5 },
                sampled_quantities: Vec::new(),
                method: IsosurfaceMethod::Mc33,
                levels: Some(1..=1),
                flip_winding: Vec::new(),
            },
        )?;
        ensure!(
            mesh.positions.is_empty(),
            "unexpected vertices from omitted level"
        );
        ensure!(mesh.faces.is_empty(), "unexpected faces from omitted level");
        Ok(())
    })();

    let _ = fs::remove_dir_all(&root);
    result
}

#[test]
fn flip_face_winding_swaps_new_face_orientation() {
    let mut faces = [UVec3::new(1, 2, 3), UVec3::new(4, 5, 6)];
    flip_face_winding(&mut faces);
    assert_eq!(faces, [UVec3::new(1, 3, 2), UVec3::new(4, 6, 5)]);
}
