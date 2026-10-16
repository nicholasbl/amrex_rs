//! Conversion from AMReX plotfiles into the source-independent MSG model.

use std::time::{Duration, Instant};

use anyhow::{Context, Result, bail, ensure};
use glam::{DMat4, DVec3, I64Vec3, UVec3};
use multires_rs::{
    MsgDataset, MsgField, MsgFieldLevel, MsgGeometry, MsgLevel, MsgScalarGrid, MsgScalarType,
    cell_center_to_point_geometry, sparse::Aabb3u,
};

use crate::{CoordinateSystem, DataReader, IndexDomain, PlotFile};

/// One AMReX component requested for conversion to MSG.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AmrexMsgField {
    pub component: usize,
    pub scalar_type: MsgScalarType,
}

/// Handling of invalid or unrepresentable AMReX source values.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum AmrexInvalidValuePolicy {
    /// Reject the conversion when a value is non-finite or cannot be represented
    /// by the requested logical scalar precision.
    #[default]
    Reject,
    /// Replace invalid values with physical zero before they enter the MSG model.
    ReplaceWithZero,
}

/// Options for converting one AMReX plotfile timestep into an owned MSG dataset.
#[derive(Debug, Clone, Default)]
pub struct AmrexToMsgOptions {
    /// Requested components. Empty selects every plotfile component as `f32`.
    pub fields: Vec<AmrexMsgField>,
    pub invalid_value_policy: AmrexInvalidValuePolicy,
}

/// Work and invalid-value counts from AMReX-to-MSG conversion.
#[derive(Debug, Clone, Default)]
pub struct AmrexToMsgReport {
    pub elapsed: Duration,
    pub patch_count: usize,
    pub sample_count: usize,
    /// Replacement counts in the same order as [`AmrexToMsgOptions::fields`],
    /// or plotfile component order when all fields were selected.
    pub replaced_values: Vec<usize>,
}

/// Convert selected cell-centered AMReX components into point-associated MSG fields.
///
/// Values are preserved at their cell-center locations; no interpolation or
/// boundary samples are introduced. Non-Cartesian AMReX coordinate systems are
/// rejected because MSG geometry is affine.
pub fn plotfile_to_msg(
    plot_file: &PlotFile,
    options: &AmrexToMsgOptions,
) -> Result<(MsgDataset, AmrexToMsgReport)> {
    let start = Instant::now();
    ensure!(
        matches!(
            plot_file.header().coordinate_system,
            CoordinateSystem::Cartesian
        ),
        "MSG's affine geometry currently supports only Cartesian AMReX plotfiles"
    );

    let requests = resolve_fields(plot_file, options)?;
    let levels = build_levels(plot_file)?;
    let geometry = build_geometry(plot_file, &levels)?;
    let mut dataset = MsgDataset::new(plot_file.header().simulation_time, geometry, levels)
        .context("constructing MSG dataset geometry")?;
    let reader = DataReader::new(plot_file);
    let mut report = AmrexToMsgReport {
        replaced_values: vec![0; requests.len()],
        ..AmrexToMsgReport::default()
    };

    for (request_index, request) in requests.iter().enumerate() {
        let variable = &plot_file.variables()[request.component];
        let mut field_levels = Vec::with_capacity(dataset.levels().len());
        for level in plot_file.levels() {
            let domain = level.index_domain();
            ensure!(
                domain.index_type == glam::IVec3::ZERO,
                "AMReX component {:?} level {} is not cell-centered",
                variable.name,
                level.index()
            );
            let extent = index_extent(domain)?;
            let mut values = MsgScalarGrid::new(extent, request.scalar_type);
            for patch in level.patches()? {
                report.patch_count += 1;
                let patch_box = patch.index_box();
                ensure!(
                    patch_box.index_type == domain.index_type,
                    "patch {} centering differs from level {}",
                    patch.index(),
                    level.index()
                );
                let aabb = local_aabb(patch_box, domain)?;
                let source = reader.component(patch, request.component)?;
                let expected = aabb
                    .volume_usize()
                    .context("AMReX patch sample count exceeds usize")?;
                let mut decoded = Vec::with_capacity(expected);
                for z in patch_box.min.z..=patch_box.max.z {
                    for y in patch_box.min.y..=patch_box.max.y {
                        for x in patch_box.min.x..=patch_box.max.x {
                            let source_value =
                                source.get(glam::IVec3::new(x, y, z)).with_context(|| {
                                    format!(
                                        "component {} value is absent at ({x}, {y}, {z})",
                                        request.component
                                    )
                                })?;
                            decoded.push(sanitize_value(
                                source_value,
                                request.scalar_type,
                                options.invalid_value_policy,
                                &mut report.replaced_values[request_index],
                            )?);
                        }
                    }
                }
                values
                    .write_f64_values(aabb, &decoded)
                    .map_err(|error| anyhow::anyhow!("writing sparse MSG patch: {error:?}"))?;
                report.sample_count += decoded.len();
            }
            field_levels.push(MsgFieldLevel::new(values));
        }
        dataset
            .add_field(MsgField::new(
                variable.name.clone(),
                request.scalar_type,
                field_levels,
            ))
            .with_context(|| format!("adding MSG field {:?}", variable.name))?;
    }

    dataset
        .rebuild_acceleration()
        .context("building per-field MSG acceleration")?;
    report.elapsed = start.elapsed();
    Ok((dataset, report))
}

fn resolve_fields(plot_file: &PlotFile, options: &AmrexToMsgOptions) -> Result<Vec<AmrexMsgField>> {
    let requests = if options.fields.is_empty() {
        (0..plot_file.variables().len())
            .map(|component| AmrexMsgField {
                component,
                scalar_type: MsgScalarType::F32,
            })
            .collect::<Vec<_>>()
    } else {
        options.fields.clone()
    };
    for (index, request) in requests.iter().enumerate() {
        ensure!(
            request.component < plot_file.variables().len(),
            "AMReX component {} is out of range",
            request.component
        );
        ensure!(
            !requests[..index]
                .iter()
                .any(|other| other.component == request.component),
            "AMReX component {} was requested more than once",
            request.component
        );
    }
    Ok(requests)
}

fn build_levels(plot_file: &PlotFile) -> Result<Vec<MsgLevel>> {
    let mut output = Vec::with_capacity(plot_file.header().index_domains.len());
    for level in plot_file.levels() {
        let domain = level.index_domain();
        ensure!(
            domain.index_type == glam::IVec3::ZERO,
            "level {} is not cell-centered",
            level.index()
        );
        let refinement_ratio_from_coarse = if level.index() == 0 {
            None
        } else {
            Some(derive_refinement_ratio(
                plot_file.header().cell_sizes[level.index() - 1],
                level.cell_size(),
                level.index(),
            )?)
        };
        output.push(MsgLevel {
            index_origin: domain.min,
            point_extent: index_extent(domain)?,
            scale: level.cell_size(),
            refinement_ratio_from_coarse,
        });
    }
    Ok(output)
}

fn build_geometry(plot_file: &PlotFile, levels: &[MsgLevel]) -> Result<MsgGeometry> {
    let base = levels.first().context("plotfile has no AMR levels")?;
    let translation = plot_file.header().domain.min - base.index_origin.as_dvec3() * base.scale;
    cell_center_to_point_geometry(DMat4::from_translation(translation))
        .context("constructing cell-center-to-point MSG geometry")
}

fn derive_refinement_ratio(coarse: DVec3, fine: DVec3, level: usize) -> Result<UVec3> {
    ensure!(
        coarse.is_finite() && fine.is_finite(),
        "level scales must be finite"
    );
    ensure!(
        coarse.cmpgt(DVec3::ZERO).all() && fine.cmpgt(DVec3::ZERO).all(),
        "level scales must be positive"
    );
    let ratio = coarse / fine;
    let rounded = ratio.round();
    let relative = (ratio - rounded).abs() / rounded;
    ensure!(
        rounded.cmpge(DVec3::ONE).all()
            && relative.cmple(DVec3::splat(1.0e-10)).all()
            && rounded.cmple(DVec3::splat(f64::from(u32::MAX))).all(),
        "level {level} has a non-integral refinement ratio {ratio:?}"
    );
    Ok(rounded.as_uvec3())
}

fn sanitize_value(
    value: f64,
    scalar_type: MsgScalarType,
    policy: AmrexInvalidValuePolicy,
    replacement_count: &mut usize,
) -> Result<f64> {
    let representable = value.is_finite()
        && match scalar_type {
            MsgScalarType::F32 => (value as f32).is_finite(),
            MsgScalarType::F64 => true,
        };
    if representable {
        return Ok(value);
    }
    match policy {
        AmrexInvalidValuePolicy::Reject => {
            bail!("source value {value} is non-finite or cannot be represented as {scalar_type:?}")
        }
        AmrexInvalidValuePolicy::ReplaceWithZero => {
            *replacement_count += 1;
            Ok(0.0)
        }
    }
}

fn index_extent(domain: &IndexDomain) -> Result<UVec3> {
    let extent = domain.max.as_i64vec3() - domain.min.as_i64vec3() + I64Vec3::ONE;
    ensure!(
        extent.cmpgt(I64Vec3::ZERO).all(),
        "AMReX index domain is empty"
    );
    Ok(UVec3::new(
        u32::try_from(extent.x).context("x extent exceeds u32")?,
        u32::try_from(extent.y).context("y extent exceeds u32")?,
        u32::try_from(extent.z).context("z extent exceeds u32")?,
    ))
}

fn local_aabb(index_box: &IndexDomain, domain: &IndexDomain) -> Result<Aabb3u> {
    ensure!(
        index_box.min.cmpge(domain.min).all() && index_box.max.cmple(domain.max).all(),
        "AMReX patch lies outside its level domain"
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
        for value in 0..8 {
            shard.write_all(&(value as f64).to_le_bytes())?;
        }
        Ok(())
    }

    #[test]
    fn derives_anisotropic_refinement() {
        assert_eq!(
            derive_refinement_ratio(DVec3::new(4.0, 6.0, 8.0), DVec3::new(2.0, 2.0, 2.0), 1)
                .unwrap(),
            UVec3::new(2, 3, 4)
        );
    }

    #[test]
    fn invalid_values_are_explicitly_rejected_or_replaced() {
        let mut replacements = 0;
        assert!(
            sanitize_value(
                f64::INFINITY,
                MsgScalarType::F64,
                AmrexInvalidValuePolicy::Reject,
                &mut replacements
            )
            .is_err()
        );
        assert_eq!(
            sanitize_value(
                f64::NAN,
                MsgScalarType::F32,
                AmrexInvalidValuePolicy::ReplaceWithZero,
                &mut replacements
            )
            .unwrap(),
            0.0
        );
        assert_eq!(replacements, 1);
    }

    #[test]
    fn converts_cell_centers_and_round_trips_msg() -> Result<()> {
        let root =
            std::env::temp_dir().join(format!("amrex_rs_to_msg_test_{}", std::process::id()));
        let _ = fs::remove_dir_all(&root);
        write_single_cube_plotfile(&root)?;
        let result = (|| -> Result<()> {
            let plotfile = PlotFile::open(&root)?;
            let (dataset, report) = plotfile_to_msg(&plotfile, &AmrexToMsgOptions::default())?;
            ensure!(report.sample_count == 8, "unexpected sample count");
            ensure!(
                dataset.geometry().sample_offset() == DVec3::splat(0.5),
                "cell-center conversion lost its half-cell offset"
            );
            let mut bytes = Vec::new();
            multires_rs::write_msg(
                &dataset,
                &multires_rs::MsgWriteOptions::default(),
                &mut bytes,
            )?;
            let decoded = multires_rs::read_msg(&bytes)?;
            ensure!(
                decoded
                    .field("density")
                    .context("density is absent")?
                    .levels()[0]
                    .values()
                    .get(UVec3::new(1, 1, 1))
                    == Some(7.0_f32.into()),
                "MSG round trip changed the AMReX value"
            );
            Ok(())
        })();
        let _ = fs::remove_dir_all(&root);
        result
    }
}
