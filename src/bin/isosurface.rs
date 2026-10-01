use std::{
    env,
    fs::{self, File},
    io::{BufWriter, Write},
    ops::RangeInclusive,
    path::{Path, PathBuf},
    time::{Duration, Instant},
};

use amrex_rs::{
    CompactPlot, IsosurfaceOptions, Mesh3D, PlotFile, Sample, Surface, Variable,
    isosurface::IsosurfaceTimings, isosurface_compact, read_compact,
};
use anyhow::{Context, Result, bail, ensure};

struct Args {
    input: PathBuf,
    variable: String,
    isovalue: f64,
    output: PathBuf,
    samples: Vec<(String, RangeInclusive<f64>)>,
}

fn usage(program: &str) -> String {
    format!(
        "Usage: {program} <input> <variable> <isovalue> <output.obj> [options]\n\
         \n\
         <input> may be an AMReX plotfile directory or a compact archive.\n\
         \n\
         Options:\n\
           --sample <variable> <min> <max>  Map a quantity to U or V (repeat at most twice)\n\
           -h, --help                       Show this help"
    )
}

fn parse_args() -> Result<Option<Args>> {
    let mut values = env::args();
    let program = values.next().unwrap_or_else(|| "isosurface".to_string());
    let values = values.collect::<Vec<_>>();

    if values
        .iter()
        .any(|value| value == "-h" || value == "--help")
    {
        println!("{}", usage(&program));
        return Ok(None);
    }
    ensure!(values.len() >= 4, "{}", usage(&program));

    let input = PathBuf::from(&values[0]);
    let variable = values[1].clone();
    let isovalue = values[2]
        .parse::<f64>()
        .with_context(|| format!("invalid isovalue {:?}", values[2]))?;
    let output = PathBuf::from(&values[3]);
    let mut samples = Vec::new();
    let mut index = 4;

    while index < values.len() {
        match values[index].as_str() {
            "--sample" => {
                ensure!(
                    index + 3 < values.len(),
                    "--sample requires VARIABLE MIN MAX"
                );
                ensure!(samples.len() < 2, "--sample may be specified at most twice");
                let name = values[index + 1].clone();
                let min = values[index + 2]
                    .parse::<f64>()
                    .with_context(|| format!("invalid sample minimum {:?}", values[index + 2]))?;
                let max = values[index + 3]
                    .parse::<f64>()
                    .with_context(|| format!("invalid sample maximum {:?}", values[index + 3]))?;
                samples.push((name, min..=max));
                index += 4;
            }
            option => bail!("unknown option {option:?}\n{}", usage(&program)),
        }
    }

    Ok(Some(Args {
        input,
        variable,
        isovalue,
        output,
        samples,
    }))
}

fn component_id(variables: &[Variable], name: &str) -> Result<u32> {
    let variable = variables
        .iter()
        .find(|variable| variable.name == name)
        .with_context(|| format!("input has no variable named {name:?}"))?;
    u32::try_from(variable.index).context("variable index does not fit in u32")
}

fn isosurface_options(variables: &[Variable], args: &Args) -> Result<IsosurfaceOptions> {
    let surface_id = component_id(variables, &args.variable)?;
    let sampled_quantities = args
        .samples
        .iter()
        .map(|(name, range)| {
            Ok(Sample {
                id: component_id(variables, name)?,
                range: range.clone(),
            })
        })
        .collect::<Result<Vec<_>>>()?;

    Ok(IsosurfaceOptions {
        surface: Surface {
            id: surface_id,
            value: args.isovalue,
        },
        sampled_quantities,
        levels: None,
        flip_winding: false,
    })
}

fn load_compact(path: &Path) -> Result<CompactPlot> {
    let file = std::fs::File::open(path)
        .with_context(|| format!("reading compact archive {}", path.display()))?;

    let mmap = unsafe { memmap2::Mmap::map(&file).context("memory mapping file") }?;

    read_compact(&mmap).with_context(|| format!("reading compact archive {}", path.display()))
}

pub struct ExtractionTimings {
    pub input_load_time: Duration,
    pub isosurface_time: IsosurfaceTimings,
}

fn extract_isosurface(args: &Args) -> Result<(Mesh3D, ExtractionTimings)> {
    if args.input.is_dir() {
        let now = Instant::now();
        let plotfile = PlotFile::open(&args.input)
            .with_context(|| format!("opening plotfile {}", args.input.display()))?;
        let options = isosurface_options(plotfile.variables(), args)?;
        let input_load_time = now.elapsed();
        amrex_rs::isosurface(&plotfile, options).map(|(m, t)| {
            (
                m,
                ExtractionTimings {
                    input_load_time,
                    isosurface_time: t,
                },
            )
        })
    } else {
        let now = Instant::now();
        let compact = load_compact(&args.input)?;
        let options = isosurface_options(compact.variables(), args)?;
        let input_load_time = now.elapsed();

        isosurface_compact(&compact, options).map(|(m, t)| {
            (
                m,
                ExtractionTimings {
                    input_load_time,
                    isosurface_time: t,
                },
            )
        })
    }
}

fn write_obj(path: &Path, mesh: &Mesh3D, write_uvs: bool) -> Result<()> {
    let file = File::create(path).with_context(|| format!("creating {}", path.display()))?;
    let mut output = BufWriter::new(file);
    writeln!(output, "# generated by amrex_rs isosurface")?;

    for position in &mesh.positions {
        writeln!(output, "v {} {} {}", position[0], position[1], position[2])?;
    }
    if write_uvs {
        for uv in &mesh.uv {
            writeln!(output, "vt {} {}", uv[0], uv[1])?;
        }
    }

    for face in &mesh.indices {
        let indices = face.map(|index| u64::from(index) + 1);
        if write_uvs {
            writeln!(
                output,
                "f {0}/{0} {1}/{1} {2}/{2}",
                indices[0], indices[1], indices[2]
            )?;
        } else {
            writeln!(output, "f {} {} {}", indices[0], indices[1], indices[2])?;
        }
    }
    output.flush()?;
    Ok(())
}

fn main() -> Result<()> {
    let Some(args) = parse_args()? else {
        return Ok(());
    };
    let (mesh, timings) = extract_isosurface(&args)?;
    write_obj(&args.output, &mesh, !args.samples.is_empty())?;
    eprintln!(
        "wrote {} vertices and {} faces to {}",
        mesh.positions.len(),
        mesh.indices.len(),
        args.output.display()
    );

    eprintln!(
        "timings: input_load {}, plotfile_compact_load {}, dual_grid {}",
        timings.input_load_time.as_secs_f32(),
        timings
            .isosurface_time
            .plotfile_compact_load_time
            .as_secs_f32(),
        timings.isosurface_time.dual_grid_time.as_secs_f32()
    );

    for (level_i, per_level_timings) in timings.isosurface_time.per_level_timings.iter().enumerate()
    {
        eprintln!(
            "timings, level {}: mesh {}, merge {}, wind: {}",
            level_i,
            per_level_timings.mesh_generation_time.as_secs_f32(),
            per_level_timings.merge_in_time.as_secs_f32(),
            per_level_timings.winding_time.as_secs_f32()
        );
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use std::fs;

    use super::*;

    #[test]
    fn obj_writer_emits_vertices_uvs_and_matching_indices() {
        let path = env::temp_dir().join(format!("amrex_rs_obj_test_{}.obj", std::process::id()));
        let mesh = Mesh3D {
            positions: vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            uv: vec![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
            indices: vec![[0, 1, 2]],
        };

        write_obj(&path, &mesh, true).unwrap();
        let text = fs::read_to_string(&path).unwrap();
        let _ = fs::remove_file(&path);
        assert!(text.contains("v 1 0 0\n"));
        assert!(text.contains("vt 1 0\n"));
        assert!(text.contains("f 1/1 2/2 3/3\n"));
    }
}
