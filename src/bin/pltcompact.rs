use std::{
    env,
    fs::{self, File},
    io::{BufWriter, Write},
    path::PathBuf,
};

use amrex_rs::{CompactNormalization, CompactOptions, PlotFile, write_compact};
use anyhow::{Context, Result, bail, ensure};
use serde::Deserialize;

fn main() -> Result<()> {
    let args = Args::parse()?;

    let plotfile = PlotFile::open(&args.input)
        .with_context(|| format!("opening plotfile {}", args.input.display()))?;

    let mut ids = Vec::with_capacity(args.variables.len());
    let mut normalizations = Vec::new();
    for requested in args.variables {
        let variable = plotfile
            .variable(&requested.name)
            .with_context(|| format!("variable {:?} was not found", requested.name))?;
        let component_id = u32::try_from(variable.index).context("component id exceeds u32")?;
        ensure!(
            !ids.contains(&component_id),
            "variable {:?} was requested more than once",
            requested.name
        );
        ids.push(component_id);
        if let Some([min, max]) = requested.normalize {
            normalizations.push(CompactNormalization {
                component_id,
                min,
                max,
            });
        }
    }

    let file = File::create(&args.output)
        .with_context(|| format!("creating compact archive {}", args.output.display()))?;
    let mut writer = BufWriter::new(file);

    let timings = write_compact(
        &plotfile,
        CompactOptions {
            component_ids: ids,
            normalizations,
        },
        &mut writer,
    )
    .with_context(|| format!("writing compact archive {}", args.output.display()))?;
    writer
        .flush()
        .with_context(|| format!("flushing compact archive {}", args.output.display()))?;

    eprintln!(
        "timings: compact {}, archive {}, serialization {}",
        timings.compact_time.as_secs_f32(),
        timings.archive_time.as_secs_f32(),
        timings.serialization_time.as_secs_f32()
    );

    Ok(())
}

struct Args {
    input: PathBuf,
    output: PathBuf,
    variables: Vec<VariableConfig>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Config {
    input: Option<PathBuf>,
    output: Option<PathBuf>,
    #[serde(default)]
    variables: Vec<VariableConfig>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct VariableConfig {
    name: String,
    normalize: Option<[f64; 2]>,
}

impl Args {
    fn parse() -> Result<Self> {
        const USAGE: &str = "usage:\n  pltcompact <variable1> <variable2> ... <input_plt> <output_archive>\n  pltcompact --config <config.toml> [<input_plt> <output_archive>]\nCommand-line input and output override paths in the config. Specifying no variables implies all will be captured";
        let mut args = env::args().skip(1).collect::<Vec<_>>();

        if args.iter().any(|arg| arg == "-h" || arg == "--help") {
            println!("{USAGE}");
            std::process::exit(0);
        }

        if args.first().is_some_and(|arg| arg == "--config") {
            ensure!(args.len() == 2 || args.len() == 4, "{USAGE}");
            let path = PathBuf::from(&args[1]);
            let source = fs::read_to_string(&path)
                .with_context(|| format!("reading config {}", path.display()))?;
            let config: Config = toml::from_str(&source)
                .with_context(|| format!("parsing config {}", path.display()))?;
            let (input, output) = if args.len() == 4 {
                (PathBuf::from(&args[2]), PathBuf::from(&args[3]))
            } else {
                (
                    config
                        .input
                        .context("config has no input; provide one on the command line")?,
                    config
                        .output
                        .context("config has no output; provide one on the command line")?,
                )
            };
            return Ok(Self {
                input,
                output,
                variables: config.variables,
            });
        }

        let Some(output) = args.pop() else {
            bail!("missing output: {USAGE}");
        };

        let Some(input) = args.pop() else {
            bail!("missing input: {USAGE}");
        };

        Ok(Self {
            input: input.into(),
            output: output.into(),
            variables: args
                .into_iter()
                .map(|name| VariableConfig {
                    name,
                    normalize: None,
                })
                .collect(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_toml_variable_normalization() -> Result<()> {
        let config: Config = toml::from_str(
            r#"
                input = "plt00010"
                output = "plt00010.compact"

                [[variables]]
                name = "density"
                normalize = [1.0e20, 1.000000001e20]

                [[variables]]
                name = "temperature"
            "#,
        )?;

        ensure!(config.variables.len() == 2, "unexpected variable count");
        ensure!(
            config.input == Some(PathBuf::from("plt00010")),
            "input path changed while parsing"
        );
        ensure!(
            config.output == Some(PathBuf::from("plt00010.compact")),
            "output path changed while parsing"
        );
        ensure!(
            config.variables[0].normalize == Some([1.0e20, 1.000000001e20]),
            "normalization bounds changed while parsing"
        );
        ensure!(
            config.variables[1].normalize.is_none(),
            "optional normalization unexpectedly present"
        );
        Ok(())
    }

    #[test]
    fn config_paths_are_optional() -> Result<()> {
        let config: Config = toml::from_str(
            r#"
                [[variables]]
                name = "density"
                normalize = [0.0, 1.0]
            "#,
        )?;

        ensure!(config.input.is_none(), "unexpected config input");
        ensure!(config.output.is_none(), "unexpected config output");
        ensure!(config.variables.len() == 1, "unexpected variable count");
        Ok(())
    }
}
