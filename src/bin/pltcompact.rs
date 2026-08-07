use std::{env, fs::File, io::BufWriter, path::PathBuf};

use amrex_rs::{CompactOptions, PlotFile, write_compact};
use anyhow::{Context, Result, bail};

fn main() -> Result<()> {
    let args = Args::parse()?;

    let plotfile = PlotFile::open(&args.input)
        .with_context(|| format!("opening plotfile {}", args.input.display()))?;

    let ids: Vec<_> = args
        .vars
        .into_iter()
        .filter_map(|x| {
            let r = plotfile.variable(&x).map(|y| y.index as u32);
            if r.is_none() {
                println!("Variable {x} not found; check spelling or caps");
            }
            r
        })
        .collect();

    let file = File::create(&args.output)
        .with_context(|| format!("creating compact archive {}", args.output.display()))?;
    let mut writer = BufWriter::new(file);

    write_compact(
        &plotfile,
        CompactOptions { component_ids: ids },
        &mut writer,
    )
    .with_context(|| format!("writing compact archive {}", args.output.display()))?;

    Ok(())
}

struct Args {
    input: PathBuf,
    output: PathBuf,
    vars: Vec<String>,
}

impl Args {
    fn parse() -> Result<Self> {
        const USAGE: &str = "usage: pltcompact <variable1> <variable2> <input_plt> <output_archive>\nSpecifying no variables implies all will be captured";
        let mut args: Vec<_> = env::args().skip(1).collect();

        let Some(output) = args.pop() else {
            bail!("missing output: {USAGE}");
        };

        let Some(input) = args.pop() else {
            bail!("missing input: {USAGE}");
        };

        Ok(Self {
            input: input.into(),
            output: output.into(),
            vars: args,
        })
    }
}
