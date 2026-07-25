use anyhow::{Context, Result, bail};

fn main() -> Result<()> {
    let Some(path) = std::env::args().next_back() else {
        bail!("Missing path to header")
    };

    let header = amrex_rs::read_header(&path).context("reading header")?;

    println!("Header: {header:?}");

    Ok(())
}
