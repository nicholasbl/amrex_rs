use std::io::Write;

use crate::PlotFile;

use anyhow::Result;

#[derive(Debug, Default)]
pub struct CompressionOptions {
    quantity_ids: Vec<u32>,
}

fn compress(
    plot_file: &PlotFile,
    options: CompressionOptions,
    dest: &mut impl Write,
) -> Result<()> {
    // our file format

    todo!()
}
