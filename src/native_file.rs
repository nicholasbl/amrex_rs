use std::io::{BufRead, BufReader, Read};

use anyhow::{Context, Result, bail};

struct Header {
    generator: String,
    variables: Vec<Variable>,
}

struct Variable {
    name: String,
    index: usize,
}

fn read_header(reader: impl Read) -> Result<Header> {
    let reader = BufReader::new(reader);

    let mut lines = reader.lines();

    // first line is a string
    let Some(generator) = lines.next() else {
        bail!("missing generator line")
    };

    let generator = generator?;

    // next is the number of quantities

    let Some(num_quant) = lines.next() else {
        bail!("missing number of quantities");
    };

    let num_quant = num_quant?.parse::<usize>()?;

    return Header;
}
