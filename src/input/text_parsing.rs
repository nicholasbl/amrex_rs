use std::{
    fs::File,
    io::{BufRead, BufReader},
    path::Path,
};

use anyhow::{Context, Result, bail};

pub(crate) struct Reader {
    reader: BufReader<File>,
    buffer: String,
}

impl Reader {
    pub(crate) fn new(path: impl AsRef<Path>) -> Result<Self> {
        let reader = BufReader::new(File::open(path).context("opening file")?);

        Ok(Self {
            reader,
            buffer: String::default(),
        })
    }

    pub(crate) fn next_line(&mut self) -> Result<Option<&str>> {
        self.buffer.clear();
        let res = self.reader.read_line(&mut self.buffer)?;

        if res > 0 {
            Ok(Some(self.buffer.as_str().trim()))
        } else {
            Ok(None)
        }
    }

    pub(crate) fn demand_next_line(&mut self) -> Result<&str> {
        let Some(text) = self.next_line()? else {
            bail!("unable to read next line");
        };

        Ok(text)
    }
}

pub(crate) struct LineChomper<'a> {
    line: &'a str,
}

impl<'a> LineChomper<'a> {
    pub(crate) fn new(s: &'a str) -> Self {
        Self { line: s.trim() }
    }

    pub(crate) fn skip(&mut self, c: char) {
        while let Some(s) = self.line.strip_prefix(c) {
            self.line = s;
        }
    }
    pub(crate) fn skip_whitespace(&mut self) {
        self.skip(' ');
    }

    pub(crate) fn take_till(&mut self, c: char) -> &str {
        self.skip_whitespace();
        if let Some((before, after)) = self.line.split_once(c) {
            self.line = after.trim();
            before.trim()
        } else {
            Default::default()
        }
    }

    pub(crate) fn has_more(&self) -> bool {
        !self.line.is_empty()
    }
}

pub(crate) fn read_parsed<T>(lines: &mut Reader) -> Result<T>
where
    T::Err: std::error::Error,
    T: std::str::FromStr,
    T::Err: Send + Sync + 'static,
{
    let text = lines.demand_next_line()?;

    Ok(text.parse()?)
}

#[derive(Debug, Clone, Copy)]
pub(crate) enum Delimiter {
    Whitespace,
    Char(char),
}

pub(crate) fn read_fixed_array<T, const N: usize>(
    delim: Delimiter,
    lines: &mut Reader,
) -> Result<[T; N]>
where
    T::Err: std::error::Error,
    T: std::str::FromStr,
    T::Err: Send + Sync + 'static,
{
    let text = lines.demand_next_line()?;

    read_fixed_array_from_str(delim, text)
}

pub(crate) fn read_fixed_array_from_str<T, const N: usize>(
    delim: Delimiter,
    line: &str,
) -> Result<[T; N]>
where
    T::Err: std::error::Error,
    T: std::str::FromStr,
    T::Err: Send + Sync + 'static,
{
    let vec: Result<Vec<_>, _> = match delim {
        Delimiter::Whitespace => line.split_whitespace().map(|x| x.parse::<T>()).collect(),
        Delimiter::Char(c) => line.split(c).map(|x| x.parse::<T>()).collect(),
    };

    let vec = vec.context("reading fixed array")?;

    let Ok(ret) = vec.try_into() else {
        bail!("Unable to obtain {N} values for array");
    };

    Ok(ret)
}

pub(crate) fn read_dyn_array<T>(delim: Delimiter, lines: &mut Reader) -> Result<Vec<T>>
where
    T::Err: std::error::Error,
    T: std::str::FromStr,
    T::Err: Send + Sync + 'static,
{
    let text = lines.demand_next_line()?;

    let vec: Result<Vec<_>, _> = match delim {
        Delimiter::Whitespace => text.split_whitespace().map(|x| x.parse::<T>()).collect(),
        Delimiter::Char(c) => text.split(c).map(|x| x.parse::<T>()).collect(),
    };

    vec.context("reading dynamic array")
}
