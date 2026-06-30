use std::{
    collections::HashMap,
    fs::File,
    io::{BufRead, BufReader},
    path::{Path, PathBuf},
    sync::{Arc, OnceLock},
};

use anyhow::{Context, Result, anyhow, bail};

use glam::prelude::*;

struct Reader {
    reader: BufReader<File>,
    buffer: String,
}

impl Reader {
    fn new(path: impl AsRef<Path>) -> Result<Self> {
        let reader = BufReader::new(File::open(path).context("opening file")?);

        Ok(Self {
            reader,
            buffer: String::default(),
        })
    }

    fn next_line(&mut self) -> Result<Option<&str>> {
        self.buffer.clear();
        let res = self.reader.read_line(&mut self.buffer)?;

        if res > 0 {
            Ok(Some(self.buffer.as_str().trim()))
        } else {
            Ok(None)
        }
    }

    fn demand_next_line(&mut self) -> Result<&str> {
        let Some(text) = self.next_line()? else {
            bail!("unable to read next line");
        };

        Ok(text)
    }
}

struct LineChomper<'a> {
    line: &'a str,
}

impl<'a> LineChomper<'a> {
    fn new(s: &'a str) -> Self {
        Self { line: s.trim() }
    }

    fn skip(&mut self, c: char) {
        while let Some(s) = self.line.strip_prefix(c) {
            self.line = s;
        }
    }
    fn skip_whitespace(&mut self) {
        self.skip(' ');
    }

    fn take_till(&mut self, c: char) -> &str {
        self.skip_whitespace();
        if let Some((before, after)) = self.line.split_once(c) {
            self.line = after.trim();
            before.trim()
        } else {
            Default::default()
        }
    }

    fn has_more(&self) -> bool {
        !self.line.is_empty()
    }
}

#[derive(Debug)]
pub struct BoundingBox {
    pub min: DVec3,
    pub max: DVec3,
}

#[derive(Debug, Clone)]
pub struct IndexDomain {
    pub min: IVec3,
    pub max: IVec3, // inclusive
    pub index_type: IVec3,
}

#[derive(Debug)]
pub struct Header {
    pub simulation_time: f64,
    pub finest_level: usize,
    pub domain: BoundingBox,
    pub variables: Vec<Variable>,
    pub refinement_ratios: Vec<usize>,
    pub index_domains: Vec<IndexDomain>,
    pub level_steps: Vec<usize>,
    pub cell_sizes: Vec<DVec3>,
    pub coordinate_system: CoordinateSystem,
    pub boundary_width: usize,
    pub level_records: Vec<PerLevelRecord>,
}

#[derive(Debug)]
pub struct Variable {
    pub name: String,
    pub index: usize,
}

#[derive(Debug)]
pub enum CoordinateSystem {
    Cartesian,
    Cylindrical,
    Spherical,
}

#[derive(Debug)]
pub struct PerLevelRecord {
    pub level_index: usize,
    pub number_of_grid_patches: usize,
    pub simulation_time: f64,
    pub level_step: usize,
    pub grids: Vec<BoundingBox>,
    pub path: String,
}

/// A read-only view of an AMReX plotfile directory.
pub struct PlotFile {
    root: PathBuf,
    header: Header,
    cell_headers: Vec<OnceLock<CellHeader>>,
}

impl PlotFile {
    /// Open a plotfile and eagerly read its top-level `Header`.
    /// Per-level `Cell_H` metadata is loaded on first access.
    pub fn open(root: impl AsRef<Path>) -> Result<Self> {
        let root = root.as_ref().to_path_buf();
        let header = read_header(root.join("Header"))?;
        let cell_headers = (0..header.level_records.len())
            .map(|_| OnceLock::new())
            .collect();

        Ok(Self {
            root,
            header,
            cell_headers,
        })
    }

    pub fn header(&self) -> &Header {
        &self.header
    }

    pub fn variables(&self) -> &[Variable] {
        &self.header.variables
    }

    pub fn variable(&self, name: &str) -> Option<&Variable> {
        self.header
            .variables
            .iter()
            .find(|variable| variable.name == name)
    }

    pub fn level(&self, index: usize) -> Option<Level<'_>> {
        (index < self.header.level_records.len()).then_some(Level {
            plot_file: self,
            index,
        })
    }

    pub fn levels(&self) -> Levels<'_> {
        Levels {
            plot_file: self,
            next: 0,
        }
    }

    /// Load and cross-check every per-level `Cell_H` file.
    pub fn validate_all(&self) -> Result<()> {
        for level in self.levels() {
            let _ = level.patches()?;
        }
        Ok(())
    }

    fn cell_header(&self, level: usize) -> Result<&CellHeader> {
        let cache = self
            .cell_headers
            .get(level)
            .context("level index is out of range")?;

        if let Some(header) = cache.get() {
            return Ok(header);
        }

        let record = &self.header.level_records[level];
        let path = self.root.join(format!("{}_H", record.path));
        let parsed = read_hyper_cell_header(path)
            .with_context(|| format!("reading Cell_H metadata for level {level}"))?;
        validate_cell_header(&self.header, level, &parsed)?;

        // A concurrent reader may have populated the cache while this file was parsed.
        let _ = cache.set(parsed);
        Ok(cache.get().expect("Cell_H cache was just initialized"))
    }
}

pub struct Levels<'a> {
    plot_file: &'a PlotFile,
    next: usize,
}

impl<'a> Iterator for Levels<'a> {
    type Item = Level<'a>;

    fn next(&mut self) -> Option<Self::Item> {
        let index = self.next;
        if index >= self.plot_file.header.level_records.len() {
            return None;
        }
        self.next += 1;
        Some(Level {
            plot_file: self.plot_file,
            index,
        })
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.plot_file.header.level_records.len() - self.next;
        (remaining, Some(remaining))
    }
}

impl ExactSizeIterator for Levels<'_> {}

#[derive(Clone, Copy)]
pub struct Level<'a> {
    plot_file: &'a PlotFile,
    index: usize,
}

impl<'a> Level<'a> {
    pub fn index(self) -> usize {
        self.index
    }

    pub fn time(self) -> f64 {
        self.record().simulation_time
    }

    pub fn step(self) -> usize {
        self.record().level_step
    }

    pub fn cell_size(self) -> DVec3 {
        self.plot_file.header.cell_sizes[self.index]
    }

    pub fn index_domain(self) -> &'a IndexDomain {
        &self.plot_file.header.index_domains[self.index]
    }

    pub fn patch_count(self) -> usize {
        self.record().number_of_grid_patches
    }

    pub fn patches(self) -> Result<Patches<'a>> {
        let cell_header = self.plot_file.cell_header(self.index)?;
        Ok(Patches {
            physical_bounds: &self.record().grids,
            cell_header,
            next: 0,
        })
    }

    fn record(self) -> &'a PerLevelRecord {
        &self.plot_file.header.level_records[self.index]
    }
}

pub struct Patches<'a> {
    physical_bounds: &'a [BoundingBox],
    cell_header: &'a CellHeader,
    next: usize,
}

impl<'a> Iterator for Patches<'a> {
    type Item = Patch<'a>;

    fn next(&mut self) -> Option<Self::Item> {
        let physical_bounds = self.physical_bounds.get(self.next)?;
        let fab = &self.cell_header.fabs[self.next];
        let index = self.next;
        self.next += 1;
        Some(Patch {
            index,
            physical_bounds,
            fab,
            bounds: &self.cell_header.bounds,
        })
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.physical_bounds.len() - self.next;
        (remaining, Some(remaining))
    }
}

impl ExactSizeIterator for Patches<'_> {}

#[derive(Clone, Copy)]
pub struct Patch<'a> {
    index: usize,
    physical_bounds: &'a BoundingBox,
    fab: &'a BoxInfo,
    bounds: &'a ComponentBounds,
}

impl<'a> Patch<'a> {
    pub fn index(self) -> usize {
        self.index
    }

    pub fn physical_bounds(self) -> &'a BoundingBox {
        self.physical_bounds
    }

    pub fn index_box(self) -> &'a IndexDomain {
        &self.fab.box_info
    }

    pub fn component_min(self, component: usize) -> Option<f64> {
        self.bounds.get_minima(self.index, component)
    }

    pub fn component_max(self, component: usize) -> Option<f64> {
        self.bounds.get_maxima(self.index, component)
    }

    pub fn data_file(self) -> &'a str {
        &self.fab.file_name
    }

    pub fn data_offset(self) -> u64 {
        self.fab.file_offset
    }
}

pub fn read_header(path: impl AsRef<Path>) -> Result<Header> {
    let mut reader = Reader::new(path)?;

    // first line is a format version tag
    let Some(format_and_version) = reader.next_line()? else {
        bail!("missing format and version line");
    };

    match format_and_version.trim() {
        "HyperCLaw-V1.1" => read_hyper(&mut reader),
        _ => bail!("Unknown format {format_and_version}"),
    }
}

fn read_parsed<T>(lines: &mut Reader) -> Result<T>
where
    T::Err: std::error::Error,
    T: std::str::FromStr,
    T::Err: Send + Sync + 'static,
{
    let text = lines.demand_next_line()?;

    Ok(text.parse()?)
}

#[derive(Debug, Clone, Copy)]
enum Delimiter {
    Whitespace,
    Char(char),
}

fn read_fixed_array<T, const N: usize>(delim: Delimiter, lines: &mut Reader) -> Result<[T; N]>
where
    T::Err: std::error::Error,
    T: std::str::FromStr,
    T::Err: Send + Sync + 'static,
{
    let text = lines.demand_next_line()?;

    read_fixed_array_from_str(delim, text)
}

fn read_fixed_array_from_str<T, const N: usize>(delim: Delimiter, line: &str) -> Result<[T; N]>
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

fn read_dyn_array<T>(delim: Delimiter, lines: &mut Reader) -> Result<Vec<T>>
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

/// Read content in this form ((0,0,0) (255,95,95) (0,0,0))
fn read_index_domain(chomper: &mut LineChomper) -> Result<IndexDomain> {
    chomper.skip_whitespace();

    chomper.skip('(');

    let first = read_fixed_array_from_str::<i32, 3>(Delimiter::Char(','), chomper.take_till(')'))?;

    chomper.skip_whitespace();

    chomper.skip('(');

    let second = read_fixed_array_from_str::<i32, 3>(Delimiter::Char(','), chomper.take_till(')'))?;

    chomper.skip_whitespace();

    chomper.skip('(');

    let third = read_fixed_array_from_str::<i32, 3>(Delimiter::Char(','), chomper.take_till(')'))?;

    chomper.skip(')');

    chomper.skip_whitespace();

    Ok(IndexDomain {
        min: first.into(),
        max: second.into(),
        index_type: third.into(),
    })
}

fn read_index_domains(lines: &mut Reader) -> Result<Vec<IndexDomain>> {
    let text = lines.demand_next_line()?;

    let mut chomper = LineChomper::new(text);

    let mut ret = Vec::default();

    while chomper.has_more() {
        ret.push(read_index_domain(&mut chomper)?);
    }

    Ok(ret)
}

fn read_record(lines: &mut Reader) -> Result<PerLevelRecord> {
    let mut iter = lines.demand_next_line()?.split_whitespace();

    let Some(a) = iter.next() else {
        bail!("missing level index")
    };
    let Some(b) = iter.next() else {
        bail!("missing number of grid patches")
    };
    let Some(c) = iter.next() else {
        bail!("missing simulation time")
    };

    let level_index = a.parse()?;
    let number_of_grid_patches = b.parse()?;
    let simulation_time = c.parse()?;

    let level_step = read_parsed::<usize>(lines)?;

    let mut grids = Vec::with_capacity(number_of_grid_patches);

    for _ in 0..number_of_grid_patches {
        let x = DVec2::from(read_fixed_array::<f64, 2>(Delimiter::Whitespace, lines)?);
        let y = DVec2::from(read_fixed_array::<f64, 2>(Delimiter::Whitespace, lines)?);
        let z = DVec2::from(read_fixed_array::<f64, 2>(Delimiter::Whitespace, lines)?);

        grids.push(BoundingBox {
            min: dvec3(x.x, y.x, z.x),
            max: dvec3(x.y, y.y, z.y),
        });
    }

    let path = lines.demand_next_line()?.trim().to_string();

    Ok(PerLevelRecord {
        level_index,
        number_of_grid_patches,
        simulation_time,
        level_step,
        grids,
        path,
    })
}

fn read_hyper(lines: &mut Reader) -> Result<Header> {
    // next is the number of quantities

    let Some(num_quant) = lines.next_line()? else {
        bail!("missing number of quantities");
    };

    let num_quant = num_quant
        .parse::<usize>()
        .context("reading number of quantities")?;

    // guard
    if num_quant > 2048 {
        bail!("too many quantities");
    }

    let mut variables = Vec::with_capacity(num_quant);

    for index in 0..num_quant {
        let name = lines.demand_next_line().context("reading quantity name")?;

        variables.push(Variable {
            name: name.into(),
            index,
        });
    }

    let num_dims = read_parsed::<usize>(lines).context("reading number of dimensions")?;

    if num_dims != 3 {
        bail!("only 3d grids are supported at this time");
    }

    let simulation_time = read_parsed(lines).context("reading simulation time")?;

    let finest_level = read_parsed(lines).context("reading finest amr level")?;

    let domain = BoundingBox {
        min: read_fixed_array(Delimiter::Whitespace, lines)
            .context("reading minimum domain bounds")?
            .into(),
        max: read_fixed_array(Delimiter::Whitespace, lines)
            .context("reading minimum domain bounds")?
            .into(),
    };

    let refinement_ratios = read_dyn_array::<usize>(Delimiter::Whitespace, lines)
        .context("reading refinement ratios")?;

    if refinement_ratios.len() != finest_level {
        bail!("refinement ratios do not match refinement level");
    }

    let index_domains = read_index_domains(lines).context("reading index domains")?;

    if index_domains.len() != finest_level + 1 {
        bail!("mismatch in index domain count");
    }

    let level_steps =
        read_dyn_array(Delimiter::Whitespace, lines).context("reading level steps")?;

    if level_steps.len() != finest_level + 1 {
        bail!("mismatch in level steps count");
    }

    let cell_sizes: Result<Vec<_>> = (0..finest_level + 1)
        .map(|_| read_fixed_array::<f64, 3>(Delimiter::Whitespace, lines).map(DVec3::from))
        .collect();

    let cell_sizes = cell_sizes.context("reading cell sizes per level")?;

    let coordinate_system = match read_parsed::<usize>(lines)? {
        0 => CoordinateSystem::Cartesian,
        1 => CoordinateSystem::Cylindrical,
        2 => CoordinateSystem::Spherical,
        x => bail!("unknown coordinate system {x}"),
    };

    let boundary_width = read_parsed::<usize>(lines).context("reading boundary cell width")?;

    let mut level_records = vec![];
    for level_i in 0..=finest_level {
        let record = read_record(lines).context("reading record line")?;
        if record.level_index != level_i {
            bail!("inconsistent level indicies")
        }
        level_records.push(record);
    }

    Ok(Header {
        simulation_time,
        variables,
        finest_level,
        domain,
        refinement_ratios,
        index_domains,
        level_steps,
        cell_sizes,
        coordinate_system,
        boundary_width,
        level_records,
    })
}

// #[derive(Debug, Clone, Copy)]
// enum VisMfVersion {
//     VersionV1, // encoded as 1
// }

#[derive(Debug, Clone, Copy)]
pub enum V1StorageMode {
    OneFilePerCpu,
    NFiles,
}

#[derive(Debug)]
pub struct CellHeader {
    pub mode: V1StorageMode,
    pub ghost_cell_width: usize,
    pub fabs: Vec<BoxInfo>,
    pub bounds: ComponentBounds,
}

pub fn read_hyper_cell_header(path: impl AsRef<Path>) -> Result<CellHeader> {
    let mut reader = Reader::new(path)?;

    // first line is a format version tag
    let Some(format_and_version) = reader.next_line()? else {
        bail!("missing version line");
    };

    match format_and_version.trim().parse::<u32>()? {
        1 => read_hyper_cell_header_version1(&mut reader),
        _ => bail!("Unknown format {format_and_version}"),
    }
}

fn read_hyper_cell_header_version1(lines: &mut Reader) -> Result<CellHeader> {
    let mode = match read_parsed(lines)? {
        0 => V1StorageMode::OneFilePerCpu,
        1 => V1StorageMode::NFiles,
        _ => {
            bail!("unknown storage mode")
        }
    };

    let Some(num_quant) = lines.next_line()? else {
        bail!("missing number of quantities");
    };

    let num_quant = num_quant
        .parse::<usize>()
        .context("reading number of quantities")?;

    // guard
    if num_quant > 2048 {
        bail!("too many quantities");
    }

    let ghost_cell_width = read_parsed::<usize>(lines)?;

    let (box_count, hash_sig) = {
        let mut box_open_line_iter = lines
            .demand_next_line()?
            .strip_prefix('(')
            .ok_or_else(|| anyhow!("missing opening paren"))?
            .split_whitespace()
            .map(|x| x.parse::<usize>());

        let Some(box_count) = box_open_line_iter.next() else {
            bail!("unable to obtain box count");
        };

        let Some(hash_sig) = box_open_line_iter.next() else {
            bail!("unable to obtain hash signature");
        };

        if box_open_line_iter.next().is_some() {
            bail!("unexpected value in BoxArray header");
        }

        (box_count?, hash_sig?)
    };

    if hash_sig != 0 {
        bail!("unsupported hash signature")
    }

    let boxes: Result<Vec<_>> = (0..box_count)
        .map(|_| {
            lines.demand_next_line().and_then(|x| {
                let mut chomper = LineChomper::new(x);
                let domain = read_index_domain(&mut chomper)?;
                if chomper.has_more() {
                    bail!("unexpected value after Box");
                }
                Ok(domain)
            })
        })
        .collect();

    let boxes = boxes.context("reading boxes")?;

    // read the final )
    if !matches!(lines.demand_next_line()?, ")") {
        bail!("malformed box list");
    }

    let fab_count: usize = read_parsed(lines)?;

    if fab_count != box_count {
        bail!("miscount in fab and box counts");
    }

    let mut fabs = Vec::with_capacity(fab_count);
    let mut name_cache = HashMap::new();
    for b in boxes {
        fabs.push(read_merge_fab_line(&mut name_cache, b, lines)?);
    }

    // skip newlines
    let bounds = ComponentBounds::read(lines, box_count, num_quant)?;

    Ok(CellHeader {
        mode,
        ghost_cell_width,
        fabs,
        bounds,
    })
}

fn get_patch_min_line(lines: &mut Reader) -> Result<(usize, usize), anyhow::Error> {
    let mut patch_min_line = lines.demand_next_line()?;
    while patch_min_line.is_empty() {
        patch_min_line = lines.demand_next_line()?;
    }

    let mut iter = patch_min_line.split(',');

    let Some(count) = iter.next() else {
        bail!("missing min value count");
    };

    let Some(quant) = iter.next() else {
        bail!("missing quantity count");
    };

    if iter.next().is_some() {
        bail!("unexpected value in matrix dimensions");
    }

    Ok((count.parse()?, quant.parse()?))
}

#[derive(Debug)]
pub struct BoxInfo {
    pub box_info: IndexDomain,
    pub file_name: Arc<str>,
    pub file_offset: u64,
}

fn get_cell_hash_place(name: &str) -> Result<usize> {
    let suffix = name
        .strip_prefix("Cell_D_")
        .context("unsupported FAB data filename")?;
    Ok(suffix.parse()?)
}

fn read_merge_fab_line(
    name_cache: &mut HashMap<usize, Arc<str>>,
    domain: IndexDomain,
    lines: &mut Reader,
) -> Result<BoxInfo> {
    let line = lines.demand_next_line()?;

    let Some(line) = line.strip_prefix("FabOnDisk: ").map(|x| x.trim()) else {
        bail!("missing expected fab prefix");
    };

    let mut iter = line.split_whitespace();

    let Some(name) = iter.next() else {
        bail!("missing fab name")
    };

    let Some(offset) = iter.next() else {
        bail!("missing fab offset")
    };

    if iter.next().is_some() {
        bail!("unexpected value in FabOnDisk record");
    }

    let offset = offset.parse::<u64>()?;

    let name = name_cache
        .entry(get_cell_hash_place(name)?)
        .or_insert_with(|| Arc::from(name))
        .clone();

    Ok(BoxInfo {
        box_info: domain,
        file_name: name,
        file_offset: offset,
    })
}

fn read_matrix(
    expected_box_count: usize,
    expected_quant_count: usize,
    lines: &mut Reader,
) -> Result<Vec<f64>> {
    let (bcount, qcount) = get_patch_min_line(lines)?;

    if bcount != expected_box_count || qcount != expected_quant_count {
        bail!("unexpected box or quantity count");
    }

    let value_count = bcount
        .checked_mul(qcount)
        .context("matrix dimensions overflow")?;
    let mut dest = Vec::with_capacity(value_count);

    for _ in 0..bcount {
        let row: Result<Vec<_>, _> = lines
            .demand_next_line()?
            .split_terminator(',')
            .map(|x| x.parse::<f64>())
            .collect();
        let row = row.context("reading component bounds row")?;

        if row.len() != qcount {
            bail!("expected {qcount} matrix values, found {}", row.len());
        }
        dest.extend(row);
    }

    Ok(dest)
}

fn validate_cell_header(header: &Header, level: usize, cell_header: &CellHeader) -> Result<()> {
    let record = header
        .level_records
        .get(level)
        .context("level index is out of range")?;
    let expected_patches = record.number_of_grid_patches;

    if record.grids.len() != expected_patches
        || cell_header.fabs.len() != expected_patches
        || cell_header.bounds.patch_count != expected_patches
    {
        bail!("patch metadata count mismatch at level {level}");
    }

    if cell_header.bounds.component_count != header.variables.len() {
        bail!("component count mismatch at level {level}");
    }

    Ok(())
}

#[derive(Debug)]
pub struct ComponentBounds {
    pub patch_count: usize,
    pub component_count: usize,
    pub minima: Vec<f64>,
    pub maxima: Vec<f64>,
}

impl ComponentBounds {
    fn read(
        lines: &mut Reader,
        expected_box_count: usize,
        expected_quant_count: usize,
    ) -> Result<ComponentBounds> {
        Ok(Self {
            patch_count: expected_box_count,
            component_count: expected_quant_count,
            minima: read_matrix(expected_box_count, expected_quant_count, lines)?,
            maxima: read_matrix(expected_box_count, expected_quant_count, lines)?,
        })
    }

    pub fn get_minima(&self, patch: usize, component_id: usize) -> Option<f64> {
        if patch >= self.patch_count || component_id >= self.component_count {
            return None;
        }
        self.minima
            .get(patch * self.component_count + component_id)
            .copied()
    }

    pub fn get_maxima(&self, patch: usize, component_id: usize) -> Option<f64> {
        if patch >= self.patch_count || component_id >= self.component_count {
            return None;
        }
        self.maxima
            .get(patch * self.component_count + component_id)
            .copied()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn asset(name: &str) -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("assets")
            .join(name)
    }

    #[test]
    fn plotfile_exposes_level_metadata_without_loading_cell_headers() -> Result<()> {
        let plotfile = PlotFile::open(asset("Header").parent().unwrap())?;

        assert_eq!(plotfile.variables().len(), 56);
        assert_eq!(plotfile.variable("density").unwrap().index, 0);
        assert_eq!(plotfile.levels().len(), 4);
        assert_eq!(
            plotfile
                .levels()
                .map(Level::patch_count)
                .collect::<Vec<_>>(),
            [576, 1453, 3985, 19172]
        );

        Ok(())
    }

    #[test]
    fn reads_real_cell_header_metadata() -> Result<()> {
        let header = read_hyper_cell_header(asset("Cell_H"))?;

        assert!(matches!(header.mode, V1StorageMode::NFiles));
        assert_eq!(header.ghost_cell_width, 0);
        assert_eq!(header.fabs.len(), 576);
        assert_eq!(header.bounds.patch_count, 576);
        assert_eq!(header.bounds.component_count, 56);
        assert_eq!(header.bounds.minima.len(), 576 * 56);
        assert_eq!(header.bounds.maxima.len(), 576 * 56);
        assert_eq!(&*header.fabs[0].file_name, "Cell_D_00031");
        assert_eq!(header.fabs[0].file_offset, 0);
        assert_eq!(header.bounds.get_minima(0, 56), None);

        Ok(())
    }

    #[test]
    fn lazily_joins_level_and_cell_metadata() -> Result<()> {
        let root = std::env::temp_dir().join(format!("amrex_rs_test_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(root.join("Level_0"))?;
        std::fs::copy(asset("Header"), root.join("Header"))?;
        std::fs::copy(asset("Cell_H"), root.join("Level_0/Cell_H"))?;

        let result = (|| -> Result<()> {
            let plotfile = PlotFile::open(&root)?;
            let level = plotfile.level(0).context("missing level zero")?;
            let mut patches = level.patches()?;

            assert_eq!(patches.len(), 576);
            let first = patches.next().context("missing first patch")?;
            assert_eq!(first.index(), 0);
            assert_eq!(first.data_file(), "Cell_D_00031");
            assert_eq!(first.data_offset(), 0);
            assert_eq!(first.index_box().min, IVec3::ZERO);
            assert_eq!(first.index_box().max, IVec3::splat(15));
            assert_eq!(first.component_min(56), None);
            Ok(())
        })();

        let _ = std::fs::remove_dir_all(&root);
        result
    }
}
