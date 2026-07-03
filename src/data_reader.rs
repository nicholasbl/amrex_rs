use std::{
    collections::HashMap,
    fs::File,
    mem::{align_of, size_of},
    ops::Range,
    path::{Path, PathBuf},
    sync::{Arc, Mutex},
};

use anyhow::{Context, Result, bail, ensure};
use glam::IVec3;
use memmap2::{Mmap, MmapOptions};

use crate::{Level, Patch, Patches, PlotFile, Variable};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScalarType {
    F32,
    F64,
}

impl ScalarType {
    fn byte_width(self) -> usize {
        match self {
            Self::F32 => size_of::<f32>(),
            Self::F64 => size_of::<f64>(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ByteOrder {
    LittleEndian,
    BigEndian,
}

#[derive(Debug, Clone, Copy)]
struct Encoding {
    scalar: ScalarType,
    byte_order: ByteOrder,
}

/// Reads FAB data from the binary `Cell_D_*` shards of a plotfile.
///
/// Files are mapped read-only and must not be truncated or replaced while a
/// view returned by this reader remains alive.
pub struct DataReader<'a> {
    plot_file: &'a PlotFile,
    mappings: Mutex<HashMap<PathBuf, Arc<Mmap>>>,
}

impl<'a> DataReader<'a> {
    pub(crate) fn new(plot_file: &'a PlotFile) -> Self {
        Self {
            plot_file,
            mappings: Mutex::new(HashMap::new()),
        }
    }

    pub fn component(&self, patch: Patch<'_>, component: usize) -> Result<ComponentView> {
        ensure!(
            std::ptr::eq(self.plot_file, patch.plot_file),
            "patch belongs to a different plotfile"
        );
        ensure!(
            component < self.plot_file.variables().len(),
            "component index {component} is out of range"
        );

        let path = self.data_path(patch)?;
        let mapping = self.mapping(&path)?;
        component_view(mapping, patch, component)
            .with_context(|| format!("reading component {component} from {}", path.display()))
    }

    pub fn variable(&self, patch: Patch<'_>, variable: &Variable) -> Result<ComponentView> {
        self.component(patch, variable.index)
    }

    pub fn level_components<'reader>(
        &'reader self,
        level: Level<'a>,
        component: usize,
    ) -> Result<LevelComponents<'reader, 'a>> {
        ensure!(
            component < self.plot_file.variables().len(),
            "component index {component} is out of range"
        );
        Ok(LevelComponents {
            reader: self,
            patches: level.patches()?,
            component,
        })
    }

    fn data_path(&self, patch: Patch<'_>) -> Result<PathBuf> {
        let record = self
            .plot_file
            .header()
            .level_records
            .get(patch.level_index())
            .context("patch level is out of range")?;
        let directory = Path::new(&record.path).parent().unwrap_or(Path::new(""));
        Ok(self.plot_file.root.join(directory).join(patch.data_file()))
    }

    fn mapping(&self, path: &Path) -> Result<Arc<Mmap>> {
        let mut mappings = self.mappings.lock().expect("mapping cache mutex poisoned");
        if let Some(mapping) = mappings.get(path) {
            return Ok(mapping.clone());
        }

        let file = File::open(path)
            .with_context(|| format!("opening FAB data file {}", path.display()))?;

        // SAFETY: The mapping is read-only and kept alive by Arc for every
        // ComponentView. Callers must not mutate, truncate, or replace an open
        // plotfile's data shards, as documented on DataReader.
        let mapping = unsafe { MmapOptions::new().map(&file) }
            .with_context(|| format!("mapping FAB data file {}", path.display()))?;
        let mapping = Arc::new(mapping);
        mappings.insert(path.to_path_buf(), mapping.clone());
        Ok(mapping)
    }
}

pub struct LevelComponents<'reader, 'plot> {
    reader: &'reader DataReader<'plot>,
    patches: Patches<'plot>,
    component: usize,
}

impl<'plot> Iterator for LevelComponents<'_, 'plot> {
    type Item = Result<(Patch<'plot>, ComponentView)>;

    fn next(&mut self) -> Option<Self::Item> {
        let patch = self.patches.next()?;
        Some(
            self.reader
                .component(patch, self.component)
                .map(|view| (patch, view)),
        )
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.patches.size_hint()
    }
}

impl ExactSizeIterator for LevelComponents<'_, '_> {}

#[derive(Clone)]
pub struct ComponentView {
    mapping: Arc<Mmap>,
    byte_range: Range<usize>,
    shape: [usize; 3],
    origin: IVec3,
    component: usize,
    encoding: Encoding,
}

impl ComponentView {
    pub fn shape(&self) -> [usize; 3] {
        self.shape
    }

    pub fn origin(&self) -> IVec3 {
        self.origin
    }

    pub fn component(&self) -> usize {
        self.component
    }

    pub fn scalar_type(&self) -> ScalarType {
        self.encoding.scalar
    }

    pub fn byte_order(&self) -> ByteOrder {
        self.encoding.byte_order
    }

    pub fn len(&self) -> usize {
        self.shape[0] * self.shape[1] * self.shape[2]
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn values(&self) -> Values<'_> {
        Values {
            bytes: self.bytes(),
            encoding: self.encoding,
            front: 0,
            back: self.len(),
        }
    }

    pub fn get_linear(&self, index: usize) -> Option<f64> {
        (index < self.len()).then(|| decode(self.bytes(), self.encoding, index))
    }

    pub fn get(&self, index: IVec3) -> Option<f64> {
        let local = index - self.origin;
        if local.cmplt(IVec3::ZERO).any()
            || local.x as usize >= self.shape[0]
            || local.y as usize >= self.shape[1]
            || local.z as usize >= self.shape[2]
        {
            return None;
        }

        let linear = local.x as usize
            + self.shape[0] * (local.y as usize + self.shape[1] * local.z as usize);
        self.get_linear(linear)
    }

    /// Return a direct slice only for aligned, native-endian 64-bit data.
    pub fn as_f64_slice(&self) -> Option<&[f64]> {
        if self.encoding.scalar != ScalarType::F64
            || self.encoding.byte_order != native_byte_order()
        {
            return None;
        }

        let bytes = self.bytes();
        if !bytes.len().is_multiple_of(size_of::<f64>())
            || !(bytes.as_ptr() as usize).is_multiple_of(align_of::<f64>())
        {
            return None;
        }

        // SAFETY: Alignment and length are checked above, the mapping outlives
        // the returned slice, and all bit patterns are valid f64 values.
        Some(unsafe {
            std::slice::from_raw_parts(bytes.as_ptr().cast::<f64>(), bytes.len() / size_of::<f64>())
        })
    }

    pub fn as_bytes(&self) -> &[u8] {
        self.bytes()
    }

    fn bytes(&self) -> &[u8] {
        &self.mapping[self.byte_range.clone()]
    }
}

pub struct Values<'a> {
    bytes: &'a [u8],
    encoding: Encoding,
    front: usize,
    back: usize,
}

impl Iterator for Values<'_> {
    type Item = f64;

    fn next(&mut self) -> Option<Self::Item> {
        if self.front == self.back {
            return None;
        }
        let index = self.front;
        self.front += 1;
        Some(decode(self.bytes, self.encoding, index))
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.back - self.front;
        (remaining, Some(remaining))
    }
}

impl DoubleEndedIterator for Values<'_> {
    fn next_back(&mut self) -> Option<Self::Item> {
        if self.front == self.back {
            return None;
        }
        self.back -= 1;
        Some(decode(self.bytes, self.encoding, self.back))
    }
}

impl ExactSizeIterator for Values<'_> {}

fn component_view(mapping: Arc<Mmap>, patch: Patch<'_>, component: usize) -> Result<ComponentView> {
    let fab_offset = usize::try_from(patch.data_offset()).context("FAB offset is too large")?;
    ensure!(fab_offset < mapping.len(), "FAB offset is past end of file");

    let tail = &mapping[fab_offset..];
    let newline = tail
        .iter()
        .take(4096)
        .position(|byte| *byte == b'\n')
        .context("FAB header line is missing or too long")?;
    let header = std::str::from_utf8(&tail[..newline]).context("FAB header is not UTF-8")?;
    let parsed = parse_fab_header(header)?;

    ensure!(
        parsed.component_count == patch.plot_file.variables().len(),
        "FAB component count does not match plotfile"
    );

    let ghost = IVec3::splat(i32::try_from(patch.ghost_cell_width())?);
    ensure!(
        parsed.lower == patch.index_box().min - ghost
            && parsed.upper == patch.index_box().max + ghost
            && parsed.index_type == patch.index_box().index_type,
        "FAB box does not match Cell_H metadata"
    );

    let shape = shape(parsed.lower, parsed.upper)?;
    let cell_count = shape
        .into_iter()
        .try_fold(1usize, |total, width| total.checked_mul(width))
        .context("FAB cell count overflow")?;
    let bytes_per_component = cell_count
        .checked_mul(parsed.encoding.scalar.byte_width())
        .context("FAB component byte count overflow")?;
    let component_offset = component
        .checked_mul(bytes_per_component)
        .context("FAB component offset overflow")?;
    let data_start = fab_offset
        .checked_add(newline + 1)
        .context("FAB data offset overflow")?;
    let start = data_start
        .checked_add(component_offset)
        .context("FAB data offset overflow")?;
    let end = start
        .checked_add(bytes_per_component)
        .context("FAB data range overflow")?;
    ensure!(
        end <= mapping.len(),
        "FAB component extends past end of file"
    );

    Ok(ComponentView {
        mapping,
        byte_range: start..end,
        shape,
        origin: parsed.lower,
        component,
        encoding: parsed.encoding,
    })
}

struct FabHeader {
    encoding: Encoding,
    lower: IVec3,
    upper: IVec3,
    index_type: IVec3,
    component_count: usize,
}

fn parse_fab_header(header: &str) -> Result<FabHeader> {
    let rest = header
        .strip_prefix("FAB ")
        .context("unsupported FAB header prefix")?;
    let descriptor_len = balanced_prefix_len(rest)?;
    let descriptor = &rest[..descriptor_len];
    let box_and_components = &rest[descriptor_len..];

    let descriptor_values = integers(descriptor)?;
    ensure!(descriptor_values.len() >= 10, "malformed RealDescriptor");
    let byte_width = usize::try_from(descriptor_values[0])?;
    let format = &descriptor_values[1..9];
    let order_len = usize::try_from(descriptor_values[9])?;
    ensure!(
        descriptor_values.len() == 10 + order_len,
        "malformed RealDescriptor byte order"
    );
    ensure!(byte_width == order_len, "inconsistent RealDescriptor width");

    let scalar = match (byte_width, format) {
        (8, [64, 11, 52, 0, 1, 12, 0, 1023]) => ScalarType::F64,
        (4, [32, 8, 23, 0, 1, 9, 0, 127]) => ScalarType::F32,
        _ => bail!("unsupported floating-point descriptor"),
    };
    let order = &descriptor_values[10..];
    let byte_order = if order.first() == Some(&(byte_width as i64)) {
        ByteOrder::LittleEndian
    } else if order.last() == Some(&(byte_width as i64)) {
        ByteOrder::BigEndian
    } else {
        bail!("unsupported RealDescriptor byte order");
    };

    let values = integers(box_and_components)?;
    ensure!(values.len() == 10, "malformed FAB box or component count");
    let to_i32 = |value: i64| i32::try_from(value).context("FAB index exceeds i32 range");

    Ok(FabHeader {
        encoding: Encoding { scalar, byte_order },
        lower: IVec3::new(to_i32(values[0])?, to_i32(values[1])?, to_i32(values[2])?),
        upper: IVec3::new(to_i32(values[3])?, to_i32(values[4])?, to_i32(values[5])?),
        index_type: IVec3::new(to_i32(values[6])?, to_i32(values[7])?, to_i32(values[8])?),
        component_count: usize::try_from(values[9])?,
    })
}

fn balanced_prefix_len(text: &str) -> Result<usize> {
    ensure!(text.starts_with('('), "missing RealDescriptor");
    let mut depth = 0usize;
    for (index, byte) in text.bytes().enumerate() {
        match byte {
            b'(' => depth += 1,
            b')' => {
                depth = depth.checked_sub(1).context("unbalanced RealDescriptor")?;
                if depth == 0 {
                    return Ok(index + 1);
                }
            }
            _ => {}
        }
    }
    bail!("unterminated RealDescriptor")
}

fn integers(text: &str) -> Result<Vec<i64>> {
    let bytes = text.as_bytes();
    let mut result = Vec::new();
    let mut index = 0;
    while index < bytes.len() {
        let start = index;
        if bytes[index] == b'-' {
            index += 1;
        }
        let digit_start = index;
        while index < bytes.len() && bytes[index].is_ascii_digit() {
            index += 1;
        }
        if digit_start != index {
            result.push(text[start..index].parse()?);
        } else {
            index = start + 1;
        }
    }
    Ok(result)
}

fn shape(lower: IVec3, upper: IVec3) -> Result<[usize; 3]> {
    let extent = upper.as_i64vec3() - lower.as_i64vec3() + 1;
    ensure!(extent.cmpgt(glam::I64Vec3::ZERO).all(), "empty FAB box");
    Ok([
        usize::try_from(extent.x)?,
        usize::try_from(extent.y)?,
        usize::try_from(extent.z)?,
    ])
}

fn decode(bytes: &[u8], encoding: Encoding, index: usize) -> f64 {
    let width = encoding.scalar.byte_width();
    let start = index * width;
    match encoding.scalar {
        ScalarType::F32 => {
            let value: [u8; 4] = bytes[start..start + 4].try_into().unwrap();
            (match encoding.byte_order {
                ByteOrder::LittleEndian => f32::from_le_bytes(value),
                ByteOrder::BigEndian => f32::from_be_bytes(value),
            }) as f64
        }
        ScalarType::F64 => {
            let value: [u8; 8] = bytes[start..start + 8].try_into().unwrap();
            match encoding.byte_order {
                ByteOrder::LittleEndian => f64::from_le_bytes(value),
                ByteOrder::BigEndian => f64::from_be_bytes(value),
            }
        }
    }
}

fn native_byte_order() -> ByteOrder {
    #[cfg(target_endian = "little")]
    {
        ByteOrder::LittleEndian
    }
    #[cfg(target_endian = "big")]
    {
        ByteOrder::BigEndian
    }
}

#[cfg(test)]
mod tests {
    use std::{io::Write, path::PathBuf};

    use super::*;

    fn asset(name: &str) -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("assets")
            .join(name)
    }

    #[test]
    fn parses_self_describing_double_header() -> Result<()> {
        let header = "FAB ((8, (64 11 52 0 1 12 0 1023)),(8, (8 7 6 5 4 3 2 1)))((0,0,0) (15,15,15) (0,0,0)) 56";
        let parsed = parse_fab_header(header)?;
        assert_eq!(parsed.encoding.scalar, ScalarType::F64);
        assert_eq!(parsed.encoding.byte_order, ByteOrder::LittleEndian);
        assert_eq!(parsed.lower, IVec3::ZERO);
        assert_eq!(parsed.upper, IVec3::splat(15));
        assert_eq!(parsed.component_count, 56);
        Ok(())
    }

    #[test]
    fn value_iterator_decodes_without_allocating() {
        let values = [1.25_f64, -2.5, 10.0];
        let bytes = values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect::<Vec<_>>();
        let decoded = Values {
            bytes: &bytes,
            encoding: Encoding {
                scalar: ScalarType::F64,
                byte_order: ByteOrder::LittleEndian,
            },
            front: 0,
            back: values.len(),
        }
        .collect::<Vec<_>>();
        assert_eq!(decoded, values);
    }

    #[test]
    fn maps_one_component_from_a_fab_shard() -> Result<()> {
        let root = std::env::temp_dir().join(format!("amrex_rs_data_test_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(root.join("Level_0"))?;
        std::fs::copy(asset("Header"), root.join("Header"))?;
        std::fs::copy(asset("Cell_H"), root.join("Level_0/Cell_H"))?;

        let result = (|| -> Result<()> {
            let mut shard = File::create(root.join("Level_0/Cell_D_00031"))?;
            let header = "FAB ((8, (64 11 52 0 1 12 0 1023)),(8, (8 7 6 5 4 3 2 1)))((0,0,0) (15,15,15) (0,0,0)) 56\n";
            assert_eq!(header.len(), 90);
            shard.write_all(header.as_bytes())?;

            const CELL_COUNT: usize = 16 * 16 * 16;
            for component in 0..56 {
                for index in 0..CELL_COUNT {
                    let value = component as f64 * 10_000.0 + index as f64;
                    shard.write_all(&value.to_le_bytes())?;
                }
            }
            drop(shard);

            let plotfile = PlotFile::open(&root)?;
            let patch = plotfile
                .level(0)
                .context("missing level zero")?
                .patches()?
                .next()
                .context("missing patch zero")?;
            let reader = plotfile.data_reader();
            let view = reader.component(patch, 3)?;

            assert_eq!(view.shape(), [16, 16, 16]);
            assert_eq!(view.origin(), IVec3::ZERO);
            assert_eq!(view.len(), CELL_COUNT);
            assert_eq!(view.values().len(), CELL_COUNT);
            assert_eq!(view.get_linear(0), Some(30_000.0));
            assert_eq!(view.get(IVec3::new(15, 15, 15)), Some(34_095.0));
            assert_eq!(view.values().next_back(), Some(34_095.0));
            // The 90-byte ASCII header deliberately leaves f64 data unaligned.
            assert!(view.as_f64_slice().is_none());

            Ok(())
        })();

        let _ = std::fs::remove_dir_all(&root);
        result
    }
}
