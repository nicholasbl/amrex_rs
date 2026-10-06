use anyhow::{Context, Result, ensure};
use rkyv::{Archive, Deserialize, Serialize};

use super::CompactScalarEncoding;
use crate::utility::{SparseGrid3, SparseGridChunkView};
use crate::{BoundingBox, CoordinateSystem, Header, IndexDomain, Variable};

pub(super) const FORMAT_MAGIC: u32 = u32::from_le_bytes(*b"AMRC");
pub(super) const FORMAT_VERSION: u32 = 3;
pub(super) const PREFIX_LENGTH: u64 = 8;
pub(super) const FOOTER_LENGTH: u64 = 8;

#[derive(Archive, Deserialize, Serialize)]
pub(super) struct CompactArchiveHeader {
    pub(super) chunk_bits: u32,
    pub(super) header: ArchiveHeader,
    pub(super) components: Vec<ArchiveStoredComponent>,
    pub(super) levels: Vec<ArchiveLevel>,
}

#[derive(Archive, Deserialize, Serialize)]
pub(super) struct ArchiveStoredComponent {
    pub(super) component_id: u32,
    pub(super) encoding: u8,
    pub(super) min: f64,
    pub(super) max: f64,
}

#[derive(Archive, Deserialize, Serialize)]
pub(super) struct ArchiveHeader {
    pub(super) simulation_time: f64,
    pub(super) finest_level: u64,
    pub(super) domain: ArchiveBoundingBox,
    pub(super) variables: Vec<ArchiveVariable>,
    pub(super) refinement_ratios: Vec<u64>,
    pub(super) index_domains: Vec<ArchiveIndexDomain>,
    pub(super) level_steps: Vec<u64>,
    pub(super) cell_sizes: Vec<[f64; 3]>,
    pub(super) coordinate_system: u8,
    pub(super) boundary_width: u64,
}

#[derive(Archive, Deserialize, Serialize, Clone)]
pub(super) struct ArchiveVariable {
    pub(super) name: String,
    pub(super) index: u64,
}

#[derive(Archive, Deserialize, Serialize)]
pub(super) struct ArchiveBoundingBox {
    pub(super) min: [f64; 3],
    pub(super) max: [f64; 3],
}

#[derive(Archive, Deserialize, Serialize)]
pub(super) struct ArchiveIndexDomain {
    pub(super) min: [i32; 3],
    pub(super) max: [i32; 3],
    pub(super) index_type: [i32; 3],
}

#[derive(Archive, Deserialize, Serialize)]
pub(super) struct ArchiveLevel {
    pub(super) level_index: u64,
    pub(super) index_origin: [i32; 3],
    pub(super) physical_origin: [f64; 3],
    pub(super) cell_size: [f64; 3],
    pub(super) components: Vec<ArchiveComponentBlock>,
    pub(super) active_cubes: ArchiveBlock,
}

#[derive(Archive, Deserialize, Serialize)]
pub(super) struct ArchiveComponentBlock {
    pub(super) data: ArchiveBlock,
    pub(super) ranges: Vec<ArchiveChunkRange>,
}

#[derive(Archive, Deserialize, Serialize, Clone, Copy)]
pub(super) struct ArchiveBlock {
    pub(super) offset: u64,
    pub(super) stored_length: u64,
    pub(super) decoded_length: u64,
    pub(super) compression: ArchiveCompression,
}

#[derive(Archive, Deserialize, Serialize, Clone, Copy)]
#[repr(u8)]
pub(super) enum ArchiveCompression {
    None,
}

#[derive(Archive, Deserialize, Serialize)]
pub(super) struct ArchiveChunkRange {
    pub(super) min: [u32; 3],
    pub(super) max: [u32; 3],
    pub(super) value_min: f64,
    pub(super) value_max: f64,
}

#[derive(Archive, Deserialize, Serialize)]
pub(super) struct ArchiveScalarGrid<T> {
    bounds: [u32; 3],
    chunks: Vec<ArchiveScalarChunk<T>>,
}

#[derive(Archive, Deserialize, Serialize)]
enum ArchiveScalarChunk<T> {
    Uniform {
        key: [u32; 3],
        value: T,
        mask: Box<[u64; crate::utility::MASK_WORDS]>,
    },
    Dense {
        key: [u32; 3],
        values: Vec<T>,
        mask: Box<[u64; crate::utility::MASK_WORDS]>,
    },
}

#[derive(Archive, Deserialize, Serialize)]
pub(super) struct ArchiveMaskGrid {
    bounds: [u32; 3],
    chunks: Vec<ArchiveMaskChunk>,
}

#[derive(Archive, Deserialize, Serialize)]
struct ArchiveMaskChunk {
    key: [u32; 3],
    mask: Box<[u64; crate::utility::MASK_WORDS]>,
}

pub(super) fn encoding_fields(encoding: CompactScalarEncoding) -> (u8, f64, f64) {
    match encoding {
        CompactScalarEncoding::F32 => (0, 0.0, 0.0),
        CompactScalarEncoding::F64 => (1, 0.0, 0.0),
        CompactScalarEncoding::UNorm32 { min, max } => (2, min, max),
    }
}

pub(super) fn archived_encoding(tag: u8, min: f64, max: f64) -> Result<CompactScalarEncoding> {
    let encoding = match tag {
        0 => CompactScalarEncoding::F32,
        1 => CompactScalarEncoding::F64,
        2 => CompactScalarEncoding::UNorm32 { min, max },
        _ => anyhow::bail!("unsupported compact scalar encoding {tag}"),
    };
    encoding.validate()?;
    Ok(encoding)
}

impl ArchiveHeader {
    pub(super) fn from_header(header: &Header) -> Self {
        Self {
            simulation_time: header.simulation_time,
            finest_level: header.finest_level as u64,
            domain: ArchiveBoundingBox::from_bounding_box(&header.domain),
            variables: header
                .variables
                .iter()
                .map(ArchiveVariable::from_variable)
                .collect(),
            refinement_ratios: header.refinement_ratios.iter().map(|&x| x as u64).collect(),
            index_domains: header
                .index_domains
                .iter()
                .map(ArchiveIndexDomain::from_index_domain)
                .collect(),
            level_steps: header.level_steps.iter().map(|&x| x as u64).collect(),
            cell_sizes: header
                .cell_sizes
                .iter()
                .map(|&v| dvec3_to_array(v))
                .collect(),
            coordinate_system: match header.coordinate_system {
                CoordinateSystem::Cartesian => 0,
                CoordinateSystem::Cylindrical => 1,
                CoordinateSystem::Spherical => 2,
            },
            boundary_width: header.boundary_width as u64,
        }
    }
}

impl ArchiveVariable {
    fn from_variable(variable: &Variable) -> Self {
        Self {
            name: variable.name.clone(),
            index: variable.index as u64,
        }
    }
}

impl TryFrom<ArchiveVariable> for Variable {
    type Error = anyhow::Error;

    fn try_from(variable: ArchiveVariable) -> Result<Self> {
        Ok(Self {
            name: variable.name,
            index: usize::try_from(variable.index)
                .context("variable index does not fit in usize")?,
        })
    }
}

impl ArchiveBoundingBox {
    fn from_bounding_box(bounds: &BoundingBox) -> Self {
        Self {
            min: dvec3_to_array(bounds.min),
            max: dvec3_to_array(bounds.max),
        }
    }
}

impl ArchiveIndexDomain {
    fn from_index_domain(domain: &IndexDomain) -> Self {
        Self {
            min: ivec3_to_array(domain.min),
            max: ivec3_to_array(domain.max),
            index_type: ivec3_to_array(domain.index_type),
        }
    }
}

pub(super) fn archive_scalar_grid<T>(grid: &SparseGrid3<T>) -> ArchiveScalarGrid<T>
where
    T: Copy + PartialEq,
{
    let mut chunks = Vec::with_capacity(grid.chunk_count());
    grid.for_each_chunk(|chunk| match chunk {
        SparseGridChunkView::Uniform { key, value, mask } => {
            chunks.push(ArchiveScalarChunk::Uniform {
                key: uvec3_to_array(key),
                value,
                mask: Box::new(mask),
            });
        }
        SparseGridChunkView::Dense { key, values, mask } => {
            chunks.push(ArchiveScalarChunk::Dense {
                key: uvec3_to_array(key),
                values: values.to_vec(),
                mask: Box::new(mask),
            });
        }
    });
    chunks.sort_by_key(ArchiveScalarChunk::key);
    ArchiveScalarGrid {
        bounds: uvec3_to_array(grid.bounds()),
        chunks,
    }
}

impl<T> ArchiveScalarGrid<T>
where
    T: Copy + PartialEq,
{
    pub(super) fn into_grid(self) -> Result<SparseGrid3<T>> {
        let mut grid =
            SparseGrid3::with_chunk_capacity(array_to_uvec3(self.bounds), self.chunks.len());
        for chunk in self.chunks {
            match chunk {
                ArchiveScalarChunk::Uniform { key, value, mask } => {
                    grid.insert_uniform_chunk(array_to_uvec3(key), value, *mask);
                }
                ArchiveScalarChunk::Dense { key, values, mask } => {
                    ensure!(
                        values.len() == crate::utility::CHUNK_VOLUME,
                        "dense chunk has {} values; expected {}",
                        values.len(),
                        crate::utility::CHUNK_VOLUME
                    );
                    grid.insert_dense_chunk(array_to_uvec3(key), values.into_boxed_slice(), *mask);
                }
            }
        }
        Ok(grid)
    }
}

impl<T> ArchiveScalarChunk<T> {
    fn key(&self) -> [u32; 3] {
        match self {
            Self::Uniform { key, .. } | Self::Dense { key, .. } => *key,
        }
    }
}

impl ArchiveMaskGrid {
    pub(super) fn from_grid(grid: &SparseGrid3<()>) -> Self {
        let mut chunks = Vec::with_capacity(grid.chunk_count());
        grid.for_each_chunk(|chunk| match chunk {
            SparseGridChunkView::Uniform { key, mask, .. }
            | SparseGridChunkView::Dense { key, mask, .. } => {
                chunks.push(ArchiveMaskChunk {
                    key: uvec3_to_array(key),
                    mask: Box::new(mask),
                });
            }
        });
        chunks.sort_by_key(|chunk| chunk.key);
        Self {
            bounds: uvec3_to_array(grid.bounds()),
            chunks,
        }
    }

    pub(super) fn into_grid(self) -> SparseGrid3<()> {
        let mut grid =
            SparseGrid3::with_chunk_capacity(array_to_uvec3(self.bounds), self.chunks.len());
        for chunk in self.chunks {
            grid.insert_uniform_chunk(array_to_uvec3(chunk.key), (), *chunk.mask);
        }
        grid
    }
}

pub(super) fn uvec3_to_array(v: glam::UVec3) -> [u32; 3] {
    [v.x, v.y, v.z]
}

pub(super) fn array_to_uvec3(v: [u32; 3]) -> glam::UVec3 {
    glam::UVec3::new(v[0], v[1], v[2])
}

pub(super) fn ivec3_to_array(v: glam::IVec3) -> [i32; 3] {
    [v.x, v.y, v.z]
}

pub(super) fn array_to_ivec3(v: [i32; 3]) -> glam::IVec3 {
    glam::IVec3::new(v[0], v[1], v[2])
}

pub(super) fn dvec3_to_array(v: glam::DVec3) -> [f64; 3] {
    [v.x, v.y, v.z]
}

pub(super) fn array_to_dvec3(v: [f64; 3]) -> glam::DVec3 {
    glam::DVec3::new(v[0], v[1], v[2])
}
