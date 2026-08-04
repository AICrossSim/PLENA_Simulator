use quantize::{tensor_to_f32_vec, MxDataType, QuantTensor};
use tokio::sync::oneshot::Receiver;
use tokio::sync::Mutex;

use crate::{addr_to_cell, Cell};

/// Behaviour modelling of matrix SRAM.
///
/// The timing aspect is to be considered by the matrix machine itself.
pub struct MatrixSram {
    tile_size: u32,
    tiles: Vec<Mutex<Cell<QuantTensor>>>,
    ty: MxDataType,
}

impl MatrixSram {
    /// Create a matrix SRAM with given tile size and depth.
    pub fn new(tile_size: u32, depth: usize, ty: MxDataType) -> Self {
        let tiles = (0..(depth / tile_size as usize))
            .map(|_| {
                Mutex::new(Cell::Ready(QuantTensor::zeros(
                    (tile_size * tile_size) as usize,
                    ty,
                )))
            })
            .collect();
        Self {
            tile_size,
            tiles,
            ty,
        }
    }

    pub fn tile_size(&self) -> u32 {
        self.tile_size
    }

    pub fn ty(&self) -> MxDataType {
        self.ty
    }

    pub fn size_in_bytes(&self) -> usize {
        let elements_per_tile =
            (self.tile_size * self.tile_size) as usize;
        let bytes_per_tile = match self.ty {
            MxDataType::Plain(element) => (
                elements_per_tile * element.size_in_bits() as usize
            )
                .div_ceil(8),
            MxDataType::Mx { elem, scale, block } => {
                assert!(elements_per_tile.is_multiple_of(block as usize));
                (
                    elements_per_tile * elem.size_in_bits() as usize
                )
                    .div_ceil(8)
                    + (
                        elements_per_tile / block as usize
                            * scale.size_in_bits() as usize
                    )
                        .div_ceil(8)
            }
        };
        bytes_per_tile * self.tiles.len()
    }

    pub async fn read(&self, addr: u32) -> QuantTensor {
        let idx = addr_to_cell(addr, self.tile_size * self.tile_size, self.tiles.len());
        tracing::trace!(
            addr,
            tile_idx = idx,
            tile_size = self.tile_size,
            "MRAM read"
        );
        let mut guard = self.tiles[idx].lock().await;
        let resolved = guard
            .resolve_with(|tensor| {
                assert!(tensor.data_type() == self.ty);
                QuantTensor::quantize_materialized(
                    tensor.as_tensor().shallow_clone(),
                    self.ty,
                )
            })
            .await
            .clone();
        crate::trap_out_of_range(&resolved, "matrix SRAM tile", addr);
        resolved
    }

    pub async fn write(&self, addr: u32, tensor: QuantTensor) {
        let idx = addr_to_cell(addr, self.tile_size * self.tile_size, self.tiles.len());
        assert!(tensor.data_type() == self.ty);
        let stored = QuantTensor::quantize_materialized(
            tensor.as_tensor().shallow_clone(),
            self.ty,
        );
        *self.tiles[idx].lock().await = Cell::Ready(stored);
    }

    pub async fn write_delayed(&self, addr: u32, tensor: Receiver<QuantTensor>) {
        let idx = addr_to_cell(
            addr,
            self.tile_size * self.tile_size,
            self.tiles.len(),
        );
        *self.tiles[idx].lock().await = Cell::Pending(tensor);
    }

    pub async fn continous_write_delayed(
        &self,
        addr: u32,
        write_amount: u32,
        tensor: Receiver<QuantTensor>,
    ) {
        let start_idx = addr_to_cell(addr, self.tile_size * self.tile_size, self.tiles.len());
        // Await the tensor from the channel (blocks until data arrives)
        if let Ok(tensor) = tensor.await {
            let dims = tensor.as_tensor().size();
            let chunk_size = (self.tile_size * self.tile_size) as i64;
            let total = dims[0];

            // Split the tensor into chunks of self.tile_size and store each in self.tiles.
            for i in 0..write_amount.min(
                (total as u32 + self.tile_size * self.tile_size - 1)
                    / (self.tile_size * self.tile_size),
            ) {
                let start = (i as i64) * chunk_size;
                let end = ((i as i64 + 1) * chunk_size).min(total);
                let chunk = tensor
                    .as_tensor()
                    .narrow(0, start, end - start)
                    .shallow_clone();
                let chunk_qt = QuantTensor::quantize_materialized(chunk, self.ty);
                *self.tiles[start_idx + i as usize].lock().await = Cell::Ready(chunk_qt);
            }
        } else {
            // The DMA producer dropped its sender: the prefetch was
            // cancelled/failed, so these cells keep their previous contents.
            tracing::error!(
                addr,
                write_amount,
                "delayed matrix write skipped: DMA sender dropped"
            );
        }
    }

    pub async fn as_bytes(&self) -> Vec<u8> {
        let element_ty = self.ty.element_type();
        let mut result = Vec::new();

        for tile_mutex in &self.tiles {
            let mut guard = tile_mutex.lock().await;
            let tensor = guard
                .resolve_with(|tensor| {
                    assert!(tensor.data_type() == self.ty);
                    QuantTensor::quantize_materialized(
                        tensor.as_tensor().shallow_clone(),
                        self.ty,
                    )
                })
                .await;
            let tensor_data = tensor.as_tensor();
            let f32_vec = tensor_to_f32_vec(tensor_data);
            let len = f32_vec.len();
            // Calculate bytes needed for THIS tile's actual size
            let total_bits = len * element_ty.size_in_bits() as usize;
            let bytes_needed = (total_bits + 7) / 8;
            let mut tile_bytes = vec![0u8; bytes_needed];
            element_ty.bytes_from_f32(&f32_vec, &mut tile_bytes);
            result.extend_from_slice(&tile_bytes);
        }

        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use quantize::{DataType, FpType};
    use tch::Tensor;
    use tokio::sync::oneshot;

    fn f32_plain() -> MxDataType {
        MxDataType::Plain(DataType::Fp(FpType::F32))
    }

    fn bf16_plain() -> MxDataType {
        MxDataType::Plain(DataType::Fp(FpType {
            sign: true,
            exponent: 8,
            mantissa: 7,
        }))
    }

    fn tile(ty: MxDataType, vals: &[f32]) -> QuantTensor {
        QuantTensor::new_assuming_quantized(Tensor::from_slice(vals), ty).unwrap()
    }

    #[test]
    fn test_matrix_new_dimensions() {
        let m = MatrixSram::new(2, 8, f32_plain());
        assert_eq!(m.tile_size(), 2);
        assert_eq!(m.size_in_bytes(), 64);
        let bf16 = MatrixSram::new(2, 8, bf16_plain());
        assert_eq!(bf16.size_in_bytes(), 32);
    }

    #[tokio::test]
    async fn test_matrix_write_read_roundtrip() {
        let ty = f32_plain();
        let m = MatrixSram::new(2, 8, ty); // tile_size 2 -> 4 elements per tile
        let qt = tile(ty, &[1.0, 2.0, 3.0, 4.0]);
        m.write(4, qt.clone()).await; // addr 4 -> cell 1 (4 / tile_size^2)
        let got = m.read(4).await;
        assert!(got.as_tensor().equal(qt.as_tensor()));
    }

    #[tokio::test]
    async fn test_matrix_write_materializes_declared_storage_precision() {
        let ty = bf16_plain();
        let m = MatrixSram::new(2, 8, ty);
        m.write(0, tile(ty, &[1.003, -1.003, 0.0, 2.0]))
            .await;
        let got = tensor_to_f32_vec(m.read(0).await.as_tensor());
        assert_eq!(got, vec![1.0, -1.0, 0.0, 2.0]);
    }

    #[tokio::test]
    async fn test_matrix_write_delayed_uses_tile_addressing() {
        let ty = f32_plain();
        let m = MatrixSram::new(2, 8, ty);
        let qt = tile(ty, &[5.0, 6.0, 7.0, 8.0]);
        let (tx, rx) = oneshot::channel();
        assert!(tx.send(qt.clone()).is_ok());
        m.write_delayed(4, rx).await;
        let got = m.read(4).await;
        assert!(got.as_tensor().equal(qt.as_tensor()));
    }
}
