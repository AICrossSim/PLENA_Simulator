use quantize::{MxDataType, QuantTensor};
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
        (self.tile_size * self.tile_size) as usize * self.tiles.len()
    }

    pub async fn read(&self, addr: u32) -> QuantTensor {
        let idx = addr_to_cell(addr, self.tile_size * self.tile_size);

        let mut guard = self.tiles[idx].lock().await;
        if let Cell::Pending(ref mut fut) = *guard {
            *guard = Cell::Ready(fut.await.unwrap());
        }
        let Cell::Ready(ref t) = *guard else {
            unreachable!()
        };
        t.clone()
    }

    pub async fn write(&self, addr: u32, tensor: QuantTensor) {
        let idx = addr_to_cell(addr, self.tile_size * self.tile_size);

        assert!(tensor.data_type() == self.ty);
        *self.tiles[idx].lock().await = Cell::Ready(tensor);
    }

    pub async fn write_delayed(&self, addr: u32, tensor: Receiver<QuantTensor>) {
        // NOTE: preserved from original — uses `tile_size` not `tile_size * tile_size`.
        let idx = addr_to_cell(addr, self.tile_size);

        *self.tiles[idx].lock().await = Cell::Pending(tensor);
    }

    pub async fn continous_write_delayed(
        &self,
        addr: u32,
        write_amount: u32,
        tensor: Receiver<QuantTensor>,
    ) {
        let start_idx = addr_to_cell(addr, self.tile_size * self.tile_size);
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
                let chunk_qt = QuantTensor::quantize(chunk, self.ty);
                *self.tiles[start_idx + i as usize].lock().await = Cell::Ready(chunk_qt);
            }
        }
    }

    pub async fn as_bytes(&self) -> Vec<u8> {
        let element_ty = self.ty.element_type();
        let mut result = Vec::new();

        for tile_mutex in &self.tiles {
            let mut guard = tile_mutex.lock().await;
            if let Cell::Pending(ref mut fut) = *guard {
                *guard = Cell::Ready(fut.await.unwrap());
            }
            let Cell::Ready(ref tensor) = *guard else {
                unreachable!()
            };
            let tensor_data = tensor.as_tensor();
            let len = tensor_data.size1().unwrap() as usize;
            let f32_slice =
                unsafe { core::slice::from_raw_parts(tensor_data.data_ptr() as *const f32, len) };
            // Calculate bytes needed for THIS tile's actual size
            let total_bits = len * element_ty.size_in_bits() as usize;
            let bytes_needed = (total_bits + 7) / 8;
            let mut tile_bytes = vec![0u8; bytes_needed];
            element_ty.bytes_from_f32(f32_slice, &mut tile_bytes);
            result.extend_from_slice(&tile_bytes);
        }

        result
    }
}
