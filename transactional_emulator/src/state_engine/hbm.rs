use std::sync::Arc;

use memory::{ErasedMemoryModel, MemoryModel};

pub async fn read_bytes(hbm: &Arc<dyn ErasedMemoryModel>, addr: u64, len: usize) -> Vec<u8> {
    let mut out = vec![0u8; len];
    let mut copied = 0usize;
    while copied < len {
        let current = addr + copied as u64;
        let aligned = current / 64 * 64;
        let within = (current % 64) as usize;
        let count = (64 - within).min(len - copied);
        let block = hbm.read(aligned).await;
        out[copied..copied + count].copy_from_slice(&block[within..within + count]);
        copied += count;
    }
    out
}

pub async fn write_bytes(hbm: &Arc<dyn ErasedMemoryModel>, addr: u64, data: &[u8]) {
    let mut written = 0usize;
    while written < data.len() {
        let current = addr + written as u64;
        let aligned = current / 64 * 64;
        let within = (current % 64) as usize;
        let count = (64 - within).min(data.len() - written);
        let mut block = if within == 0 && count == 64 {
            [0u8; 64]
        } else {
            hbm.read(aligned).await
        };
        block[within..within + count].copy_from_slice(&data[written..written + count]);
        hbm.write(aligned, block).await;
        written += count;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use memory::MemoryBacked;

    #[tokio::test]
    async fn unaligned_round_trip_preserves_neighbors() {
        let backing = Arc::new(MemoryBacked::with_capacity(192));
        backing.with_data(|data| data.fill(0xA5));
        let hbm: Arc<dyn ErasedMemoryModel> = backing.clone();
        write_bytes(&hbm, 60, &[1, 2, 3, 4, 5, 6, 7, 8]).await;
        assert_eq!(read_bytes(&hbm, 60, 8).await, [1, 2, 3, 4, 5, 6, 7, 8]);
        backing.with_data(|data| {
            assert_eq!(data[59], 0xA5);
            assert_eq!(data[68], 0xA5);
        });
    }
}
