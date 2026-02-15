use quantize::{DataType, MxDataType, QuantTensor};
use tch::Tensor;
use tokio::sync::oneshot::Receiver;
use tokio::sync::Mutex;

/// Vector SRAM that stores data in pure binary format with row-based storage.
///
/// The SRAM supports two data types:
/// - FP (Floating Point): Stored as binary representation of the FP type
/// - INT (Integer): Stored as binary representation of integers
///
/// The SRAM is organized as rows, where each row has a width of `vlen * element_size_in_bytes`.
/// During read/write operations, data is clipped to VLEN-sized vectors.
pub struct VectorSram {
    /// Vector length (VLEN) - determines the size of each vector operation
    vlen: u32,
    /// Number of rows in the SRAM
    depth: usize,
    /// Data type for FP storage (used when writing QuantTensor)
    fp_type: DataType,
    /// Size of integer in bytes (used when writing integer vectors)
    int_size_bytes: usize,
    /// Raw binary storage: each row is stored as bytes
    /// Row width = vlen * element_size_in_bytes
    rows: Vec<Mutex<RowData>>,
}

/// Represents a row of data, either ready or pending from a delayed write
enum RowData {
    Ready(Vec<u8>),
    Pending(Receiver<QuantTensor>),
}

impl VectorSram {
    /// Create a new Vector SRAM with given vector length, depth, and data types.
    ///
    /// # Arguments
    /// * `vlen` - Vector length (VLEN)
    /// * `depth` - Number of rows in the SRAM
    /// * `fp_type` - Floating point data type for FP operations
    /// * `int_size_bytes` - Size of integer in bytes (typically 4 for i32)
    pub fn new(vlen: u32, depth: usize, fp_type: DataType, int_size_bytes: usize) -> Self {
        // Use FP type size for row width (can be changed if needed)
        let element_size = fp_type.size_in_bits() as usize / 8;
        let row_width = vlen as usize * element_size;

        let rows = (0..depth)
            .map(|_| Mutex::new(RowData::Ready(vec![0u8; row_width])))
            .collect();

        Self {
            vlen,
            depth,
            fp_type,
            int_size_bytes,
            rows,
        }
    }

    /// Create a new Vector SRAM from MxDataType (for backward compatibility).
    ///
    /// Extracts the Plain DataType from MxDataType and uses it for FP storage.
    pub fn from_mx_type(vlen: u32, depth: usize, mx_type: MxDataType) -> Self {
        let fp_type = match mx_type {
            MxDataType::Plain(dt) => dt,
            MxDataType::Mx { elem, .. } => elem,
        };
        Self::new(vlen, depth, fp_type, 4) // Default to 4 bytes for int (i32)
    }

    /// Get the vector length (VLEN)
    pub fn tile_size(&self) -> u32 {
        self.vlen
    }

    /// Get the data type (for backward compatibility)
    pub fn ty(&self) -> MxDataType {
        MxDataType::Plain(self.fp_type)
    }

    /// Get the size of the SRAM in bytes
    pub fn size_in_bytes(&self) -> usize {
        let element_size = self.fp_type.size_in_bits() as usize / 8;
        let row_width = self.vlen as usize * element_size;
        row_width * self.depth
    }

    /// Read a vector from the SRAM at the given address as FP (QuantTensor).
    ///
    /// The address must be a multiple of vlen (in element units).
    /// Data is read from binary storage and converted to QuantTensor.
    pub async fn read(&self, addr: u32) -> QuantTensor {
        let row_idx = self.addr_to_row_idx(addr);
        assert!(row_idx < self.depth, "Address out of bounds");

        let mut guard = self.rows[row_idx].lock().await;

        // Handle pending writes
        if let RowData::Pending(ref mut receiver) = *guard {
            let tensor = receiver.await.unwrap();
            let row_bytes = self.quant_tensor_to_bytes(&tensor);
            *guard = RowData::Ready(row_bytes);
        }

        // Read the row data
        let row_bytes = match &*guard {
            RowData::Ready(bytes) => bytes.clone(),
            RowData::Pending(_) => unreachable!(),
        };

        // Convert from binary to QuantTensor
        self.bytes_to_quant_tensor(&row_bytes, self.vlen)
    }

    /// Read a vector from the SRAM at the given address as integers.
    ///
    /// The address must be a multiple of vlen (in element units).
    /// Returns a vector of i32 values.
    pub async fn read_int(&self, addr: u32) -> Vec<i32> {
        let row_idx = self.addr_to_row_idx(addr);
        assert!(row_idx < self.depth, "Address out of bounds");

        let mut guard = self.rows[row_idx].lock().await;

        // Handle pending writes (convert to bytes first)
        if let RowData::Pending(ref mut receiver) = *guard {
            let tensor = receiver.await.unwrap();
            let row_bytes = self.quant_tensor_to_bytes(&tensor);
            *guard = RowData::Ready(row_bytes);
        }

        // Read the row data
        let row_bytes = match &*guard {
            RowData::Ready(bytes) => bytes.clone(),
            RowData::Pending(_) => unreachable!(),
        };

        // Convert from binary to integers
        self.bytes_to_int_vec(&row_bytes, self.vlen)
    }

    /// Write a vector to the SRAM at the given address as FP (QuantTensor).
    ///
    /// The address must be a multiple of vlen (in element units).
    /// Data is converted from QuantTensor to binary storage.
    pub async fn write(&self, addr: u32, tensor: QuantTensor) {
        let row_idx = self.addr_to_row_idx(addr);
        assert!(row_idx < self.depth, "Address out of bounds");

        // Clip to VLEN
        let clipped = self.clip_to_vlen(&tensor);

        // Convert to bytes
        let row_bytes = self.quant_tensor_to_bytes(&clipped);

        *self.rows[row_idx].lock().await = RowData::Ready(row_bytes);
    }

    /// Write a vector to the SRAM at the given address as integers.
    ///
    /// The address must be a multiple of vlen (in element units).
    /// Data is converted from integers to binary storage.
    pub async fn write_int(&self, addr: u32, int_vec: &[i32]) {
        let row_idx = self.addr_to_row_idx(addr);
        assert!(row_idx < self.depth, "Address out of bounds");
        assert!(int_vec.len() <= self.vlen as usize, "Vector too long");

        // Convert integers to bytes
        let row_bytes = self.int_vec_to_bytes(int_vec, self.vlen);

        *self.rows[row_idx].lock().await = RowData::Ready(row_bytes);
    }

    /// Write a vector with delayed delivery (from a channel).
    pub async fn write_delayed(&self, addr: u32, tensor: Receiver<QuantTensor>) {
        let row_idx = self.addr_to_row_idx(addr);
        assert!(row_idx < self.depth, "Address out of bounds");

        *self.rows[row_idx].lock().await = RowData::Pending(tensor);
    }

    /// Continuous write delayed - writes multiple rows from a single tensor.
    pub async fn continous_write_delayed(
        &self,
        addr: u32,
        write_amount: u32,
        tensor: Receiver<QuantTensor>,
    ) {
        let start_row_idx = self.addr_to_row_idx(addr);

        // Await the tensor from the channel and extract data immediately to make it Send
        let tensor = tensor.await.unwrap();
        let tensor_data = tensor.as_tensor();
        let total_elements = tensor_data.size1().unwrap() as usize;

        // Extract f32 data from tensor to make it Send-safe
        let len = total_elements;
        let f32_slice =
            unsafe { core::slice::from_raw_parts(tensor_data.data_ptr() as *const f32, len) };
        let data_vec: Vec<f32> = f32_slice.to_vec();

        let chunk_size = self.vlen as usize;
        let num_chunks = write_amount.min(((total_elements + chunk_size - 1) / chunk_size) as u32);

        for i in 0..num_chunks {
            let row_idx = start_row_idx + i as usize;
            if row_idx >= self.depth {
                break;
            }

            let start = (i as usize) * chunk_size;
            let end = (start + chunk_size).min(total_elements);
            let chunk_data = &data_vec[start..end];

            // Pad to VLEN if needed
            let mut padded_data = vec![0.0f32; chunk_size];
            let chunk_len = end - start;
            padded_data[..chunk_len].copy_from_slice(chunk_data);

            // Create tensor from padded data and convert to bytes
            let padded_tensor = Tensor::from_slice(&padded_data);
            let chunk_qt = QuantTensor::quantize(padded_tensor, MxDataType::Plain(self.fp_type));
            let row_bytes = self.quant_tensor_to_bytes(&chunk_qt);
            *self.rows[row_idx].lock().await = RowData::Ready(row_bytes);
        }
    }

    /// Continuous write delayed for integers - writes multiple rows from a single integer vector.
    pub async fn continous_write_delayed_int(
        &self,
        addr: u32,
        write_amount: u32,
        int_vec: Receiver<Vec<i32>>,
    ) {
        let start_row_idx = self.addr_to_row_idx(addr);

        // Await the integer vector from the channel
        let int_vec = int_vec.await.unwrap();
        // println!("addr = {:?}", addr);
        // println!("in write int_vec = {:?}", int_vec);
        let total_elements = int_vec.len();
        let chunk_size = self.vlen as usize;
        let num_chunks = write_amount.min(((total_elements + chunk_size - 1) / chunk_size) as u32);

        for i in 0..num_chunks {
            let row_idx = start_row_idx + i as usize;
            if row_idx >= self.depth {
                break;
            }

            let start = (i as usize) * chunk_size;
            let end = (start + chunk_size).min(total_elements);
            let chunk = &int_vec[start..end];

            // Convert to bytes (will pad to VLEN if needed)
            let row_bytes = self.int_vec_to_bytes(chunk, self.vlen);
            *self.rows[row_idx].lock().await = RowData::Ready(row_bytes);
        }
    }

    /// Load data from bytes into the SRAM.
    ///
    /// This is used for preloading the SRAM with test data.
    pub async fn load_from_bytes(&self, bytes: &[u8]) {
        let element_size = self.fp_type.size_in_bits() as usize / 8;
        let bytes_per_element = element_size;
        let total_elements = bytes.len() / bytes_per_element;
        let num_rows = (total_elements + self.vlen as usize - 1) / self.vlen as usize;

        for row_idx in 0..num_rows.min(self.depth) {
            let start_element = row_idx * self.vlen as usize;
            let end_element = (start_element + self.vlen as usize).min(total_elements);
            let elements_in_row = end_element - start_element;

            let start_byte = start_element * bytes_per_element;
            let end_byte = end_element * bytes_per_element;

            // Convert bytes to f32 values
            let mut vec = vec![0f32; elements_in_row];
            self.fp_type
                .convert_bytes_to_f32_vec(&bytes[start_byte..end_byte], &mut vec);

            // Pad with zeros if needed
            if elements_in_row < self.vlen as usize {
                vec.resize(self.vlen as usize, 0.0f32);
            }

            // Create QuantTensor and convert to bytes
            let tensor = Tensor::from_slice(&vec);
            let quant_tensor = QuantTensor::quantize(tensor, MxDataType::Plain(self.fp_type));
            let row_bytes = self.quant_tensor_to_bytes(&quant_tensor);
            *self.rows[row_idx].lock().await = RowData::Ready(row_bytes);
        }
    }

    /// Dump the entire SRAM content as bytes.
    ///
    /// This returns the raw binary representation of all stored data.
    pub async fn as_bytes(&self) -> Vec<u8> {
        let mut result = Vec::new();
        let mut _row_idx = 0;

        for row_mutex in &self.rows {
            let mut guard = row_mutex.lock().await;

            // Handle pending writes
            if let RowData::Pending(ref mut receiver) = *guard {
                let tensor = receiver.await.unwrap();
                let row_bytes = self.quant_tensor_to_bytes(&tensor);
                *guard = RowData::Ready(row_bytes);
            }

            // Read the row data
            let row_bytes = match &*guard {
                RowData::Ready(bytes) => bytes.clone(),
                RowData::Pending(_) => unreachable!(),
            };
            _row_idx += 1;
            result.extend_from_slice(&row_bytes);
        }

        result
    }

    // Helper methods

    /// Convert address (in element units) to row index
    fn addr_to_row_idx(&self, addr: u32) -> usize {
        assert!(addr % self.vlen == 0, "Address must be multiple of vlen");
        (addr / self.vlen) as usize
    }

    /// Clip a tensor to VLEN size
    fn clip_to_vlen(&self, tensor: &QuantTensor) -> QuantTensor {
        let tensor_data = tensor.as_tensor();
        let len = tensor_data.size1().unwrap() as i64;

        if len <= self.vlen as i64 {
            tensor.clone()
        } else {
            let clipped = tensor_data.narrow(0, 0, self.vlen as i64);
            QuantTensor::quantize(clipped, tensor.data_type())
        }
    }

    /// Convert QuantTensor to bytes (FP format)
    fn quant_tensor_to_bytes(&self, tensor: &QuantTensor) -> Vec<u8> {
        let tensor_data = tensor.as_tensor();
        let len = tensor_data.size1().unwrap() as usize;
        let f32_slice =
            unsafe { core::slice::from_raw_parts(tensor_data.data_ptr() as *const f32, len) };

        let total_bits = len * self.fp_type.size_in_bits() as usize;
        let bytes_needed = (total_bits + 7) / 8;
        let mut bytes = vec![0u8; bytes_needed];
        self.fp_type.bytes_from_f32(f32_slice, &mut bytes);
        bytes
    }

    /// Convert bytes to QuantTensor (FP format)
    fn bytes_to_quant_tensor(&self, bytes: &[u8], expected_len: u32) -> QuantTensor {
        let bytes_per_element = self.fp_type.size_in_bits() as usize / 8;
        let num_elements = bytes.len() / bytes_per_element;
        let actual_len = num_elements.min(expected_len as usize);

        let mut vec = vec![0f32; actual_len];
        self.fp_type
            .convert_bytes_to_f32_vec(&bytes[..actual_len * bytes_per_element], &mut vec);

        // Pad to expected_len if needed
        if actual_len < expected_len as usize {
            vec.resize(expected_len as usize, 0.0f32);
        }

        let tensor = Tensor::from_slice(&vec);
        QuantTensor::quantize(tensor, MxDataType::Plain(self.fp_type))
    }

    /// Convert integer vector to bytes
    fn int_vec_to_bytes(&self, int_vec: &[i32], expected_len: u32) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(expected_len as usize * self.int_size_bytes);

        // Write the actual integers
        for &val in int_vec.iter() {
            bytes.extend_from_slice(&val.to_le_bytes());
        }

        // Pad with zeros to expected_len
        bytes.resize(expected_len as usize * self.int_size_bytes, 0);

        bytes
    }

    /// Convert bytes to integer vector
    fn bytes_to_int_vec(&self, bytes: &[u8], expected_len: u32) -> Vec<i32> {
        let mut result = Vec::with_capacity(expected_len as usize);
        let mut offset = 0;

        for _ in 0..expected_len as usize {
            if offset + self.int_size_bytes <= bytes.len() {
                let mut int_bytes = [0u8; 4];
                let copy_len = self.int_size_bytes.min(4);
                int_bytes[..copy_len].copy_from_slice(&bytes[offset..offset + copy_len]);
                let val = i32::from_le_bytes(int_bytes);
                result.push(val);
                offset += self.int_size_bytes;
            } else {
                result.push(0);
            }
        }

        result
    }
}
