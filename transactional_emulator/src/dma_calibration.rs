//! Shared entry point for timing the production DMA path in calibration tools.

use std::sync::Arc;

use memory::ErasedMemoryModel;
use quantize::{DataType, FpType, IntType, MxDataType};
use sram::{MatrixSram, VectorSram};

use crate::dma::{self, AddressUnit, MxRegion};

#[derive(Clone, Copy, Debug)]
pub enum DmaOpcode {
    PrefetchMatrix,
    PrefetchVector,
    StoreVector,
}

#[derive(Clone, Copy, Debug)]
pub struct DmaFormat {
    pub family: DmaFormatFamily,
    pub element_bits: u32,
    pub scale_bits: u32,
    pub block: u32,
    pub exponent_bits: u32,
    pub mantissa_bits: u32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DmaFormatFamily {
    MxInt,
    MxFp,
}

#[derive(Clone, Debug, Eq, PartialEq, serde::Serialize)]
pub struct DmaRequestManifest {
    pub read_lines: Vec<u64>,
    pub write_lines: Vec<u64>,
    pub full_lines: u64,
    pub partial_lines: u64,
    pub read_bytes: u64,
    pub write_bytes: u64,
}

#[derive(Clone, Copy, Debug)]
pub struct DmaTransfer {
    pub opcode: DmaOpcode,
    pub element_base: u64,
    pub scale_base: u64,
    pub dim: u32,
    pub amount: u32,
    pub stride: u32,
    pub rstride: u8,
    pub write_amount: u32,
    pub format: DmaFormat,
}

fn mx_type(format: DmaFormat) -> Result<MxDataType, String> {
    if !matches!(format.element_bits, 4 | 8)
        || format.scale_bits != 8
        || !matches!(format.block, 8 | 64)
    {
        return Err(format!(
            "DMA calibration supports e4/e8:s8 with block 8/64, got e{}:s{}:b{}",
            format.element_bits, format.scale_bits, format.block
        ));
    }
    let elem = match format.family {
        DmaFormatFamily::MxInt => DataType::Int(IntType {
            width: format.element_bits,
        }),
        DmaFormatFamily::MxFp => {
            if 1 + format.exponent_bits + format.mantissa_bits != format.element_bits {
                return Err(format!(
                    "MXFP E{}M{} does not match {}-bit element width",
                    format.exponent_bits, format.mantissa_bits, format.element_bits
                ));
            }
            DataType::Fp(FpType {
                sign: true,
                exponent: format.exponent_bits as u8,
                mantissa: format.mantissa_bits as u8,
            })
        }
    };
    Ok(MxDataType::Mx {
        elem,
        scale: DataType::Fp(FpType::E8M0),
        block: format.block,
    })
}

fn region(transfer: DmaTransfer) -> Result<MxRegion, String> {
    Ok(MxRegion {
        hbm_type: mx_type(transfer.format)?,
        index: transfer.element_base,
        scale_index: transfer.scale_base,
        rstride: transfer.rstride,
        stride: transfer.stride,
        stride_unit: AddressUnit::Bytes,
    })
}

fn direction(opcode: DmaOpcode) -> dma::DmaTransferDirection {
    match opcode {
        DmaOpcode::PrefetchMatrix | DmaOpcode::PrefetchVector => dma::DmaTransferDirection::Read,
        DmaOpcode::StoreVector => dma::DmaTransferDirection::Write,
    }
}

pub fn request_manifest(transfer: DmaTransfer) -> Result<DmaRequestManifest, String> {
    if transfer.dim == 0 || transfer.amount == 0 {
        return Err("DMA dim and amount must be positive".to_owned());
    }
    let plan = dma::plan_mx_dma_lines(
        region(transfer)?,
        transfer.dim,
        transfer.amount,
        direction(transfer.opcode),
    );
    Ok(DmaRequestManifest {
        read_bytes: plan.reads.len() as u64 * 64,
        write_bytes: plan.writes.len() as u64 * 64,
        read_lines: plan.reads.into_iter().map(|line| line.address).collect(),
        write_lines: plan.writes.into_iter().map(|line| line.address).collect(),
        full_lines: plan.full_lines,
        partial_lines: plan.partial_lines,
    })
}

/// Time one DMA through the production packed-layout and line-planning path.
pub async fn execute_transactional_dma_timing(
    hbm: Arc<dyn ErasedMemoryModel>,
    transfer: DmaTransfer,
) -> Result<DmaRequestManifest, String> {
    let manifest = request_manifest(transfer)?;
    dma::execute_mx_dma_timing(
        &hbm,
        region(transfer)?,
        transfer.dim,
        transfer.amount,
        direction(transfer.opcode),
    )
    .await;
    Ok(manifest)
}

/// Execute one instruction through the same DMA and delayed-SRAM path as dispatch.rs.
pub async fn execute_transactional_dma(
    hbm: Arc<dyn ErasedMemoryModel>,
    transfer: DmaTransfer,
) -> Result<(), String> {
    if transfer.dim == 0 || transfer.amount == 0 {
        return Err("DMA dim and amount must be positive".to_owned());
    }
    let sram_type = MxDataType::Plain(DataType::Fp(FpType::BF16));
    let statistics = Arc::new(dma::DmaStatistics::default());
    let region = region(transfer)?;

    match transfer.opcode {
        DmaOpcode::PrefetchMatrix => {
            if transfer.write_amount == 0
                || !transfer.dim.is_multiple_of(transfer.write_amount)
                || !transfer.amount.is_multiple_of(transfer.write_amount)
            {
                return Err(format!(
                    "invalid matrix write_amount={} for dim={} amount={}",
                    transfer.write_amount, transfer.dim, transfer.amount
                ));
            }
            let tile_count = transfer.amount.div_ceil(transfer.write_amount).max(1) as usize;
            let mram = MatrixSram::new(transfer.dim, transfer.dim as usize * tile_count, sram_type);
            let receiver = dma::transfer_mx_from_hbm(
                &hbm,
                &statistics,
                region,
                mram.ty(),
                transfer.dim,
                transfer.amount,
                transfer.write_amount,
            );
            mram.continous_write_delayed(0, transfer.amount, receiver)
                .await;
        }
        DmaOpcode::PrefetchVector => {
            let vram = VectorSram::from_mx_type(
                transfer.dim,
                transfer.amount.div_ceil(1).max(1) as usize,
                sram_type,
            );
            let receiver = dma::transfer_mx_from_hbm(
                &hbm,
                &statistics,
                region,
                vram.ty(),
                transfer.dim,
                transfer.amount,
                1,
            );
            vram.continous_write_delayed(0, transfer.amount, receiver)
                .await;
        }
        DmaOpcode::StoreVector => {
            let vram = Arc::new(VectorSram::from_mx_type(
                transfer.dim,
                transfer.amount as usize,
                sram_type,
            ));
            dma::transfer_mx_to_hbm(
                &hbm,
                &statistics,
                &vram,
                region,
                0,
                transfer.dim,
                transfer.amount,
            )
            .await;
        }
    }
    Ok(())
}
