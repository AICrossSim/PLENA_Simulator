//! Compiler/Simulator route ABI execution, including the actual opcode path.
use std::sync::{Arc, Mutex};

use memory::{ErasedMemoryModel, MemoryBacked, NaiveTiming, WithTiming};
use quantize::{DataType, FpType, MxDataType, QuantTensor, tensor_to_f32_vec};
use runtime::{Executor, Instant};
use sram::{MatrixSram, VectorSram};
use tch::Tensor;

use super::{Accelerator, Scoreboard, TimingDriver};
use crate::matrix_machine::MatrixMachine;
use crate::op::Opcode;
use crate::runtime_config::{BLEN, BROADCAST_AMOUNT, HLEN, MATRIX_SRAM_TYPE, MLEN, VLEN};
use crate::timing::{TimingMode, set_timing_mode};
use crate::vector_machine::VectorMachine;

async fn accelerator(vram: Arc<VectorSram>) -> Accelerator {
    let mram = Arc::new(MatrixSram::new(
        *MLEN,
        (*MLEN as usize) * 16,
        *MATRIX_SRAM_TYPE,
    ));
    let matrix = MatrixMachine::new(mram, vram.clone(), *MLEN, *HLEN, *BLEN, *BROADCAST_AMOUNT);
    let vector = VectorMachine::new(vram, *VLEN, *HLEN);
    let hbm: Arc<dyn ErasedMemoryModel> = Arc::new(WithTiming::new(
        NaiveTiming::preset_ddr4_2400p(4),
        MemoryBacked::with_capacity(4096),
    ));
    Accelerator::new(matrix, vector, hbm)
}

async fn execute(accelerator: &mut Accelerator, ops: &[Opcode], pipelined: bool) {
    if pipelined {
        let mut scoreboard = Scoreboard::new(false);
        accelerator
            .do_ops(
                ops,
                None,
                TimingDriver::Scoreboard {
                    scoreboard: &mut scoreboard,
                },
            )
            .await;
    } else {
        accelerator.do_ops(ops, None, TimingDriver::Serial).await;
    }
}

#[tokio::test]
async fn dispatch_topk_honors_sigmoid_and_bias_without_changing_route_weights() {
    for pipelined in [false, true] {
        set_timing_mode(if pipelined {
            TimingMode::Scoreboard
        } else {
            TimingMode::Serial
        });
        let executor = Executor::new();
        let result = Arc::new(Mutex::new(None));
        let result_task = result.clone();
        executor.spawn(async move {
            let fp = DataType::Fp(FpType::BF16);
            let ty = MxDataType::Plain(fp);
            let vram = Arc::new(VectorSram::new(*VLEN, 8, fp, 4));
            let mut logits = vec![-100.0f32; *VLEN as usize];
            logits[..3].copy_from_slice(&[3.0, 2.0, 1.0]);
            let mut bias = vec![0.0f32; *VLEN as usize];
            bias[1] = 2.0;
            vram.write(0, QuantTensor::quantize(Tensor::from_slice(&logits), ty))
                .await;
            vram.write(*VLEN, QuantTensor::quantize(Tensor::from_slice(&bias), ty))
                .await;
            let mut a = accelerator(vram).await;
            a.reg_file.write_gp(6, (3 << 8) | 2 | (1 << 22) | (1 << 23));
            a.reg_file.write_gp(13, *VLEN);
            a.reg_file.write_gp(1, 0);
            a.reg_file.write_gp(2, 0);
            a.reg_file.write_gp(3, 0);
            let words = [
                0x38 | (6 << 6),
                0x38 | (13 << 6) | (1 << 10),
                0x37 | (1 << 6) | (2 << 10) | (3 << 14) | (15 << 18),
            ];
            let ops: Vec<_> = words.into_iter().map(Opcode::decode).collect();
            execute(&mut a, &ops, pipelined).await;
            let ids = [a.scalar_sram.read_int(0), a.scalar_sram.read_int(1)];
            let weights = [a.scalar_sram.read_fp(0), a.scalar_sram.read_fp(1)];
            *result_task.lock().unwrap() = Some((ids, weights));
        });
        executor.enter(Instant::ETERNITY).await;
        let (ids, weights) = result.lock().unwrap().take().unwrap();
        assert_eq!(ids, [1, 0]);
        let sigmoid = |v: f32| 1.0 / (1.0 + (-v).exp());
        let denominator = sigmoid(2.0) + sigmoid(3.0);
        assert!((weights[0] - sigmoid(2.0) / denominator).abs() < 0.003);
        assert!((weights[1] - sigmoid(3.0) / denominator).abs() < 0.003);
    }
}

#[tokio::test]
async fn route_opcodes_execute_all_unique_experts_and_preserve_each_tokens_weight_sum() {
    for pipelined in [false, true] {
        set_timing_mode(if pipelined {
            TimingMode::Scoreboard
        } else {
            TimingMode::Serial
        });
        let executor = Executor::new();
        let result = Arc::new(Mutex::new(None));
        let result_task = result.clone();
        executor.spawn(async move {
            let fp = DataType::Fp(FpType::BF16);
            let ty = MxDataType::Plain(fp);
            let vram = Arc::new(VectorSram::new(*VLEN, 20, fp, 4));
            for (token, selected) in [[0, 1], [1, 2], [2, 3], [0, 3]].into_iter().enumerate() {
                let mut logits = vec![-100.0f32; *VLEN as usize];
                for expert in selected {
                    logits[expert] = 0.0;
                }
                vram.write(
                    token as u32 * *VLEN,
                    QuantTensor::quantize(Tensor::from_slice(&logits), ty),
                )
                .await;
                vram.write(
                    (4 + token) as u32 * *VLEN,
                    QuantTensor::quantize(
                        Tensor::ones([*VLEN as i64], (tch::Kind::Float, tch::Device::Cpu)),
                        ty,
                    ),
                )
                .await;
                vram.write(
                    (8 + token) as u32 * *VLEN,
                    QuantTensor::quantize(
                        Tensor::zeros([*VLEN as i64], (tch::Kind::Float, tch::Device::Cpu)),
                        ty,
                    ),
                )
                .await;
            }
            let mut a = accelerator(vram.clone()).await;
            a.reg_file.write_gp(6, (4 << 8) | 2);
            let mut ops = vec![
                Opcode::decode(0x38 | (6 << 6)),
                Opcode::decode(0x39 | (7 << 6) | (4 << 10) | (5 << 14) | (15 << 18)),
            ];
            for token in 0..4u32 {
                ops.extend([
                    Opcode::S_ADDI_INT {
                        rd: 10,
                        rs1: 0,
                        imm: token * 2,
                    },
                    Opcode::S_ADDI_INT {
                        rd: 11,
                        rs1: 0,
                        imm: token * *VLEN,
                    },
                    Opcode::S_ADDI_INT {
                        rd: 12,
                        rs1: 0,
                        imm: token * 2,
                    },
                    Opcode::V_TOPK {
                        rd: 10,
                        rs1: 11,
                        rs2: 12,
                        rmask: 15,
                    },
                ]);
            }
            ops.push(Opcode::decode(0x3A));
            ops.push(Opcode::S_ADDI_INT {
                rd: 13,
                rs1: 13,
                imm: 1,
            });
            for token in 0..4u32 {
                ops.extend([
                    Opcode::S_ADDI_INT {
                        rd: 1,
                        rs1: 0,
                        imm: (4 + token) * *VLEN,
                    },
                    Opcode::S_ADDI_INT {
                        rd: 2,
                        rs1: 0,
                        imm: (12 + token) * *VLEN,
                    },
                    Opcode::S_ADDI_INT {
                        rd: 3,
                        rs1: 0,
                        imm: (8 + token) * *VLEN,
                    },
                    Opcode::decode(0x3C | (2 << 6) | (1 << 10) | (token << 18)),
                    Opcode::V_ADD_VV {
                        rd: 3,
                        rs1: 3,
                        rs2: 2,
                        rmask: 0,
                        lmask: 0,
                    },
                ]);
            }
            ops.push(Opcode::decode(0x3B));
            execute(&mut a, &ops, pipelined).await;
            let mut outputs = Vec::new();
            for token in 0..4 {
                outputs.push(tensor_to_f32_vec(
                    &vram.read((8 + token) * *VLEN).await.as_tensor(),
                ));
            }
            *result_task.lock().unwrap() = Some((a.reg_file.read_gp(13), outputs));
        });
        executor.enter(Instant::ETERNITY).await;
        let (experts, outputs) = result.lock().unwrap().take().unwrap();
        assert_eq!(experts, 4);
        for output in outputs {
            for value in output {
                assert_eq!(value, 1.0);
            }
        }
    }
}
