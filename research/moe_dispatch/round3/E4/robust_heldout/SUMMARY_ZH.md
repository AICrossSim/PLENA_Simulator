# 开发集近优候选的留出集鲁棒诊断

近优集合在开发集冻结：各模式/C0/C1/族已实际评估的候选，GM距该族已测最好值不超过1%。全域证明开放，不能称经证明的全域1%近优集合。本诊断不修改selected_designs.json或任何主表。

三指标均使用相同模式C0冻结B1的逐窗口配对比值：GM；最差ceil(0.1×135)个比值的算术平均CVaR10；各batch配对GM的最大值。每配置在135个留出窗口上独立两遍，或引用已经核验的E5两遍物理回放。200次bootstrap只重采样18个开发窗口。留出赢家是后验诊断，不重新定主硬件，也不构成盲测。

| 模式 | 约束 | 族 | 近优候选 | 三目标同配置 | GM赢家 | CVaR10赢家 | 最坏batch赢家 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| pipelined | C0 | 3+3 | 1 | True | 1x48x128+4x3x512 [f72a23d0] | 1x48x128+4x3x512 [f72a23d0] | 1x48x128+4x3x512 [f72a23d0] |
| pipelined | C0 | 4+2 | 1 | True | 1x64x128+2x32x64 [d581ce11] | 1x64x128+2x32x64 [d581ce11] | 1x64x128+2x32x64 [d581ce11] |
| pipelined | C0 | 5+1 | 2 | True | 1x40x256+2x2x512 [541d03ba] | 1x40x256+2x2x512 [541d03ba] | 1x40x256+2x2x512 [541d03ba] |
| pipelined | C0 | homogeneous | 3 | False | 1x48x128+1x48x128 [c9c20530] | 1x48x128+1x48x128 [bee39f37] | 1x48x128+1x48x128 [bee39f37] |
| pipelined | C0 | single | 6 | True | 1x48x256 [0bee7728] | 1x48x256 [0bee7728] | 1x48x256 [0bee7728] |
| pipelined | C1 | 3+3 | 1 | True | 1x48x128+4x3x512 [f72a23d0] | 1x48x128+4x3x512 [f72a23d0] | 1x48x128+4x3x512 [f72a23d0] |
| pipelined | C1 | 4+2 | 1 | True | 1x64x128+2x32x64 [d581ce11] | 1x64x128+2x32x64 [d581ce11] | 1x64x128+2x32x64 [d581ce11] |
| pipelined | C1 | 5+1 | 1 | True | 1x40x256+2x2x512 [538f4e66] | 1x40x256+2x2x512 [538f4e66] | 1x40x256+2x2x512 [538f4e66] |
| pipelined | C1 | homogeneous | 3 | False | 1x48x128+1x48x128 [c9c20530] | 1x48x128+1x48x128 [bee39f37] | 1x48x128+1x48x128 [bee39f37] |
| pipelined | C1 | single | 6 | True | 1x48x256 [0bee7728] | 1x48x256 [0bee7728] | 1x48x256 [0bee7728] |
| port_tight | C0 | 3+3 | 42 | True | 3x16x128+3x16x128 [4f81fd22] | 3x16x128+3x16x128 [4f81fd22] | 3x16x128+3x16x128 [4f81fd22] |
| port_tight | C0 | 4+2 | 24 | False | 1x32x128+2x32x128 [c85baa7e] | 1x32x128+1x64x128 [def7102d] | 1x32x128+2x32x128 [c85baa7e] |
| port_tight | C0 | 5+1 | 20 | False | 2x40x128+4x8x64 [7a8d8175] | 1x4x512+4x20x128 [18076db8] | 1x64x32+2x40x128 [1e2e542a] |
| port_tight | C0 | homogeneous | 39 | True | 3x16x128+3x16x128 [d41c0d1a] | 3x16x128+3x16x128 [d41c0d1a] | 3x16x128+3x16x128 [d41c0d1a] |
| port_tight | C0 | single | 38 | True | 3x32x128 [e76e0dba] | 3x32x128 [e76e0dba] | 3x32x128 [e76e0dba] |
| port_tight | C1 | 3+3 | 31 | True | 3x16x128+3x16x128 [4f81fd22] | 3x16x128+3x16x128 [4f81fd22] | 3x16x128+3x16x128 [4f81fd22] |
| port_tight | C1 | 4+2 | 26 | False | 2x16x128+4x16x128 [1f5a2035] | 1x32x128+1x64x128 [def7102d] | 1x64x128+2x16x128 [f954e13f] |
| port_tight | C1 | 5+1 | 15 | False | 2x40x128+4x8x64 [7a8d8175] | 1x4x512+4x20x128 [18076db8] | 1x64x32+2x40x128 [1e2e542a] |
| port_tight | C1 | homogeneous | 36 | True | 3x16x128+3x16x128 [d41c0d1a] | 3x16x128+3x16x128 [d41c0d1a] | 3x16x128+3x16x128 [d41c0d1a] |
| port_tight | C1 | single | 38 | True | 3x32x128 [e76e0dba] | 3x32x128 [e76e0dba] | 3x32x128 [e76e0dba] |
| pipelined | C0 | all | 13 | True | 1x48x256 [0bee7728] | 1x48x256 [0bee7728] | 1x48x256 [0bee7728] |
| pipelined | C1 | all | 12 | True | 1x48x256 [0bee7728] | 1x48x256 [0bee7728] | 1x48x256 [0bee7728] |
| port_tight | C0 | all | 160 | True | 3x32x128 [e76e0dba] | 3x32x128 [e76e0dba] | 3x32x128 [e76e0dba] |
| port_tight | C1 | all | 143 | True | 3x32x128 [e76e0dba] | 3x32x128 [e76e0dba] | 3x32x128 [e76e0dba] |

同形状不同容量/端口/数据流仍是不同配置，赢家相同按完整物理配置ID判断。MILP为资源分配松弛，LPT为有限资源合法回放；不称任意时序全局最优。本节均为BF16、256 GB/s上限、相位流体解析估计，非原生HBM/RTL。
