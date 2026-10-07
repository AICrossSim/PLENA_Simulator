# E6 边界

MoE层结果直接来自E4，不重复计算。完整模型每token时间缺少与冻结DeepSeek捕获匹配的attention/router/norm时序以及层映射，留空并标缺失。其他模型的旧PLENA时间不能拼接成这套模型的端到端结果。GPU比较未做。整模型计时项为未完成（输入缺失）。
