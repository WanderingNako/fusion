# config.py

# 随机种子（保证可复现）
SEED = 42
# 模型基础参数
D_MODEL = 512
BLOCK_SIZE = 64  # 融合算子分块大小
# 测试参数
BATCH_SIZE = 2
SEQ_LENS = [1024, 2048, 4096]  # 待测试的序列长度
# 性能测量参数
WARMUP_TIMES = 5    # CUDA预热次数
TEST_TIMES = 10     # 性能测试次数（取平均）
# 梯度校验参数（短序列，提升校验速度）
GRAD_CHECK_SEQ_LEN = 128
GRAD_CHECK_ATOL = 1e-5  # 梯度误差容忍度
GRAD_CHECK_EPS = 1e-6   # 数值梯度步长
