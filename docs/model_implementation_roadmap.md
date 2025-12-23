# 高级交易模型实施路线图

本文档基于 `README.md` 中列出的待开发模型，结合数据处理优化建议（样本权重、特征归一化），整理了详细的实施检查点。

> **核心原则**：所有新模型必须集成“样本权重（class_weight）”或“焦点损失（Focal Loss）”以解决类别不平衡问题。

## 1. 梯度提升树变体 (LightGBM / CatBoost)

目标：利用比 XGBoost 更高效、对类别特征支持更好的算法。

- [ ] **数据准备**：
    - 复用 `randomForest/random_forest.py` 的数据加载逻辑。
    - **优化**：在 `split_data` 前，按币种进行 MinMaxScaler 归一化（解决多币种尺度差异）。
- [ ] **LightGBM 实现**：
    - 安装 `lightgbm`。
    - 构建 Dataset，设置 `categoricals`（如果有）。
    - **关键参数**：`is_unbalance=True` (与 `scale_pos_weight` 互斥) 或手动计算权重数组传入 `weight` 参数。
- [ ] **CatBoost 实现**：
    - 安装 `catboost`。
    - **关键参数**：`auto_class_weights='Balanced'` 或 `class_weights`。
    - 利用其原生的 GPU 支持加速训练。

## 2. Transformer 家族 (Time-Series Transformers)

目标：捕捉长距离时间依赖关系（Long-range Dependencies），优于 LSTM。

- [ ] **基础 Transformer**：
    - 定义 `Time2Vec` 或位置编码层 (Positional Encoding)。
    - 构建 Encoder-Decoder 架构或仅 Encoder 架构（用于分类）。
    - 损失函数：`CrossEntropyLoss`，配合 `weight` 参数（针对稀缺类别加权）。
- [ ] **Informer**：
    - 引入 ProbSparse Attention 机制以降低计算复杂度。
    - 适用于超长序列（Long Sequence）预测。
- [ ] **Temporal Fusion Transformer (TFT)** (Optional):
    - Consider only if interpretability is a strict requirement.

## 3. 深度时间序列模型 (N-BEATS)

目标：通过堆叠全连接层和残差连接，实现纯神经网络的高性能预测。

- [ ] **架构设计**：
    - 实现 Generic Block（通用块）或 Interpretable Block（趋势+季节性块）。
    - 堆叠多个 Stack。
    - 修改输出层为分类层（Softmax），而非默认的回归输出。

## 4. 强化学习 (Reinforcement Learning - PPO/DQN)

目标：跳过“预测价格”步骤，直接学习“交易动作”。

- [ ] **环境构建 (Trading Environment)**：
    - 创建符合 OpenAI Gym 接口的 `CryptoTradingEnv`。
    - **State**: 当前窗口的技术指标特征。
    - **Action**: Hold(0), Buy(1), Sell(2)。
    - **Reward**: 动作执行后的收益率（需扣除手续费，并惩罚频繁交易）。
- [ ] **Agent 训练**：
    - 使用 `Stable Baselines3` 库。
    - 训练 PPO（On-policy）或 DQN（Off-policy）智能体。
    - 验证：在测试集上运行 Agent，统计总收益和夏普比率。


## 5. 通用优化任务 (适用于所有模型)

- [ ] **阈值移动 (Threshold Moving)**：
    - 在预测阶段，不使用默认的 `argmax`。
    - 设置置信度阈值（如 Buy 概率 > 0.4 即买入），并通过回测寻找最佳阈值。
- [ ] **数据管道升级**：
    - 实现 Feature Scaling (Per-Asset)。
    - 实现 Class Weight Calculation (自动统计训练集 Label 频率)。
