# ManekiNeko（招财猫）
> 本项目旨在探索躺着赚取加密货币的可能性

[English](README.md) | [中文](README_zh.md)

包含以下部分：
- [x] `getTradeData` 从币安获取交易数据并保存到 sqlite
- [x] `lstmTrain` 使用 LSTM 预测加密货币价格
- [x] `xGBoost` 使用 XGBoost 预测加密货币价格
- [x] `randomForest` 使用随机森林预测加密货币价格
- [ ] `transformer` 使用 Transformer 预测加密货币价格
- [ ] `informer` 使用 Informer 预测加密货币价格
- [ ] `LightGBM / CatBoost` 高效梯度提升变体
- [ ] `N-BEATS` 用于可解释时间序列预测的深度神经架构
- [ ] `Reinforcement Learning (PPO/DQN)` 训练智能体直接做出交易决策

> **[模型实施路线图](docs/model_implementation_roadmap.md)**: 上述高级模型的详细实施计划。
<br>

# 特征工程与信号生成

## 技术指标
本项目生成了广泛的技术指标来捕捉市场动态：
- **动量 (Momentum)**: RSI (14周期)
- **趋势 (Trend)**: EMA (12/26), MACD (信号线, 柱状图), SMA (3, 6, 12, 20)
- **波动率 (Volatility)**: 布林带 (上轨, 中轨, 下轨), 波动率标准差 (5周期)
- **价格差异 (Price Differentials)**: 价格与 SMA 及布林带的距离

## 信号标注逻辑
模型将交易视为多分类问题，信号源自局部极值检测：

### 1. 固定区间极值（主要信号）
数据被划分为不重叠的窗口（默认大小：300）。
- **标签 4 (买入)**: 局部最小值（入场点）
- **标签 6 (卖出)**: 局部最大值（出场点）
*注意：算法会将最小值与随后的最大值配对，以确保有效的交易周期。*

### 2. 滑动窗口极值（"滑点"信号）
滚动窗口扫描时间序列以捕捉更细粒度的市场转折点。
- **标签 2 (滑点买入)**: 滑动窗口内的局部最小值
- **标签 3 (滑点卖出)**: 滑动窗口内的局部最大值

### 3. 默认状态
- **标签 1 (持有)**: 未检测到特定信号

# `getTradeData` 使用方法
复制 `config_example.ini` 为 `config.ini` 并修改 api key 和 secret key

```bash
pip install requirements.txt
python getTradeData.py
```


# 许可证
MIT
