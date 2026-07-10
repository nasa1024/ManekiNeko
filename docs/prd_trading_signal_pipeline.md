# PRD: ManekiNeko 可交易区间分类与盈利验证系统

版本：v1.0  
状态：Draft  
目标仓库：`nasa1024/ManekiNeko`  
目标分支：`prd/profitable-signal-pipeline-v1`

---

## 1. 背景

ManekiNeko 当前已经具备以下基础能力：

- 从 Binance 获取 K 线数据并保存到 SQLite。
- 生成 RSI、EMA、MACD、SMA、布林带、波动率、价格偏离等技术指标。
- 通过局部极值生成 `Hold / Slip Buy / Slip Sell / Primary Buy / Primary Sell` 等分类标签。
- 使用 XGBoost、RandomForest、LSTM 等模型进行实验。

现阶段的核心方向应从“预测价格”或“预测极值点”升级为：

> 根据过去一段时间的市场状态，判断未来一段时间内是否进入具备正期望的可交易区间。

这个 PRD 的目标是把项目升级成一条可复现、可回测、可逐步上线的交易信号流水线。

---

## 2. 产品目标

### 2.1 核心目标

构建一个面向现货 long-only 的交易信号训练与验证系统，输出高置信度的买入区间信号，并通过严格回测验证其扣费后是否具备正期望。

### 2.2 业务目标

1. 将当前“极值点分类”升级为“可交易区间分类”。
2. 明确把手续费、滑点、止损、止盈、最长持仓时间纳入标签定义和回测。
3. 建立从数据采集、特征工程、标签生成、模型训练、回测、paper trading 到实盘监控的闭环。
4. 优先验证策略正期望，而不是单纯追求 classification accuracy。

### 2.3 非目标

当前阶段不做：

- 杠杆交易。
- 合约交易。
- 高频秒级交易。
- 自动全仓交易。
- 不经回测和 paper trading 的真钱上线。
- 直接预测未来价格作为主任务。

---

## 3. 成功标准

第一阶段策略必须满足以下最低标准，才允许进入 paper trading：

| 指标 | 最低要求 |
|---|---:|
| 测试集交易次数 | >= 300 |
| 扣费后 Profit Factor | > 1.2 |
| 最大回撤 | < 15% |
| 年化 Sharpe | > 1.0 |
| 平均单笔净收益 | > 2 × 单笔交易成本 |
| 不同年份表现 | 不能只在单一年份有效 |
| 不同币种表现 | 不能只依赖单一币种 |
| Paper trading | 连续 3 个月不崩溃 |

注意：以上标准不是盈利保证，只是进入下一阶段的最低门槛。

---

## 4. 策略范围

### 4.1 初始交易方向

第一版只做：

```text
现货 long-only
```

暂不做：

```text
short / margin / futures / leverage
```

### 4.2 推荐周期

第一版推荐：

```text
主周期：15m
辅助大周期：1h / 4h
```

15m 用于入场判断，1h / 4h 用于趋势过滤。

### 4.3 推荐币种

第一版只选择高流动性 USDT 交易对：

```text
BTCUSDT, ETHUSDT, BNBUSDT, SOLUSDT, XRPUSDT, ADAUSDT, DOGEUSDT, AVAXUSDT 等
```

建议从前 10 个币开始，验证通过后扩展到前 30~50 个高成交量币。

---

## 5. 数据需求

### 5.1 数据来源

- Binance Kline 数据。
- 需要字段：open_time, open, high, low, close, volume, close_time, quote_asset_volume, number_of_trades, taker_buy_base_asset_volume, taker_buy_quote_asset_volume。

### 5.2 数据量建议

第一版建议：

```text
15m K线
前 30 个高流动性 USDT 交易对
2020-01-01 至今
```

粗略估算：

```text
15m 一年约 35,040 根 K线
6 年每币约 210,000 根
30 个币约 600 万根 K线
```

### 5.3 有效样本目标

深度学习不是看总 K 线数量，而是看有效正样本数量。

建议每个核心标签至少具备：

```text
5,000 ~ 20,000 个有效样本
```

如果买入区间标签占比为：

| Buy Zone 占比 | 需要总样本量 |
|---:|---:|
| 0.5% | 100万 ~ 400万 |
| 2% | 25万 ~ 100万 |
| 5% | 10万 ~ 40万 |

---

## 6. 标签设计

### 6.1 当前问题

现有极值标签更接近：

```text
某个 index 是否为事后局部最低点 / 最高点
```

这可以用于训练，但不完全等价于实盘问题。实盘更关心：

```text
现在买入后，未来 H 根 K 线内是否能先达到止盈，且没有先触发止损？
```

### 6.2 新标签目标

使用类似 Triple Barrier 的交易结果标签。

对每个时间点 `t`，设置：

```text
lookback = 过去输入窗口长度
horizon = 未来最大持仓窗口
profit_barrier = 止盈线
loss_barrier = 止损线
fee_rate = 手续费
slippage_rate = 滑点
```

计算从 `t` 开始，未来 `horizon` 根内：

1. 是否先触发止盈。
2. 是否先触发止损。
3. 是否到期仍未触发。

### 6.3 推荐初始参数

15m 第一版：

```yaml
lookback: 240
horizon: 48        # 12 小时
profit_barrier: 0.012
loss_barrier: -0.006
fee_rate_round_trip: 0.002
slippage_round_trip: 0.001
```

可根据币种波动率动态调整 barrier。

### 6.4 多分类标签

保留项目已有语义，但重新定义为交易区间：

```text
1 = Hold / No Trade
2 = Buy Zone
3 = Sell / Exit Zone
4 = Strong Buy
6 = Strong Sell / Strong Exit
```

第一版也可以简化为三分类：

```text
0 = Hold
1 = Buy Zone
2 = Avoid / Bad Entry
```

### 6.5 Buy Zone 扩展逻辑

不只标记最低点，而是扩展为买入区间：

```python
def expand_buy_zone(signal, buy_indices, pre=12, post=2):
    for idx in buy_indices:
        start = max(0, idx - pre)
        end = min(len(signal), idx + post + 1)
        signal[start:end] = 2
        signal[idx] = 4
    return signal
```

原则：

- 最低点前的区间比最低点后的区间更重要。
- 实盘目标是提前识别可买区，而不是事后确认最低点。

---

## 7. 特征工程设计

### 7.1 特征原则

所有输入特征必须只使用当前和过去数据。

禁止：

- 将 `signal` 作为输入特征。
- 使用未来 rolling 统计。
- 在 train/test 切分前对全量数据 fit scaler。
- 输入绝对未来时间信息。

### 7.2 推荐保留特征

价格结构：

```text
log_return_1
log_return_3
log_return_6
log_return_12
high_low_range = (high - low) / close
open_close_range = (close - open) / open
```

趋势偏离：

```text
close / sma_20 - 1
close / ema_12 - 1
close / ema_26 - 1
ema_12 / ema_26 - 1
```

波动率：

```text
rolling_std_12
rolling_std_48
atr_like_range
bollinger_position
```

成交量：

```text
volume / rolling_volume_mean_20 - 1
quote_volume_change
trade_count_change
buy_volume_ratio
```

多周期上下文：

```text
1h trend filter
4h trend filter
market regime feature
```

### 7.3 建议去掉或变换的特征

不建议直接输入：

```text
open_time
close_time
绝对 open/high/low/close
绝对 sma/ema/bollinger 上下轨
```

建议改成相对特征，减少不同币种价格尺度差异。

---

## 8. 模型设计

### 8.1 Baseline 模型

必须先跑 baseline：

```text
LightGBM / XGBoost
TCN
Small GRU
```

如果 baseline 没有正期望，不应直接扩大深度模型。

### 8.2 深度学习输入格式

保留时间结构：

```text
X shape = [batch_size, lookback, feature_dim]
```

禁止深度学习主模型直接使用：

```text
[batch_size, lookback * feature_dim]
```

### 8.3 推荐第一版深度模型：Patch Transformer Classifier

结构：

```text
[B, T, F]
→ Patch 时间切片
→ Linear(F * patch_len -> d_model)
→ Positional Encoding
→ Transformer Encoder
→ Mean/Attention Pooling
→ Classifier
```

推荐参数：

```yaml
lookback: 240
patch_len: 8
num_tokens: 30
d_model: 96
nhead: 4
encoder_layers: 2
dropout: 0.1
```

优点：

- 保留时间序列结构。
- 大幅减少 attention token 数量。
- 显存占用远低于直接对 240 根 K 线做标准 Transformer。

### 8.4 显存优化要求

必须实现：

1. Dataset 保持在 CPU，不允许初始化时整体 `.to(device)`。
2. 每个 batch 在训练循环中再移动到 GPU。
3. 支持 AMP mixed precision。
4. 支持 gradient accumulation。
5. 支持减小 batch size。
6. 可选 activation checkpointing。

错误示例：

```python
self.X = torch.tensor(X).to(device)
```

正确示例：

```python
self.X = torch.tensor(X, dtype=torch.float32)

for x, y in loader:
    x = x.to(device, non_blocking=True)
    y = y.to(device, non_blocking=True)
```

---

## 9. 训练与验证设计

### 9.1 时间切分

禁止随机切分。

推荐：

```text
Train:      2020-01-01 ~ 2023-12-31
Validation: 2024-01-01 ~ 2024-12-31
Test:       2025-01-01 ~ 2025-12-31
Paper:      2026-01-01 ~ now
```

### 9.2 Purge / Embargo

滑动窗口样本高度重叠，切分边界必须加隔离区。

```text
purge_length >= lookback + horizon
```

例如：

```text
lookback = 240
horizon = 48
purge >= 288 根 15m K线
```

### 9.3 类别不平衡

必须支持至少一种：

- class_weight。
- sample_weight。
- focal loss。
- balanced sampling。

重点关注 Buy Zone 的 precision 和 recall，而不是全量 accuracy。

---

## 10. 回测设计

### 10.1 回测必须扣除成本

每笔交易必须扣除：

```text
买入手续费
卖出手续费
滑点
```

第一版默认：

```yaml
round_trip_fee: 0.002
round_trip_slippage: 0.001
```

### 10.2 交易规则

第一版 long-only：

```text
当 buy_zone_prob > threshold 且大周期趋势过滤通过：买入
达到止盈：卖出
达到止损：卖出
达到最长持仓时间：卖出
出现 strong_sell / exit_zone：卖出
```

### 10.3 回测指标

必须输出：

```text
net_return
max_drawdown
profit_factor
sharpe
sortino
win_rate
avg_win
avg_loss
avg_trade_return
trade_count
monthly_return
per_symbol_return
```

### 10.4 阈值搜索

不要默认使用 `argmax`。

需要在 validation 上搜索：

```text
buy_threshold: 0.55 ~ 0.90
stop_loss: 0.4% ~ 1.2%
take_profit: 0.8% ~ 2.5%
max_holding_bars: 16 ~ 96
```

最终只在 test 上评估一次，避免过拟合测试集。

---

## 11. 系统模块规划

建议新增以下模块：

```text
manekineko/
  data/
    collect_binance.py
    symbol_universe.py
  features/
    technical.py
    multi_timeframe.py
    normalize.py
  labeling/
    triple_barrier.py
    zone_expansion.py
  datasets/
    window_dataset.py
  models/
    patch_transformer.py
    tcn.py
  training/
    train_classifier.py
    losses.py
    metrics.py
  backtest/
    event_backtester.py
    costs.py
    reports.py
  configs/
    signal_v1.yaml
```

---

## 12. CLI 规划

建议提供统一命令：

```bash
python -m manekineko.data.collect_binance --config configs/signal_v1.yaml
python -m manekineko.features.build_features --config configs/signal_v1.yaml
python -m manekineko.labeling.build_labels --config configs/signal_v1.yaml
python -m manekineko.training.train_classifier --config configs/signal_v1.yaml
python -m manekineko.backtest.run_backtest --config configs/signal_v1.yaml
```

---

## 13. 配置样例

```yaml
data:
  interval: 15m
  symbols:
    - BTCUSDT
    - ETHUSDT
    - BNBUSDT
    - SOLUSDT
  start_date: "2020-01-01"
  end_date: null

features:
  use_absolute_price: false
  use_time_features: true
  use_multi_timeframe: true
  higher_timeframes: [1h, 4h]

labeling:
  lookback: 240
  horizon: 48
  profit_barrier: 0.012
  loss_barrier: -0.006
  fee_round_trip: 0.002
  slippage_round_trip: 0.001
  buy_zone_pre: 12
  buy_zone_post: 2

model:
  type: patch_transformer
  patch_len: 8
  d_model: 96
  nhead: 4
  num_layers: 2
  dropout: 0.1

training:
  batch_size: 64
  accumulation_steps: 1
  amp: true
  epochs: 50
  early_stopping_patience: 8
  class_weight: balanced

backtest:
  buy_threshold: 0.65
  take_profit: 0.012
  stop_loss: 0.006
  max_holding_bars: 48
  min_quote_volume: 1000000
```

---

## 14. 实施里程碑

### P0: 数据与标签审计

- [ ] 确认所有输入特征无未来信息。
- [ ] 移除输入中的 `signal`, `tag`, `open_time`, `close_time`。
- [ ] scaler 只在 train 上 fit。
- [ ] 实现时间切分与 purge。

### P1: Triple Barrier 标签

- [ ] 实现 `triple_barrier.py`。
- [ ] 实现 buy zone 扩展。
- [ ] 输出标签分布报告。
- [ ] 对比旧 extrema 标签和新交易结果标签。

### P2: Baseline 训练

- [ ] LightGBM / XGBoost baseline。
- [ ] 输出 non-hold precision / recall。
- [ ] 输出 validation 阈值搜索结果。

### P3: Patch Transformer

- [ ] 实现 CPU Dataset。
- [ ] 实现 Patch Transformer Classifier。
- [ ] 支持 AMP。
- [ ] 支持 gradient accumulation。
- [ ] 与 baseline 对比。

### P4: 回测系统

- [ ] 扣费回测。
- [ ] 输出交易明细。
- [ ] 输出收益曲线。
- [ ] 输出 per-symbol / per-month 报告。

### P5: Paper Trading

- [ ] 接入实时数据。
- [ ] 固定模型版本。
- [ ] 记录每次信号、价格、执行结果。
- [ ] 连续运行至少 3 个月。

---

## 15. 风控要求

第一版实盘前必须具备：

```text
单币最大仓位限制
总仓位限制
连续亏损熔断
最大日亏损熔断
低流动性过滤
异常波动过滤
API 失败保护
模型置信度不足时不交易
```

推荐初始设置：

```yaml
max_position_per_symbol: 5%
max_total_exposure: 30%
max_daily_loss: 3%
max_consecutive_losses: 5
cooldown_after_loss_bars: 16
```

---

## 16. 主要风险

1. 标签过拟合历史局部极值，无法泛化到未来。
2. 类别不平衡导致模型过度预测 Hold。
3. 回测未充分扣除滑点和手续费。
4. 多币种训练中价格尺度不一致。
5. 随机切分或 scaler 泄漏导致测试表现虚高。
6. 大模型显存压力高，导致 batch 太小、训练不稳定。
7. 市场制度变化导致历史规律失效。

---

## 17. 第一版推荐方向

最小可行路线：

```text
15m 现货 long-only
前 10~30 个高流动性 USDT 交易对
Triple Barrier 标签
Buy Zone 区间扩展
XGBoost / LightGBM baseline
Patch Transformer Classifier
严格扣费回测
Validation 阈值搜索
Test 一次性验证
Paper trading 3 个月
```

第一版不要追求复杂模型，先证明：

```text
标签定义 + 特征工程 + 回测逻辑
```

能否产生稳定正期望。

---

## 18. 验收标准

PRD 进入实现阶段的验收条件：

- [ ] 数据范围和币种范围确定。
- [ ] 标签参数确定。
- [ ] 回测成本假设确定。
- [ ] baseline 模型和深度模型接口确定。
- [ ] 所有训练、回测、报告输出可复现。

代码进入 paper trading 的验收条件：

- [ ] 测试集交易次数 >= 300。
- [ ] 扣费后 Profit Factor > 1.2。
- [ ] 最大回撤 < 15%。
- [ ] 年化 Sharpe > 1.0。
- [ ] 至少两个不同年份有效。
- [ ] 至少多个币种有效。
- [ ] 所有信号和成交模拟可追溯。

---

## 19. 后续扩展

验证通过后再考虑：

- Meta-labeling 二层过滤模型。
- Perceiver-style latent bottleneck。
- Informer / Linformer 等低复杂度 Transformer。
- 多周期联合模型。
- 强化学习仓位管理。
- 动态 market regime 策略切换。
