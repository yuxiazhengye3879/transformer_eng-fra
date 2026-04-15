# Transformer 英法翻译（点积注意力 vs 加性注意力）

本项目使用 Transformer 完成英文到法文翻译任务，数据来自 `dataset/eng-fra_train_data.txt` 与 `dataset/eng-fra_test_data.txt`。

主程序会连续训练两套模型并自动比较：
- `dot`：缩放点积注意力（Scaled Dot-Product Attention）
- `additive`：加性注意力（Bahdanau 风格）

## 你要求的“准备编写哪些程序文件”

本次实现新增并维护如下程序文件：
- `main.py`：项目主入口，一键启动训练与对比
- `src/data_utils.py`：数据读取、分词、词表、Dataset、collate
- `src/model.py`：Transformer 编码器-解码器与可切换注意力实现
- `src/train_eval.py`：训练、评估、生成对比报告
- `src/__init__.py`：包初始化
- `requirements.txt`：依赖列表
- `README.md`：运行说明与效果对比
- `outputs/comparison.md`：自动生成的对比结果
- `outputs/results_dot.json`、`outputs/results_additive.json`：详细指标与样例
- `outputs/run_summary.json`：一次运行的完整配置与汇总

## 环境与运行

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 运行主程序

```bash
python main.py
```

默认会运行 `dot` 和 `additive` 两种注意力并输出结果到 `outputs/`。
当前默认配置即为全量训练（`train-limit=-1`, `test-limit=-1`）和 `100` 个 epoch。

## 常用参数

```bash
python main.py \
  --epochs 100 \
  --batch-size 64 \
  --d-model 128 \
  --num-heads 4 \
  --num-layers 2 \
  --ff-dim 256 \
  --train-limit -1 \
  --test-limit -1
```

参数说明：
- `--train-limit` / `--test-limit`：`-1` 表示使用全部数据
- 若只做快速调试，可改为较小值（例如 `--train-limit 2000 --test-limit 500`）

## 注意力机制改造说明

在 `src/model.py` 的 `MultiHeadAttention` 中支持两种打分函数：

1. 点积注意力：
$$
\text{score}(q, k) = \frac{qk^T}{\sqrt{d_k}}
$$

2. 加性注意力：
$$
\text{score}(q, k) = v^T \tanh(W_q q + W_k k)
$$

实现方式：通过 `attention_type` 参数切换 `dot` / `additive`，并在编码器自注意力、解码器自注意力、解码器交叉注意力中统一生效。

## 实验对比结果（本地一次实测）

实验命令：

```bash
python main.py --epochs 1 --batch-size 64 --d-model 64 --num-heads 4 --num-layers 1 --ff-dim 128 --train-limit 2000 --test-limit 500
```

指标结果：

| Attention | Train Loss | Eval Loss | Token Acc | BLEU |
|---|---:|---:|---:|---:|
| dot | 7.7457 | 7.2939 | 0.1039 | 0.00 |
| additive | 7.6403 | 7.0702 | 0.1490 | 0.56 |

结论（在该小规模、1 epoch 设置下）：
- 加性注意力的 `Eval Loss` 更低
- 加性注意力的 `Token Acc` 与 `BLEU` 略优于点积注意力
- 由于训练轮次少，翻译样例质量仍较弱，建议增大 `epochs`、`d_model` 并使用更大训练集获得更稳定结论

## 输出文件说明

- `outputs/comparison.md`：可读版对比结果（含翻译样例）
- `outputs/loss_comparison.png`：两种注意力在每个 epoch 的 train/eval loss 同图对比（Matplotlib）
- `outputs/results_dot.json`：点积注意力详细历史指标
- `outputs/results_additive.json`：加性注意力详细历史指标
- `outputs/model_dot.pt`、`outputs/model_additive.pt`：模型参数

## 可视化分析说明

训练结束后脚本会自动生成 `outputs/loss_comparison.png`，图中包含 4 条曲线：
- `dot-train`、`dot-eval`
- `additive-train`、`additive-eval`

通过该图可以直接比较两种注意力的收敛速度与泛化趋势（例如哪一条 `eval loss` 曲线下降更快、更稳定）。

## 提交与推送

如果你已经配置好远程仓库，可用以下命令提交：

```bash
git add .
git commit -m "feat: transformer en-fr translation with additive attention comparison"
git push
```

如果 `git push` 失败，通常是远程地址或认证未配置（PAT/SSH key）。
