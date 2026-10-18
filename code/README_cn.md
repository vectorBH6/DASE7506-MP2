# MP2 — 代码与运行说明

请先阅读 [项目指南](../guide/GUIDE.md)，了解任务说明、示例、研究问题、评分规则、截止日期以及提交要求。

本目录包含完整的数据集、一个空的预测接口以及官方的评测工具。**未提供参考算法、训练流程或训练好的模型权重**。请自行实现你的方法后再运行预测。下列命令均需在 **`code/`** 目录下执行。

## 1. 安装并检查数据

克隆完整的代码仓库，并使用 **Python 3.12**：

```bash
git clone https://github.com/xudongwu-0/DASE7506-MP2.git
cd DASE7506-MP2/code
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python check_data.py
```

或者在 [GitHub](https://github.com/xudongwu-0/DASE7506-MP2) 上点击 **Code → Download ZIP** 下载压缩包，解压后进入其 `code/` 目录，然后运行上述环境安装命令。

在 Windows PowerShell 中，使用 `venv\Scripts\Activate.ps1` 激活虚拟环境。所提供的工具仅需 NumPy 即可运行，请自行添加并锁定你的方法所依赖的库。安装完成后，预测与评分过程必须能够离线运行。

## 2. 开发你的方法

在 `submission.py` 中实现 `predict(bank, arrays, checkpoint)` 函数，或者提供另一个包含该函数的 Python 文件。提供的默认函数为空，会抛出 `NotImplementedError`，需要由你来实现。

- `bank` 是一个按照固定题目顺序排列的列表。每个元素包含 `id`、`question` 和四个 `candidates`；每个候选解包含 `index` 和 `text`。
- `arrays['embeddings']` 是一个 float32 类型的 NumPy 数组，形状为 **[N, 4, 1024]**，与候选解的顺序一一对应。所有方法都可以使用候选文本以及这些固定的表征向量。
- `checkpoint` 是 `pathlib.Path` 类型或 `None`。请保存推理所需的全部权重、变换、统计量与配置；为每道题返回 **0 到 3 之间的一个 Python 整数**（按 bank 中的题目顺序）。
- 请自行编写所需的训练代码。无需训练的“免训练”方法同样具备参赛资格。请仅使用训练输入/参考进行拟合，使用验证集进行开发。题目 ID 仅用于对齐。

实现完方法后，请在验证集上进行评测：

```bash
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 \
python predict.py --method submission.py --checkpoint runs/model_bundle \
  --split validation --output runs/validation-predictions.json
python score_predictions.py --split validation \
  --predictions runs/validation-predictions.json --output runs/validation-score.json
```

`runs/model_bundle` 是由你自己创建的资源，仓库中并未提供。如果你的方法不需要资源文件，可以省略 `--checkpoint`。在 Windows 上，请在执行上述 Python 命令之前先设置线程相关的环境变量。运行脚本必须传入 `--method`，并且**永远不会读取参考答案**。

在最终测试前，请冻结你的方法和配置：

```bash
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 \
python predict.py --method submission.py --checkpoint runs/model_bundle \
  --split test --output runs/test-predictions.json
python score_predictions.py --split test \
  --predictions runs/test-predictions.json --output runs/test-score.json
```

请提交评分器输出中的 **`accuracy`** 字段（以百分比表示）。评分器使用 `common.parse_answer` 检查所选候选解的最终数值答案，并接受等价的分数/小数形式。它只评判最终答案是否正确，而**不评判推理过程是否合理**。预测文件中自行声明的分数会被忽略；缺失或乱序的 ID、无效的索引以及 split/协议不匹配的情况都会被拒绝。

## 3. 数据与文件

| 文件 | 用途 |
|---|---|
| `submission.py` | 空的预测接口，供你实现自己的方法。 |
| `predict.py` | 运行你冻结后的方法；写入预测结果，但不读取参考答案。 |
| `score_predictions.py`、`common.py` | 固定的独立评分器、数值解析工具与数据加载工具。 |
| `check_data.py` | 校验数据哈希、划分集互斥性以及表征/参考的对齐情况。 |
| `data/candidates/gsm8k_{split}.jsonl` | 题目以及四个候选解。 |
| `data/candidates/gsm8k_{split}.npz` | 固定的整条推理链的嵌入向量；不含模型权重或正误标签。 |
| `data/gsm8k_{split}.jsonl` | 对齐后的参考答案：用于训练监督、验证集选择以及仅在测试集上评分。 |
| `data/manifest.json` | 协议、样本数量、数据来源以及数据哈希。 |
| `data/LICENSE.txt`、`data/NOTICE.txt` | 数据源许可证与署名声明。 |
| `PACKAGE_MANIFEST.json` | 发布文件的哈希，相对于包含 `code/` 和 `guide/` 的包根目录。 |

协议：**`dase7506-mp2-open-hard-verifier-v5`**。各划分集的题目数量分别为训练 **4,755**、验证 **798**、测试 **467**，且互不重叠。每一组中都有一个最终正确答案。训练集与验证集包含相对容易的题目；测试集中的题目至少具有两种不同的有效解析答案。候选解的顺序是固定的。测试集参考答案对外公开，仅用于自行报告与同侪复现；**不得用于训练、查找表或方法选择**。

教师可能提供一份不含参考答案的独立审计题库：

```bash
python predict.py --method submission.py --checkpoint runs/model_bundle \
  --split audit --data-dir /path/to/audit_inputs --output runs/audit-predictions.json
```

同一份冻结后的方法必须能够在无需重新训练或硬编码样本数的情况下接受该题库。请使用传给 `predict()` 的 `bank` 与 `arrays`，而不是打开参考答案文件，也不要假设每个输入都是公开的测试集。

## 4. 测量与复现

[项目指南](../guide/GUIDE.md#3-constraints) 中定义了资源限制与测量规则：**30 秒**内完成完整的 CPU 预测与评分（**四线程**），额外的推理文件不超过 **16 MiB**。运行脚本输出的 JSON 时间字段并未包含全部进程启动时间，请使用外部方式测量并记录你的硬件配置。

请锁定你方法所依赖的库版本，并提供精确的安装/训练/评测命令、随机种子以及与之匹配的推理资源。同侪应当能够使用你提交的代码版本与模型权重，按照上述官方命令运行而**无需重新训练**。完整的提交要求与截止日期请参见 [提交与同侪评审规则](../guide/GUIDE.md#4-submission-and-peer-review)。

## 5. 来源与致谢

题目与参考答案源自 [GSM8K](https://github.com/openai/grade-school-math)，沿用其保留的 MIT 许可证。候选解与固定的表征向量由 [Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B) 生成，对应的模型版本记录在数据清单中。仓库中**不分发任何模型权重**。在重新分发本包时，请保留 `data/LICENSE.txt` 与 `data/NOTICE.txt`。