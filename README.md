# TextCNN-news-classification

基于深度学习的**中文新闻文本分类**系统（THUCNews 14 类），提供 **TextCNN / BiLSTM / BERT** 三模型对比、**长尾类别加权**、传统方法基线以及 **REST API 部署**能力。

---

## 项目简介

本项目以 THUCNews 中文新闻语料为数据源，构建了一套从数据准备、模型训练、评估可视化到线上服务的完整分类流水线：

- **多模型对比**：TextCNN（多尺度卷积）、BiLSTM（双向时序建模）、BERT（预训练微调）统一训练/评估接口，通过 `--model` 一键切换；
- **长尾处理**：基于 `sklearn` 平衡权重对 CrossEntropyLoss 加权，缓解类别样本不均衡；
- **基线对照**：内置 TF-IDF + 逻辑回归传统方法基准，用于量化深度模型的增益；
- **服务化**：Flask 暴露 `/predict` 接口，返回类别、标签与全类别概率分布。

数据共 **14 个类别**：体育、娱乐、家居、彩票、房产、教育、时尚、时政、星座、游戏、社会、科技、股票、财经。

---

## 项目结构

```
TextCNN-news-classification/
├── config.py                  # 全局配置（路径、超参、模型选择）
├── run.py                     # 统一入口：prepare / train / evaluate / benchmark / deploy
├── prepare_data.py            # 数据加载与划分（独立脚本）
├── train.py                   # 训练主流程（含最佳模型保存、耗时预估）
├── evaluate.py                # 评估脚本（分类报告 + 混淆矩阵）
├── benchmark.py               # 传统方法基线（jieba + TF-IDF + LogisticRegression）
├── deploy.py                  # Flask REST API 服务
├── models/                    # 模型定义
│   ├── textcnn.py             # TextCNN
│   ├── bilstm.py              # BiLSTM
│   └── bert.py                # BERT 分类头
├── utils/                     # 工具函数
│   ├── data_loader.py         # THUCNews 读取 + 分层划分
│   ├── vocab.py               # 词表构建 / 编码 / Dataset
│   └── longtail.py            # 类别权重计算与分布统计
├── code/                      # 早期四阶段探索脚本（TensorFlow/Keras 版本）
│   ├── stage1.py              # 数据加载 + 按类抽样 + 分布/长度可视化
│   ├── stage2.py              # 预处理（词频统计、词表、内存映射、标签编码）
│   ├── stage3.py              # Keras TextCNN 训练
│   └── stage4.py              # 单条文本预测
├── data/
│   ├── processed/             # 划分后的数据集与类别名
│   └── vocab/vocab.pkl        # 持久化词表
└── requirements.txt
```

> `code/` 目录保留了项目早期的 Keras 原型实现，主流程已重构为 `models/` + `utils/` 的模块化 PyTorch 版本。

---

## 环境要求

- Python 3.8+
- PyTorch（推荐 CUDA 版本，`config.py` 会自动检测可用设备）
- 依赖安装：

```bash
pip install -r requirements.txt
```

主要依赖：`torch`、`transformers`、`scikit-learn`、`pandas`、`numpy`、`flask`、`jieba`、`matplotlib`、`seaborn`、`tqdm`。

---

## 快速开始

### 1. 克隆仓库

```bash
git clone https://github.com/chenjack-oss/TextCNN-news-classification.git
cd TextCNN-news-classification
```

### 2. 修改数据路径

THUCNews 语料需按「类别名 / 文本文件」的目录结构组织，并在 `config.py` 中指向你的本地路径：

```python
class Config:
    raw_data_dir = 'D:/001BS/111/THUCNews'   # ← 改为你的 THUCNews 根目录
```

> 目录结构示例：`THUCNews/体育/*.txt`、`THUCNews/财经/*.txt` ……
> 程序会自动将子目录名作为类别名，并按名称排序保证标签顺序稳定。

### 3. 数据准备

```bash
python run.py --mode prepare
# 或
python prepare_data.py
```

按 **训练 8 : 验证 1 : 测试 1** 的比例分层划分（`stratify=label`，保持类别比例），输出至 `data/processed/`。

### 4. 训练

```bash
python run.py --mode train --model textcnn
python run.py --mode train --model bilstm
python run.py --mode train --model bert
```

训练过程输出每个 epoch 的 Train/Val Loss 与准确率，并自动保存验证集表现最佳的权重为 `best_model_{model_name}.pth`。

### 5. 评估

```bash
python run.py --mode evaluate --model textcnn
```

输出 `classification_report`（各类别 precision / recall / f1），并绘制混淆矩阵保存为 `confusion_matrix_{model_name}.png`。

### 6. 传统方法基线

```bash
python run.py --mode benchmark
```

使用 jieba 分词 + TF-IDF（`max_features=50000, min_df=2`）+ 逻辑回归（`class_weight='balanced'`），输出同一测试集上的分类报告作为对照。

### 7. 启动服务

```bash
python run.py --mode deploy
```

服务默认监听 `http://0.0.0.0:5000`，提供 `POST /predict`：

```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "昨晚的比赛中，主队凭借下半场的进球逆转取胜"}'
```

响应示例：

```json
{
  "category": "体育",
  "label": 0,
  "probabilities": [0.98, 0.001, "..."]
}
```

---

## 模型说明

| 模型 | 结构要点 | 适用场景 |
|---|---|---|
| **TextCNN** | `Embedding(300)` → 多尺度 `Conv1d`（kernel 3/4/5，各 100 filters）→ ReLU → `MaxPool1d` → 拼接 → `Dropout(0.5)` → FC | 训练快、局部 n-gram 特征敏感，作为主力基线 |
| **BiLSTM** | `Embedding(300)` → 2 层双向 LSTM（hidden 256，`batch_first`）→ 末层双向隐状态拼接 → `Dropout(0.5)` → FC | 长距离依赖与语序建模 |
| **BERT** | `bert-base-chinese` → `pooler_output` → `Dropout(0.3)` → FC | 语义理解最强，代价是显存与训练时间 |

关键超参（可在 `config.py` 调整）：`batch_size=64`、`epochs=10`、`learning_rate=1e-3`、`max_len=300`（BERT 分支为 256）、`embed_size=300`。

---

## 关键实现说明

**长尾类别加权**（`utils/longtail.py` + `train.py`）

```python
class_weights = compute_class_weight('balanced', classes=np.arange(num_classes), y=labels)
criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_weights))
```

通过 `Config.use_weighted_loss` 开关控制；BERT 分支默认使用未加权损失。

**词表与编码**（`utils/vocab.py`）

- 基于 jieba 分词 + `Counter` 统计词频，按频率保留 `max_size=50000`、`min_freq=2` 的词；
- 特殊符号：`<PAD>=0`、`<UNK>=1`；
- 编码时截断/补齐至 `max_len`，词表以 pickle 持久化，避免重复分词。

**BERT 分支的独立数据管线**

BERT 不走词表，改用 `BertTokenizer` 在线编码（`add_special_tokens` + `padding='max_length'` + `truncation`），因此 `train.py` / `evaluate.py` / `deploy.py` 中均按 `model_name == 'bert'` 分支处理三元组 `(input_ids, attention_mask, labels)`。

---

## 已知注意事项

- `config.py` 中的 `raw_data_dir` 为本地绝对路径，**首次运行前必须修改**；
- 词表构建与 TF-IDF 基线需要对全量语料分词，首次运行耗时较长（脚本内置 `tqdm` 进度条）；
- 混淆矩阵绘图使用 `SimHei` 中文字体，Linux 环境若缺字体需自行配置 `plt.rcParams['font.sans-serif']`。

---

## License

本项目仅用于学习与技术交流。
