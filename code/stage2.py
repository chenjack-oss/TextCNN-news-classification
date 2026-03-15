# -*- coding: utf-8 -*-
"""
第二阶段：数据预处理（抽样数据版）
输入：D:\001BS\data\news_data_sampled.csv
输出：D:\001BS\data\...（包括内存映射文件、标签、词汇表等）
"""

import os
import json
import pickle
import numpy as np
import pandas as pd
import jieba
from collections import Counter
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# ==================== 配置路径 ====================
input_path = r"D:\001BS\data\news_data_sampled.csv"   # 抽样数据文件
output_dir = r"D:\001BS\data"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# ==================== 参数设置 ====================
MAX_LEN = 1500               # 序列长度（缩短后）
MAX_VOCAB_SIZE = 50000       # 最大词汇表大小
MIN_WORD_FREQ = 5            # 最小词频（低于此值的词视为<UNK>）
TEST_SIZE = 0.1              # 测试集比例
VAL_SIZE = 0.1               # 验证集比例
RANDOM_STATE = 42            # 随机种子
CHUNK_SIZE = 50000           # 每次读取的CSV块大小（根据内存可调整）

print("=" * 50)
print("第二阶段：数据预处理开始（抽样数据版）")
print("=" * 50)

# ==================== 第一步：统计词频 ====================
print("\n第一步：统计词频...")
word_counter = Counter()
total_lines = 0

# 获取总行数（不包括表头）
with open(input_path, 'r', encoding='utf-8') as f:
    total_lines = sum(1 for _ in f) - 1
print(f"CSV文件总样本数: {total_lines}")

# 分块读取并分词，统计词频
for chunk in tqdm(pd.read_csv(input_path, chunksize=CHUNK_SIZE, encoding='utf-8-sig'), desc="词频统计"):
    texts = chunk['text'].fillna('').astype(str)
    for text in texts:
        words = jieba.lcut(text.strip())
        # 过滤掉单字和空白
        words = [w for w in words if len(w) > 1 and w.strip()]
        word_counter.update(words)

print(f"总词数（去重前）: {sum(word_counter.values())}")
print(f"唯一词数: {len(word_counter)}")

# 构建词汇表（取最常见的 MAX_VOCAB_SIZE 个词）
vocab_words = [word for word, count in word_counter.most_common(MAX_VOCAB_SIZE) if count >= MIN_WORD_FREQ]
print(f"词汇表大小（过滤后）: {len(vocab_words)}")

word_index = {word: idx+2 for idx, word in enumerate(vocab_words)}
word_index['<PAD>'] = 0   # 填充符
word_index['<UNK>'] = 1   # 未知词

# 保存词汇表
with open(os.path.join(output_dir, 'word_index.json'), 'w', encoding='utf-8') as f:
    json.dump(word_index, f, ensure_ascii=False, indent=2)
print("词汇表已保存至 word_index.json")

# ==================== 第二步：创建内存映射文件 ====================
print("\n第二步：创建内存映射文件...")
X_path = os.path.join(output_dir, 'X_data.mmap')
# 使用 uint16 节省内存（词汇表索引最大不超过50002，在uint16范围内）
X_mmap = np.memmap(X_path, dtype=np.uint16, mode='w+', shape=(total_lines, MAX_LEN))

# 存储标签（暂时放入列表）
y_list = []

# ==================== 第三步：序列化写入 ====================
print("\n第三步：序列化写入...")
current_row = 0
for chunk in tqdm(pd.read_csv(input_path, chunksize=CHUNK_SIZE, encoding='utf-8-sig'), desc="序列化"):
    texts = chunk['text'].fillna('').astype(str)
    labels = chunk['label'].fillna('').astype(str)
    batch_size = len(texts)
    
    for i, text in enumerate(texts):
        # 分词
        words = jieba.lcut(text.strip())
        words = [w for w in words if len(w) > 1 and w.strip()]
        # 转索引
        seq = [word_index.get(w, word_index['<UNK>']) for w in words]
        # 截断/填充到固定长度
        if len(seq) > MAX_LEN:
            seq = seq[:MAX_LEN]
        else:
            seq = seq + [word_index['<PAD>']] * (MAX_LEN - len(seq))
        # 写入内存映射文件
        X_mmap[current_row + i] = seq
        # 记录标签
        y_list.append(labels.iloc[i])
    
    current_row += batch_size

X_mmap.flush()   # 确保写入磁盘
print(f"序列化完成，实际处理的样本数: {current_row}")

# ==================== 第四步：标签编码 ====================
print("\n第四步：标签编码...")
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y_list)
print("标签类别映射:")
for i, cls in enumerate(label_encoder.classes_):
    print(f"  {i}: {cls}")

# 保存标签编码器
with open(os.path.join(output_dir, 'label_encoder.pkl'), 'wb') as f:
    pickle.dump(label_encoder, f)
# 保存标签数组
np.save(os.path.join(output_dir, 'y_data.npy'), y)
print("标签数据已保存至 y_data.npy 和 label_encoder.pkl")

# ==================== 第五步：数据集划分 ====================
print("\n第五步：划分数据集...")
actual_samples = len(y)   # 实际处理的样本数
indices = np.arange(actual_samples)

# 先划分训练+验证 和 测试集
idx_temp, idx_test, y_temp, y_test = train_test_split(
    indices, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
)
# 再从剩余中划分训练和验证集
idx_train, idx_val, y_train, y_val = train_test_split(
    idx_temp, y_temp, test_size=VAL_SIZE/(1-TEST_SIZE),
    stratify=y_temp, random_state=RANDOM_STATE
)
print(f"训练集: {len(idx_train)} 条")
print(f"验证集: {len(idx_val)} 条")
print(f"测试集: {len(idx_test)} 条")

# 保存划分索引
np.savez_compressed(
    os.path.join(output_dir, 'split_indices.npz'),
    train_idx=idx_train, val_idx=idx_val, test_idx=idx_test,
    train_y=y_train, val_y=y_val, test_y=y_test
)
print("划分索引已保存至 split_indices.npz")

# ==================== 第六步：配置参数 ====================
config = {
    'max_len': MAX_LEN,
    'vocab_size': len(word_index),
    'num_classes': len(label_encoder.classes_),
    'classes': label_encoder.classes_.tolist(),
    'train_size': len(idx_train),
    'val_size': len(idx_val),
    'test_size': len(idx_test)
}
with open(os.path.join(output_dir, 'config.json'), 'w', encoding='utf-8') as f:
    json.dump(config, f, ensure_ascii=False, indent=2)
print("配置参数已保存至 config.json")

print("\n第二阶段数据预处理完成！")
print("生成的文件列表：")
print("  - X_data.mmap      (序列化数据，内存映射文件)")
print("  - y_data.npy       (标签数组)")
print("  - word_index.json  (词汇表映射)")
print("  - label_encoder.pkl (标签编码器)")
print("  - split_indices.npz (训练/验证/测试划分索引)")
print("  - config.json      (配置参数)")
print("=" * 50)