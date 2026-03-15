# -*- coding: utf-8 -*-
"""
第四阶段：使用训练好的模型对新的新闻文本进行分类预测
"""

import os
import json
import pickle
import numpy as np
import jieba
import tensorflow as tf
from tensorflow.keras.models import load_model

# ==================== 配置路径 ====================
data_dir = r"D:\001BS\data"
model_path = os.path.join(data_dir, 'textcnn_model.h5')
word_index_path = os.path.join(data_dir, 'word_index.json')
label_encoder_path = os.path.join(data_dir, 'label_encoder.pkl')
config_path = os.path.join(data_dir, 'config.json')

# 加载配置
with open(config_path, 'r', encoding='utf-8') as f:
    config = json.load(f)
max_len = config['max_len']

# 加载词汇表
with open(word_index_path, 'r', encoding='utf-8') as f:
    word_index = json.load(f)

# 加载标签编码器
with open(label_encoder_path, 'rb') as f:
    label_encoder = pickle.load(f)

# 加载模型
model = load_model(model_path)
print("模型加载成功！")

# ==================== 预测函数 ====================
def predict_news(text):
    """
    输入一条新闻文本，返回预测的类别和置信度
    """
    # 分词
    words = jieba.lcut(text.strip())
    words = [w for w in words if len(w) > 1 and w.strip()]
    # 转索引
    seq = [word_index.get(w, word_index['<UNK>']) for w in words]
    # 截断/填充
    if len(seq) > max_len:
        seq = seq[:max_len]
    else:
        seq = seq + [word_index['<PAD>']] * (max_len - len(seq))
    # 转为模型输入格式
    input_array = np.array([seq], dtype=np.int32)
    # 预测
    pred_proba = model.predict(input_array, verbose=0)[0]
    pred_class = np.argmax(pred_proba)
    confidence = pred_proba[pred_class]
    class_name = label_encoder.classes_[pred_class]
    return class_name, confidence, pred_proba

# ==================== 交互式预测 ====================
print("\n请输入新闻文本进行预测（输入 'exit' 退出）：")
while True:
    text = input("\n新闻文本: ").strip()
    if text.lower() == 'exit':
        break
    if not text:
        continue
    try:
        cls, conf, proba = predict_news(text)
        print(f"预测类别: {cls}")
        print(f"置信度: {conf:.4f}")
        # 可选：显示所有类别概率
        show_all = input("显示所有类别的概率？(y/n): ").strip().lower()
        if show_all == 'y':
            for i, p in enumerate(proba):
                print(f"  {label_encoder.classes_[i]}: {p:.4f}")
    except Exception as e:
        print(f"预测出错: {e}")