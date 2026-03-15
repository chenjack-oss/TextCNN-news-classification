import os
import pandas as pd
from sklearn.model_selection import train_test_split

def load_thucnews(data_dir):
    """
    从THUCNews文件夹读取数据，自动识别子文件夹作为类别
    返回: (df, class_names) 其中 df 包含 'text' 和 'label' 列，class_names 为类别名称列表
    """
    texts = []
    labels = []
    class_names = []   # 用于存储类别名称，按照文件夹名排序

    # 获取所有子文件夹并排序，保证每次运行标签顺序一致
    subdirs = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])
    for label, cat in enumerate(subdirs):
        class_names.append(cat)
        cat_dir = os.path.join(data_dir, cat)
        files = os.listdir(cat_dir)
        for filename in files:
            filepath = os.path.join(cat_dir, filename)
            with open(filepath, 'r', encoding='utf-8') as f:
                text = f.read().strip()
                if text:  # 忽略空文件
                    texts.append(text)
                    labels.append(label)
    df = pd.DataFrame({'text': texts, 'label': labels})
    return df, class_names

def split_data(df, test_size=0.2, val_size=0.1, random_state=42):
    """划分训练集、验证集、测试集，保持类别比例"""
    train_df, test_df = train_test_split(df, test_size=test_size, random_state=random_state, stratify=df['label'])
    # 从训练集中再划分验证集，比例 = val_size/(1-test_size)
    train_df, val_df = train_test_split(train_df, test_size=val_size/(1-test_size), random_state=random_state, stratify=train_df['label'])
    return train_df, val_df, test_df