import numpy as np
from sklearn.utils.class_weight import compute_class_weight
import torch

def get_class_weights(labels, num_classes):
    """计算类别权重，用于加权损失函数"""
    class_weights = compute_class_weight('balanced', classes=np.arange(num_classes), y=labels)
    return torch.tensor(class_weights, dtype=torch.float)

def compute_class_distribution(df):
    """输出每个类别的样本数量"""
    dist = df['label'].value_counts().sort_index()
    return dist