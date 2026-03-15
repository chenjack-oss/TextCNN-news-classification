# -*- coding: utf-8 -*-
"""
第三阶段：构建并训练 TextCNN 模型
输入：D:\001BS\data\ 下的 X_data.mmap, y_data.npy, split_indices.npz, config.json, label_encoder.pkl
输出：训练好的模型、训练曲线图、混淆矩阵图、分类报告
"""

import os
import json
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# 设置中文字体（防止绘图乱码）
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ==================== 配置路径 ====================
data_dir = r"D:\001BS\data"
X_path = os.path.join(data_dir, 'X_data.mmap')
y_path = os.path.join(data_dir, 'y_data.npy')
indices_path = os.path.join(data_dir, 'split_indices.npz')
config_path = os.path.join(data_dir, 'config.json')
label_encoder_path = os.path.join(data_dir, 'label_encoder.pkl')
model_save_path = os.path.join(data_dir, 'textcnn_model.h5')

# ==================== 加载配置和数据 ====================
with open(config_path, 'r', encoding='utf-8') as f:
    config = json.load(f)

max_len = config['max_len']
vocab_size = config['vocab_size']
num_classes = config['num_classes']
class_names = config['classes']

print("=" * 50)
print("第三阶段：模型训练开始")
print("=" * 50)
print("配置信息:")
print(f"  序列长度: {max_len}")
print(f"  词汇表大小: {vocab_size}")
print(f"  类别数: {num_classes}")
print(f"  类别: {class_names}")

# 加载标签和划分索引
y = np.load(y_path)
indices = np.load(indices_path)
train_idx = indices['train_idx']
val_idx = indices['val_idx']
test_idx = indices['test_idx']
train_y = indices['train_y']
val_y = indices['val_y']
test_y = indices['test_y']

print(f"训练集: {len(train_idx)} 条")
print(f"验证集: {len(val_idx)} 条")
print(f"测试集: {len(test_idx)} 条")

# 打开内存映射文件（只读模式）
X_mmap = np.memmap(X_path, dtype=np.uint16, mode='r', shape=(len(y), max_len))

# ==================== 定义数据生成器 ====================
def data_generator(indices, labels, batch_size):
    """生成器，每次产生一个batch，并打乱数据"""
    num_samples = len(indices)
    while True:
        # 每个 epoch 重新打乱索引
        perm = np.random.permutation(num_samples)
        for i in range(0, num_samples, batch_size):
            batch_indices = perm[i:i+batch_size]
            batch_x = X_mmap[indices[batch_indices]]  # 直接从 mmap 读取
            batch_y = labels[batch_indices]
            # 转换为 one-hot 编码
            batch_y = tf.keras.utils.to_categorical(batch_y, num_classes)
            yield batch_x, batch_y

# ==================== 构建 TextCNN 模型 ====================
def build_textcnn(vocab_size, max_len, num_classes, embedding_dim=128, num_filters=64, kernel_sizes=[3,4,5]):
    inputs = layers.Input(shape=(max_len,), dtype='int32')
    # 嵌入层
    embedding = layers.Embedding(vocab_size, embedding_dim, input_length=max_len)(inputs)
    embedding = layers.Dropout(0.2)(embedding)
    
    # 多个卷积池化层
    conv_pools = []
    for kernel_size in kernel_sizes:
        conv = layers.Conv1D(filters=num_filters, kernel_size=kernel_size, activation='relu')(embedding)
        pool = layers.GlobalMaxPooling1D()(conv)
        conv_pools.append(pool)
    
    # 拼接
    concat = layers.concatenate(conv_pools, axis=-1)
    concat = layers.Dropout(0.5)(concat)
    
    # 全连接层
    dense = layers.Dense(128, activation='relu')(concat)
    dense = layers.Dropout(0.5)(dense)
    outputs = layers.Dense(num_classes, activation='softmax')(dense)
    
    model = models.Model(inputs=inputs, outputs=outputs)
    return model

model = build_textcnn(vocab_size, max_len, num_classes)
model.summary()

# ==================== 编译模型 ====================
model.compile(optimizer=optimizers.Adam(1e-3),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# ==================== 回调函数 ====================
early_stop = callbacks.EarlyStopping(monitor='val_accuracy', patience=3, restore_best_weights=True)
reduce_lr = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2)
model_checkpoint = callbacks.ModelCheckpoint(model_save_path, monitor='val_accuracy', save_best_only=True)

# ==================== 训练 ====================
batch_size = 64
steps_per_epoch = len(train_idx) // batch_size
validation_steps = len(val_idx) // batch_size

print("\n开始训练...")
history = model.fit(
    data_generator(train_idx, train_y, batch_size),
    steps_per_epoch=steps_per_epoch,
    epochs=10,
    validation_data=data_generator(val_idx, val_y, batch_size),
    validation_steps=validation_steps,
    callbacks=[early_stop, reduce_lr, model_checkpoint],
    verbose=1
)

# ==================== 绘制训练曲线 ====================
plt.figure(figsize=(12,4))
plt.subplot(1,2,1)
plt.plot(history.history['accuracy'], label='train_acc')
plt.plot(history.history['val_accuracy'], label='val_acc')
plt.title('Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1,2,2)
plt.plot(history.history['loss'], label='train_loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.title('Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(data_dir, 'training_curves.png'), dpi=300)
plt.show()

# ==================== 测试集评估 ====================
print("\n在测试集上评估...")
# 由于测试集可能不能整除 batch_size，我们采用分批预测
y_pred = []
y_true = []
for i in range(0, len(test_idx), batch_size):
    batch_indices = test_idx[i:i+batch_size]
    batch_x = X_mmap[batch_indices]
    batch_y_true = test_y[i:i+batch_size]
    batch_y_pred = model.predict(batch_x, verbose=0)
    y_pred.extend(np.argmax(batch_y_pred, axis=1))
    y_true.extend(batch_y_true)

# 计算准确率
accuracy = np.mean(np.array(y_pred) == np.array(y_true))
print(f"测试集准确率: {accuracy:.4f}")

# 分类报告
print("\n分类报告:")
report = classification_report(y_true, y_pred, target_names=class_names)
print(report)
# 保存分类报告到文本文件
with open(os.path.join(data_dir, 'classification_report.txt'), 'w', encoding='utf-8') as f:
    f.write(report)

# 混淆矩阵
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(12,10))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=class_names, yticklabels=class_names, cmap='Blues')
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.savefig(os.path.join(data_dir, 'confusion_matrix.png'), dpi=300)
plt.show()

print(f"\n最佳模型已保存至: {model_save_path}")
print("第三阶段完成！")
print("生成的文件：")
print("  - training_curves.png")
print("  - confusion_matrix.png")
print("  - classification_report.txt")
print("  - textcnn_model.h5")
print("=" * 50)