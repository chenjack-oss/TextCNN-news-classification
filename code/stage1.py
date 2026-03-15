# -*- coding: utf-8 -*-
"""
第一阶段：数据加载 + 按类别抽样
数据集路径：D:\001BS\111\THUCNews
输出：D:\001BS\data\news_data_sampled.csv
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ==================== 设置中文字体 ====================
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ==================== 配置路径 ====================
data_root = r"D:\001BS\111\THUCNews"
output_dir = r"D:\001BS\data"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# ==================== 抽样参数 ====================
SAMPLE_PER_CLASS = 3000   # 每个类别保留的样本数
USE_SAMPLE = True         # 启用抽样

# ==================== 1. 加载数据 ====================
print("正在加载数据...")
data = []
labels = []

categories = [d for d in os.listdir(data_root) 
              if os.path.isdir(os.path.join(data_root, d))]
print(f"发现类别: {categories}")

for cat in categories:
    cat_path = os.path.join(data_root, cat)
    files = [f for f in os.listdir(cat_path) if f.endswith('.txt')]
    for file in files:
        file_path = os.path.join(cat_path, file)
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read().strip()
                if text:
                    data.append(text)
                    labels.append(cat)
        except UnicodeDecodeError:
            try:
                with open(file_path, 'r', encoding='gbk', errors='ignore') as f:
                    text = f.read().strip()
                    if text:
                        data.append(text)
                        labels.append(cat)
            except Exception as e:
                print(f"读取文件失败: {file_path}, 错误: {e}")

df = pd.DataFrame({'text': data, 'label': labels})
print(f"数据加载完成，总样本数: {len(df)}")

# ==================== 2. 按类别抽样 ====================
if USE_SAMPLE:
    print(f"\n按类别抽样，每类 {SAMPLE_PER_CLASS} 条...")
    df_sampled = df.groupby('label', group_keys=False).apply(
        lambda x: x.sample(n=min(SAMPLE_PER_CLASS, len(x)), random_state=42)
    ).reset_index(drop=True)
    print(f"抽样完成，总样本数: {len(df_sampled)}")
    df = df_sampled

# ==================== 3. 查看抽样后分布 ====================
print("\n抽样后类别分布:")
print(df['label'].value_counts())

# 绘制类别分布图
plt.figure(figsize=(14, 8))
ax = sns.countplot(y='label', data=df, order=df['label'].value_counts().index)
plt.title('抽样后各类别新闻数量', fontsize=16)
plt.xlabel('数量', fontsize=14)
plt.ylabel('类别', fontsize=14)
for p in ax.patches:
    width = p.get_width()
    plt.text(width + 0.5, p.get_y() + p.get_height()/2, f'{int(width)}', 
             ha='left', va='center', fontsize=10)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'category_distribution_sampled.png'), dpi=300)
plt.show()

# ==================== 4. 文本长度分析 ====================
df['char_len'] = df['text'].apply(len)
print("\n文本长度统计:")
print(df['char_len'].describe())

plt.figure(figsize=(14, 6))
plt.hist(df['char_len'], bins=50, alpha=0.7, edgecolor='black')
plt.xlabel('字符数', fontsize=14)
plt.ylabel('频数', fontsize=14)
plt.title('抽样后文本长度分布', fontsize=16)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'text_length_dist_sampled.png'), dpi=300)
plt.show()

# 查看分位数，确认1500覆盖比例
quantiles = [0.5, 0.75, 0.9, 0.95, 0.99]
for q in quantiles:
    print(f"{q*100:.0f}% 分位数: {df['char_len'].quantile(q):.0f}")

# ==================== 5. 保存抽样数据 ====================
output_csv = os.path.join(output_dir, 'news_data_sampled.csv')
df.to_csv(output_csv, index=False, encoding='utf-8-sig')
print(f"\n抽样数据已保存至: {output_csv}")
print("\n第一阶段完成！")