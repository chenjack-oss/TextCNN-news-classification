import os
from utils.data_loader import load_thucnews, split_data
from config import Config

config = Config()
df, class_names = load_thucnews(config.raw_data_dir)
print(f"总样本数: {len(df)}")
print("类别:", class_names)
train_df, val_df, test_df = split_data(df)

# 创建保存目录
os.makedirs('data/processed', exist_ok=True)

train_df.to_csv(config.train_path, index=False)
val_df.to_csv(config.val_path, index=False)
test_df.to_csv(config.test_path, index=False)

with open(config.class_names_path, 'w', encoding='utf-8') as f:
    for name in class_names:
        f.write(name + '\n')

print("数据准备完成！")
print(f"训练集样本数: {len(train_df)}")
print(f"验证集样本数: {len(val_df)}")
print(f"测试集样本数: {len(test_df)}")