import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader
import sys
import os

from config import Config
from utils.vocab import load_vocab, NewsDataset
from models.textcnn import TextCNN
from models.bilstm import BiLSTM
from models.bert import BertClassifier
from transformers import BertTokenizer

# 设置中文字体（可根据系统调整）
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'KaiTi']  # 多个备选
plt.rcParams['axes.unicode_minus'] = False

def load_model_and_data(config, model_name):
    config.model_name = model_name
    with open(config.class_names_path, 'r', encoding='utf-8') as f:
        class_names = [line.strip() for line in f.readlines()]
    config.num_classes = len(class_names)

    test_df = pd.read_csv(config.test_path)

    if model_name == 'bert':
        tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
        class BertDataset(torch.utils.data.Dataset):
            def __init__(self, df, tokenizer, max_len=256):
                self.labels = df['label'].values
                self.input_ids = []
                self.attention_masks = []
                from tqdm import tqdm
                for text in tqdm(df['text'], desc="预处理BERT数据"):
                    encoding = tokenizer.encode_plus(
                        text,
                        add_special_tokens=True,
                        max_length=max_len,
                        padding='max_length',
                        truncation=True,
                        return_tensors='pt'
                    )
                    self.input_ids.append(encoding['input_ids'].squeeze(0))
                    self.attention_masks.append(encoding['attention_mask'].squeeze(0))
            def __len__(self):
                return len(self.labels)
            def __getitem__(self, idx):
                return self.input_ids[idx], self.attention_masks[idx], torch.tensor(self.labels[idx])
        test_dataset = BertDataset(test_df, tokenizer)
        test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)
        model = BertClassifier(num_classes=config.num_classes).to(config.device)
        model.load_state_dict(torch.load(f'best_model_{model_name}.pth', map_location=config.device, weights_only=True))
        model.eval()
        return model, test_loader, class_names, test_df['label'].values
    else:
        vocab = load_vocab(config.vocab_path)
        test_dataset = NewsDataset(test_df, vocab, config.max_len)
        test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)
        if model_name == 'textcnn':
            model = TextCNN(vocab.vocab_size, config.embed_size, config.num_classes).to(config.device)
        elif model_name == 'bilstm':
            model = BiLSTM(vocab.vocab_size, config.embed_size, num_classes=config.num_classes).to(config.device)
        model.load_state_dict(torch.load(f'best_model_{model_name}.pth', map_location=config.device, weights_only=True))
        model.eval()
        return model, test_loader, class_names, test_df['label'].values

def evaluate_model(config, model_name):
    model, test_loader, class_names, y_true = load_model_and_data(config, model_name)
    all_preds = []
    with torch.no_grad():
        for batch in test_loader:
            if model_name == 'bert':
                input_ids, attention_mask, labels = [x.to(config.device) for x in batch]
                logits = model(input_ids, attention_mask)
            else:
                inputs, labels = [x.to(config.device) for x in batch]
                logits = model(inputs)
            preds = logits.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
    y_pred = np.array(all_preds)

    # 输出分类报告
    print(classification_report(y_true, y_pred, target_names=class_names))

    # 绘制并保存混淆矩阵
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(14, 12))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('预测类别')
    plt.ylabel('真实类别')
    plt.title(f'混淆矩阵 - {model_name}')
    # 方法一：显示（如果字体设置正确，中文会正常显示）
    plt.show()
    # 方法二：同时保存到文件（可选）
    plt.savefig(f'confusion_matrix_{model_name}.png', dpi=300, bbox_inches='tight')
    print(f"混淆矩阵已保存为 confusion_matrix_{model_name}.png")

def main(model_name):
    config = Config()
    evaluate_model(config, model_name)

if __name__ == '__main__':
    if len(sys.argv) > 1:
        model_name = sys.argv[1]
    else:
        model_name = 'textcnn'
    main(model_name)