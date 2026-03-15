import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import pandas as pd
import os
import time
import numpy as np
from transformers import BertTokenizer

from config import Config
from utils.data_loader import load_thucnews, split_data
from utils.vocab import Vocab, save_vocab, load_vocab, NewsDataset
from utils.longtail import get_class_weights, compute_class_distribution
from models.textcnn import TextCNN
from models.bilstm import BiLSTM
from models.bert import BertClassifier

def get_model(config, vocab_size):
    if config.model_name == 'textcnn':
        return TextCNN(vocab_size, config.embed_size, config.num_classes).to(config.device)
    elif config.model_name == 'bilstm':
        return BiLSTM(vocab_size, config.embed_size, num_classes=config.num_classes).to(config.device)
    elif config.model_name == 'bert':
        return BertClassifier(num_classes=config.num_classes).to(config.device)
    else:
        raise ValueError("Unsupported model")

def train_one_epoch(model, train_loader, optimizer, criterion, config, epoch, total_epochs):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    start_time = time.time()
    
    # 使用 tqdm 显示当前epoch的进度
    pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{total_epochs} [Train]')
    for batch in pbar:
        if config.model_name == 'bert':
            input_ids, attention_mask, labels = [x.to(config.device) for x in batch]
            optimizer.zero_grad()
            logits = model(input_ids, attention_mask)
        else:
            inputs, labels = [x.to(config.device) for x in batch]
            optimizer.zero_grad()
            logits = model(inputs)
        
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        
        # 更新进度条显示信息
        pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{correct/total:.4f}'})
    
    epoch_time = time.time() - start_time
    return total_loss / len(train_loader), correct / total, epoch_time

def evaluate(model, val_loader, criterion, config):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        pbar = tqdm(val_loader, desc='Evaluating')
        for batch in pbar:
            if config.model_name == 'bert':
                input_ids, attention_mask, labels = [x.to(config.device) for x in batch]
                logits = model(input_ids, attention_mask)
            else:
                inputs, labels = [x.to(config.device) for x in batch]
                logits = model(inputs)
            loss = criterion(logits, labels)
            total_loss += loss.item()
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    return total_loss / len(val_loader), correct / total

def main():
    config = Config()
    print("="*60)
    print(f"开始训练模型: {config.model_name}")
    print(f"使用设备: {config.device}")
    print("="*60)
    overall_start = time.time()

    # ---------- 1. 加载或准备数据 ----------
    print("\n[1/6] 加载原始数据...")
    df, class_names = load_thucnews(config.raw_data_dir)
    config.num_classes = len(class_names)
    print(f"检测到 {config.num_classes} 个类别: {class_names}")
    
    # 如果已存在划分文件，则直接读取
    if os.path.exists(config.train_path) and os.path.exists(config.val_path) and os.path.exists(config.test_path):
        print("加载已划分的数据集...")
        train_df = pd.read_csv(config.train_path)
        val_df = pd.read_csv(config.val_path)
        test_df = pd.read_csv(config.test_path)
    else:
        print("正在划分数据集...")
        train_df, val_df, test_df = split_data(df)
        os.makedirs('data/processed', exist_ok=True)
        train_df.to_csv(config.train_path, index=False)
        val_df.to_csv(config.val_path, index=False)
        test_df.to_csv(config.test_path, index=False)
        with open(config.class_names_path, 'w', encoding='utf-8') as f:
            for name in class_names:
                f.write(name + '\n')

    print(f"训练集: {len(train_df)} 样本, 验证集: {len(val_df)} 样本, 测试集: {len(test_df)} 样本")

    # ---------- 2. 构建词汇表 / tokenizer ----------
    print("\n[2/6] 准备词汇表/分词器...")
    if config.model_name != 'bert':
        if os.path.exists(config.vocab_path):
            print("加载已有词汇表...")
            vocab = load_vocab(config.vocab_path)
        else:
            print("构建词汇表（此过程需要分词，可能几分钟）...")
            vocab = Vocab(max_size=50000, min_freq=2)
            vocab.build_vocab(train_df['text'].tolist())
            os.makedirs('data/vocab', exist_ok=True)
            save_vocab(vocab, config.vocab_path)
            print("词汇表构建完成！")
        train_dataset = NewsDataset(train_df, vocab, config.max_len)
        val_dataset = NewsDataset(val_df, vocab, config.max_len)
        test_dataset = NewsDataset(test_df, vocab, config.max_len)
        vocab_size = vocab.vocab_size
        print(f"词汇表大小: {vocab_size}")
    else:
        print("加载 BERT 分词器...")
        tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
        class BertDataset(torch.utils.data.Dataset):
            def __init__(self, df, tokenizer, max_len=256):
                self.texts = df['text'].values
                self.labels = df['label'].values
                self.tokenizer = tokenizer
                self.max_len = max_len
            def __len__(self):
                return len(self.labels)
            def __getitem__(self, idx):
                text = str(self.texts[idx])
                label = self.labels[idx]
                encoding = self.tokenizer.encode_plus(
                    text,
                    add_special_tokens=True,
                    max_length=self.max_len,
                    padding='max_length',
                    truncation=True,
                    return_tensors='pt'
                )
                return encoding['input_ids'].squeeze(0), encoding['attention_mask'].squeeze(0), torch.tensor(label)
        train_dataset = BertDataset(train_df, tokenizer)
        val_dataset = BertDataset(val_df, tokenizer)
        test_dataset = BertDataset(test_df, tokenizer)
        vocab_size = None

    # ---------- 3. DataLoader ----------
    print("\n[3/6] 创建 DataLoader...")
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)

    # ---------- 4. 初始化模型 ----------
    print("\n[4/6] 初始化模型...")
    model = get_model(config, vocab_size)
    print(model)

    # ---------- 5. 损失函数（可选加权） ----------
    print("\n[5/6] 设置损失函数...")
    if config.use_weighted_loss and config.model_name != 'bert':
        labels = train_df['label'].values
        class_weights = get_class_weights(labels, config.num_classes).to(config.device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        print("使用加权损失，权重:", class_weights.cpu().numpy())
    else:
        criterion = nn.CrossEntropyLoss()

    # ---------- 6. 优化器 ----------
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    # ---------- 7. 训练循环 ----------
    print("\n[6/6] 开始训练...")
    best_val_acc = 0
    for epoch in range(config.epochs):
        train_loss, train_acc, epoch_time = train_one_epoch(
            model, train_loader, optimizer, criterion, config, epoch, config.epochs
        )
        val_loss, val_acc = evaluate(model, val_loader, criterion, config)
        
        print(f"\nEpoch {epoch+1}/{config.epochs} 完成 | 耗时: {epoch_time:.2f}s | "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
        
        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), f'best_model_{config.model_name}.pth')
            print(f"模型已保存 (验证集准确率: {val_acc:.4f})")
        
        # 预估剩余时间
        avg_epoch_time = (time.time() - overall_start) / (epoch + 1)
        remaining_time = avg_epoch_time * (config.epochs - epoch - 1)
        print(f"预计剩余时间: {remaining_time/60:.2f} 分钟")

    # ---------- 8. 测试集最终评估 ----------
    print("\n加载最佳模型，在测试集上评估...")
    model.load_state_dict(torch.load(f'best_model_{config.model_name}.pth'))
    test_loss, test_acc = evaluate(model, test_loader, criterion, config)
    print(f"测试集 Loss: {test_loss:.4f}, 准确率: {test_acc:.4f}")

    total_time = time.time() - overall_start
    print(f"\n总训练时间: {total_time:.2f} 秒 ({total_time/60:.2f} 分钟)")

if __name__ == '__main__':
    main()