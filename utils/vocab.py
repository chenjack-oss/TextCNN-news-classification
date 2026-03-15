import jieba
import pandas as pd
from collections import Counter
import pickle
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

class Vocab:
    def __init__(self, max_size=50000, min_freq=2):
        self.max_size = max_size
        self.min_freq = min_freq
        self.word2idx = {'<PAD>': 0, '<UNK>': 1}
        self.idx2word = {0: '<PAD>', 1: '<UNK>'}
        self.vocab_size = 2

    def build_vocab(self, texts):
        counter = Counter()
        # 使用 tqdm 显示进度
        for text in tqdm(texts, desc="正在分词并构建词汇表"):
            words = jieba.lcut(text)
            counter.update(words)
        # 按频率排序，保留高频词
        most_common = counter.most_common(self.max_size)
        for word, freq in most_common:
            if freq >= self.min_freq:
                self.word2idx[word] = self.vocab_size
                self.idx2word[self.vocab_size] = word
                self.vocab_size += 1

    def encode(self, text, max_len=300):
        words = jieba.lcut(text)[:max_len]
        ids = [self.word2idx.get(w, 1) for w in words]  # 1 for <UNK>
        if len(ids) < max_len:
            ids += [0] * (max_len - len(ids))
        return ids

def save_vocab(vocab, path):
    with open(path, 'wb') as f:
        pickle.dump(vocab, f)

def load_vocab(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

class NewsDataset(Dataset):
    def __init__(self, df, vocab, max_len=300):
        self.texts = df['text'].values
        self.labels = df['label'].values
        self.vocab = vocab
        self.max_len = max_len

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        input_ids = self.vocab.encode(text, self.max_len)
        return torch.tensor(input_ids), torch.tensor(label)