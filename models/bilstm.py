import torch
import torch.nn as nn

class BiLSTM(nn.Module):
    def __init__(self, vocab_size, embed_size=300, hidden_size=256, num_layers=2, num_classes=14, dropout=0.5):
        super(BiLSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=0)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, bidirectional=True,
                             batch_first=True, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size * 2, num_classes)  # 双向拼接

    def forward(self, x):
        # x: (batch, seq_len)
        x = self.embedding(x)          # (batch, seq_len, embed)
        lstm_out, (h_n, c_n) = self.lstm(x)
        # 取最后一个时间步的隐藏状态（双向拼接）
        h_n = h_n[-2:]                  # 最后一层双向的两个方向
        h_n = torch.cat((h_n[0], h_n[1]), dim=1)  # (batch, hidden*2)
        out = self.dropout(h_n)
        logits = self.fc(out)
        return logits