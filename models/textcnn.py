import torch
import torch.nn as nn
import torch.nn.functional as F

class TextCNN(nn.Module):
    def __init__(self, vocab_size, embed_size=300, num_classes=14, kernel_sizes=[3,4,5], num_filters=100, dropout=0.5):
        super(TextCNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=0)
        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels=embed_size, out_channels=num_filters, kernel_size=k)
            for k in kernel_sizes
        ])
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(len(kernel_sizes) * num_filters, num_classes)

    def forward(self, x):
        # x: (batch, seq_len)
        x = self.embedding(x)          # (batch, seq_len, embed)
        x = x.permute(0, 2, 1)         # (batch, embed, seq_len)
        conv_outs = [F.relu(conv(x)) for conv in self.convs]  # each: (batch, num_filters, new_seq_len)
        pooled = [F.max_pool1d(out, out.size(2)).squeeze(2) for out in conv_outs]  # each: (batch, num_filters)
        x = torch.cat(pooled, dim=1)   # (batch, len(kernels)*num_filters)
        x = self.dropout(x)
        logits = self.fc(x)
        return logits