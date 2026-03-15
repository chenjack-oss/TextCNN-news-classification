import torch

class Config:
    # 数据路径
    raw_data_dir = 'D:/001BS/111/THUCNews'   # 原始数据文件夹
    train_path = 'data/processed/train.csv'
    val_path = 'data/processed/val.csv'
    test_path = 'data/processed/test.csv'
    vocab_path = 'data/vocab/vocab.pkl'
    class_names_path = 'data/processed/class_names.txt'   # 保存类别名称列表

    # 训练参数
    batch_size = 64
    epochs = 10
    learning_rate = 1e-3
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 模型选择：'textcnn', 'bilstm', 'bert'
    model_name = 'textcnn'

    # TextCNN / BiLSTM 通用参数
    max_len = 300
    embed_size = 300

    # 长尾处理：是否使用加权损失
    use_weighted_loss = True

    # 类别数（将在数据准备后自动设置）
    num_classes = None