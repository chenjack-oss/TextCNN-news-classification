from flask import Flask, request, jsonify
import torch
import jieba
from config import Config
from utils.vocab import load_vocab
from models.textcnn import TextCNN
from models.bilstm import BiLSTM
from models.bert import BertClassifier
from transformers import BertTokenizer

app = Flask(__name__)

# 加载配置和模型
config = Config()
# 选择要部署的模型，可以在启动时指定
model_name = 'textcnn'  # 可根据需要修改为 'bilstm' 或 'bert'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 读取类别名称
with open(config.class_names_path, 'r', encoding='utf-8') as f:
    class_names = [line.strip() for line in f.readlines()]
config.num_classes = len(class_names)

if model_name == 'bert':
    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
    model = BertClassifier(num_classes=config.num_classes).to(device)
    # 修改点：添加 weights_only=True
    model.load_state_dict(torch.load(f'best_model_{model_name}.pth', map_location=device, weights_only=True))
else:
    vocab = load_vocab(config.vocab_path)
    if model_name == 'textcnn':
        model = TextCNN(vocab.vocab_size, config.embed_size, config.num_classes).to(device)
    elif model_name == 'bilstm':
        model = BiLSTM(vocab.vocab_size, config.embed_size, num_classes=config.num_classes).to(device)
    # 修改点：添加 weights_only=True
    model.load_state_dict(torch.load(f'best_model_{model_name}.pth', map_location=device, weights_only=True))
model.eval()

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    text = data.get('text', '')
    if not text:
        return jsonify({'error': 'No text provided'}), 400

    # 预处理
    if model_name == 'bert':
        encoding = tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=256,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        input_ids = encoding['input_ids'].to(device)
        attention_mask = encoding['attention_mask'].to(device)
        with torch.no_grad():
            logits = model(input_ids, attention_mask)
    else:
        input_ids = vocab.encode(text, config.max_len)
        input_tensor = torch.tensor([input_ids]).to(device)
        with torch.no_grad():
            logits = model(input_tensor)
    pred = logits.argmax(dim=1).item()
    result = {
        'category': class_names[pred],
        'label': pred,
        'probabilities': logits.softmax(dim=1).cpu().numpy().tolist()[0]
    }
    return jsonify(result)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)