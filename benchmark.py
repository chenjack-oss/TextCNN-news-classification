import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import jieba
import time
from tqdm import tqdm
from config import Config

def main():
    config = Config()
    print("="*60)
    print("开始运行基准模型 (TF-IDF + 逻辑回归)")
    print("="*60)
    overall_start = time.time()

    # 1. 加载数据
    print("\n[1/6] 加载训练数据...")
    train_df = pd.read_csv(config.train_path)
    print(f"训练集样本数: {len(train_df)}")

    print("\n[2/6] 加载测试数据...")
    test_df = pd.read_csv(config.test_path)
    print(f"测试集样本数: {len(test_df)}")

    # 读取类别名称
    with open(config.class_names_path, 'r', encoding='utf-8') as f:
        class_names = [line.strip() for line in f.readlines()]
    print(f"类别数: {len(class_names)}")

    # 2. 分词（显示进度条）
    print("\n[3/6] 正在对训练集进行分词（这可能需要几分钟）...")
    train_texts_processed = []
    for text in tqdm(train_df['text'], desc="分词进度"):
        words = jieba.lcut(text)
        train_texts_processed.append(' '.join(words))  # 用空格连接词语

    print("\n[4/6] 正在对测试集进行分词...")
    test_texts_processed = []
    for text in tqdm(test_df['text'], desc="分词进度"):
        words = jieba.lcut(text)
        test_texts_processed.append(' '.join(words))

    # 3. 构建TF-IDF特征
    print("\n[5/6] 正在构建TF-IDF特征矩阵（可能耗时）...")
    start_time = time.time()
    tfidf = TfidfVectorizer(max_features=50000, min_df=2, tokenizer=lambda x: x.split())
    X_train = tfidf.fit_transform(train_texts_processed)
    print(f"TF-IDF特征矩阵形状: {X_train.shape}")
    print(f"特征构建完成，耗时: {time.time()-start_time:.2f} 秒")

    # 4. 训练逻辑回归
    print("\n[6/6] 开始训练逻辑回归模型...")
    start_time = time.time()
    lr = LogisticRegression(max_iter=1000, class_weight='balanced', verbose=1, n_jobs=-1)  # n_jobs=-1 使用所有CPU核
    lr.fit(X_train, train_df['label'])
    print(f"逻辑回归训练完成，耗时: {time.time()-start_time:.2f} 秒")

    # 5. 预测测试集
    print("\n正在预测测试集...")
    X_test = tfidf.transform(test_texts_processed)
    y_pred = lr.predict(X_test)
    y_true = test_df['label'].values

    # 6. 输出结果
    print("\n分类报告：")
    print(classification_report(y_true, y_pred, target_names=class_names))

    total_time = time.time() - overall_start
    print(f"\n总耗时: {total_time:.2f} 秒 ({total_time/60:.2f} 分钟)")

if __name__ == '__main__':
    main()