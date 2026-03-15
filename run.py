import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='新闻文本分类系统')
    parser.add_argument('--mode', type=str, required=True,
                        choices=['prepare', 'train', 'evaluate', 'benchmark', 'deploy'],
                        help='运行模式')
    parser.add_argument('--model', type=str, default='textcnn',
                        help='选择模型: textcnn, bilstm, bert (用于train/evaluate/deploy)')
    args = parser.parse_args()

    if args.mode == 'prepare':
        from utils.data_loader import load_thucnews, split_data
        from config import Config
        config = Config()
        df, class_names = load_thucnews(config.raw_data_dir)
        train_df, val_df, test_df = split_data(df)
        os.makedirs('data/processed', exist_ok=True)
        train_df.to_csv(config.train_path, index=False)
        val_df.to_csv(config.val_path, index=False)
        test_df.to_csv(config.test_path, index=False)
        with open(config.class_names_path, 'w', encoding='utf-8') as f:
            for name in class_names:
                f.write(name + '\n')
        print("数据准备完成！")
    elif args.mode == 'train':
        import train
        config = train.Config()
        config.model_name = args.model
        train.main()
    elif args.mode == 'evaluate':
        import evaluate
        evaluate.main(args.model)
    elif args.mode == 'benchmark':
        import benchmark
        benchmark.main()
    elif args.mode == 'deploy':
        import deploy
        deploy.app.run()