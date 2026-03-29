import pandas as pd
import numpy as np
import jieba
import os


def preprocess(path: str, output_path: str = None) -> pd.DataFrame:
    """
    数据预处理函数，读取数据并进行必要的处理。
    Args:
        path (str): 数据文件的路径。
        output_path (str, optional): 输出文件的路径。如果为None，则不保存到文件。
    Returns:
        pd.DataFrame: 预处理后的 DataFrame 数据。
    """
    # 检查文件是否存在
    if not os.path.exists(path):
        raise FileNotFoundError(f"文件 {path} 不存在")

    # 读取数据
    df = pd.read_csv(path, sep='\t', header=None, names=['sentence', 'label'])

    # 删除缺失值
    df = df.dropna()

    # 统计文本长度
    df['text_length'] = df['sentence'].apply(lambda x: len(str(x)))

    # 计算均值和标准差
    mean_text_length = np.mean(df['text_length'])
    std_text_length = np.std(df['text_length'])

    print(f"文本长度的均值: {mean_text_length:.2f}")
    print(f"文本长度的标准差: {std_text_length:.2f}")

    # 分词
    df['words'] = df['sentence'].apply(lambda x: ' '.join(jieba.cut(str(x))))

    # 如果指定了输出路径，则保存到文件
    if output_path:
        df.to_csv(output_path, index=False, sep='\t', encoding='utf-8')
        print(f"预处理后的数据已保存到 {output_path}")

    return df


def main():
    """主函数，处理训练集、测试集和验证集数据。"""
    # 定义文件路径
    data_dir = './data'
    train_path = os.path.join(data_dir, 'train.txt')
    test_path = os.path.join(data_dir, 'test.txt')
    val_path = os.path.join(data_dir, 'val.txt')

    # 检查数据目录是否存在
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"数据目录 {data_dir} 不存在")

    # 处理训练集
    if os.path.exists(train_path):
        preprocess(train_path, os.path.join(data_dir, 'train_preprocessed.csv'))
    else:
        print(f"警告: 训练集文件 {train_path} 不存在")

    # 处理测试集
    if os.path.exists(test_path):
        preprocess(test_path, os.path.join(data_dir, 'test_preprocessed.csv'))
    else:
        print(f"警告: 测试集文件 {test_path} 不存在")

    # 处理验证集
    if os.path.exists(val_path):
        preprocess(val_path, os.path.join(data_dir, 'val_preprocessed.csv'))
    else:
        print(f"警告: 验证集文件 {val_path} 不存在")


if __name__ == '__main__':
    main()
