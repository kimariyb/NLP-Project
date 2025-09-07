import random
import re
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from typing import List, Tuple, Dict, Optional, Set, Union
from tqdm import tqdm


class TranslationVocab:
    """
    翻译词汇表类，用于处理源语言和目标语言的词汇表构建和ID映射
    """

    def __init__(self, name: str, special_tokens: List[str] = None):
        """
        初始化词汇表

        Args:
            name: 词汇表名称
            special_tokens: 特殊标记列表，如<SOS>, <EOS>, <PAD>等
        """
        self.name = name
        self.special_tokens = special_tokens or []

        # 词汇表集合和映射字典
        self.word2id: Dict[str, int] = {}
        self.id2word: Dict[int, str] = {}
        self.vocab_set: Set[str] = set()
        self.n_words = 0

        # 添加特殊标记
        for token in self.special_tokens:
            self.add_word(token)

    def add_word(self, word: str) -> int:
        """
        添加单词到词汇表

        Args:
            word: 要添加的单词

        Returns:
            int: 单词对应的ID
        """
        if word not in self.word2id:
            self.word2id[word] = self.n_words
            self.id2word[self.n_words] = word
            self.vocab_set.add(word)
            self.n_words += 1
        return self.word2id[word]

    def add_sentence(self, sentence: List[str]) -> None:
        """
        添加句子中的所有单词到词汇表

        Args:
            sentence: 单词列表形式的句子
        """
        for word in sentence:
            self.add_word(word)

    def words_to_ids(self, words: List[str], pad_token: str = None, max_length: int = None) -> List[int]:
        """
        将单词列表转换为ID列表

        Args:
            words: 单词列表
            pad_token: 填充标记
            max_length: 最大长度，超过截断，不足填充

        Returns:
            List[int]: ID列表
        """
        # 将单词转换为ID
        ids = [self.word2id.get(word, self.word2id.get(pad_token, 0)) for word in words]

        # 处理长度
        if max_length is not None:
            if len(ids) > max_length:
                ids = ids[:max_length]  # 截断
            elif pad_token is not None:
                # 填充
                pad_id = self.word2id.get(pad_token, 0)
                ids += [pad_id] * (max_length - len(ids))

        return ids

    def ids_to_words(self, ids: List[int], ignore_special: bool = False) -> List[str]:
        """
        将ID列表转换为单词列表

        Args:
            ids: ID列表
            ignore_special: 是否忽略特殊标记

        Returns:
            List[str]: 单词列表
        """
        words = []
        for idx in ids:
            word = self.id2word.get(idx, '')
            if not ignore_special or word not in self.special_tokens:
                words.append(word)
        return words


class TranslationDataset(Dataset):
    """
    翻译数据集类，用于加载和处理翻译并行语料
    """
    # 特殊标记常量
    SOS_TOKEN = '<sos>'
    EOS_TOKEN = '<eos>'
    PAD_TOKEN = '<pad>'
    SOS_ID = 0
    EOS_ID = 1
    PAD_ID = 2

    def __init__(self, data_path: str, max_length: Optional[int] = None):
        """
        初始化翻译数据集

        Args:
            data_path: 数据文件路径
            max_length: 最大序列长度，超过此长度的句子将被过滤
        """
        self.data_path = data_path
        self.max_length = max_length
        self.special_tokens = [self.SOS_TOKEN, self.EOS_TOKEN, self.PAD_TOKEN]

        # 初始化词汇表
        self.src_vocab = TranslationVocab('english', self.special_tokens)
        self.tgt_vocab = TranslationVocab('french', self.special_tokens)

        # 加载和处理数据
        self.pairs = self._load_and_process_data()

        # 构建词汇表
        self._build_vocab()

    @staticmethod
    def _clean_text(text: str) -> str:
        """
        清理文本，移除特殊字符并标准化

        Args:
            text: 原始文本

        Returns:
            str: 清理后的文本
        """
        text = text.lower().strip()
        # 在句号、问号、感叹号前添加空格
        text = re.sub(r"([.!?])", r" \1", text)
        # 移除除字母、句号、问号、感叹号外的所有字符
        text = re.sub(r"[^a-zA-Z.!?]+", r" ", text)
        # 移除多余空格
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def _load_and_process_data(self) -> List[Tuple[List[str], List[str]]]:
        """
        加载并处理数据

        Returns:
            List[Tuple[List[str], List[str]]]: 处理后的句子对列表
        """
        pairs = []
        try:
            with open(self.data_path, 'r', encoding='utf-8') as f:
                for line in tqdm(f, desc=f'Loading data from {self.data_path}'):
                    if '\t' not in line:
                        continue  # 跳过无效行

                    # 分割源语言和目标语言句子
                    src_text, tgt_text = line.strip().split('\t', 1)

                    # 清理文本
                    src_text = self._clean_text(src_text)
                    tgt_text = self._clean_text(tgt_text)

                    # 分词
                    src_words = src_text.split()
                    tgt_words = tgt_text.split()

                    # 过滤过长的句子
                    if self.max_length and (len(src_words) > self.max_length or len(tgt_words) > self.max_length):
                        continue

                    # 在目标语言句子末尾添加EOS标记
                    tgt_words.append(self.EOS_TOKEN)

                    pairs.append((src_words, tgt_words))
        except Exception as e:
            print(f"Error loading data: {e}")

        return pairs

    def _build_vocab(self) -> None:
        """
        构建源语言和目标语言的词汇表
        """
        print(f"Building vocabulary for {self.src_vocab.name} and {self.tgt_vocab.name}...")
        for src_words, tgt_words in tqdm(self.pairs, desc='Building vocabulary'):
            self.src_vocab.add_sentence(src_words)
            self.tgt_vocab.add_sentence(tgt_words)

        print(f"English vocabulary size: {self.src_vocab.n_words}")
        print(f"French vocabulary size: {self.tgt_vocab.n_words}")

    def __len__(self) -> int:
        """
        返回数据集大小

        Returns:
            int: 数据集大小
        """
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Tuple[List[str], List[str], int, int, int]:
        """
        获取指定索引的数据项

        Args:
            idx: 索引

        Returns:
            Tuple[List[str], List[str], int, int, int]: 源语言单词列表、目标语言单词列表、
                                                      源语言长度、目标语言长度和索引
        """
        src_words, tgt_words = self.pairs[idx]
        src_len = len(src_words)
        tgt_len = len(tgt_words)
        return src_words, tgt_words, src_len, tgt_len, idx


def translation_collate_fn(batch: List[Tuple], dataset: TranslationDataset) -> Tuple[
    torch.Tensor, torch.Tensor, List[int], List[int], List[int]]:
    """
    自定义的collate函数，用于处理批次数据

    Args:
        batch: 批次数据
        dataset: 翻译数据集实例

    Returns:
        Tuple[torch.Tensor, torch.Tensor, List[int], List[int], List[int]]:
            源语言张量、目标语言张量、源语言长度列表、目标语言长度列表和索引列表
    """
    # 初始化源序列和目标序列列表
    src = []
    tgt = []
    src_lens = []
    tgt_lens = []
    indices = []

    # 找出批次中的最大序列长度
    src_max_len = 0
    tgt_max_len = 0

    for item in batch:
        src_words, tgt_words, src_len, tgt_len, idx = item
        src_max_len = max(src_max_len, src_len)
        tgt_max_len = max(tgt_max_len, tgt_len)

        src_lens.append(src_len)
        tgt_lens.append(tgt_len)
        indices.append(idx)

    # 转换为ID并填充
    for item in batch:
        src_words, tgt_words, _, _, _ = item

        # 转换为ID并填充到最大长度
        src_indices = dataset.src_vocab.words_to_ids(
            src_words, pad_token=dataset.PAD_TOKEN, max_length=src_max_len
        )
        tgt_indices = dataset.tgt_vocab.words_to_ids(
            tgt_words, pad_token=dataset.PAD_TOKEN, max_length=tgt_max_len
        )

        src.append(src_indices)
        tgt.append(tgt_indices)

    # 转换为张量，格式为 [batch_size, seq_len]
    src_tensor = torch.LongTensor(src)
    tgt_tensor = torch.LongTensor(tgt)

    return src_tensor, tgt_tensor, src_lens, tgt_lens, indices


def split_dataset(dataset: TranslationDataset, train_ratio: float = 0.8, random_seed: int = 42) -> Tuple[
    Subset, Subset]:
    """
    将数据集按照指定比例分割为训练集和验证集

    Args:
        dataset: 要分割的数据集
        train_ratio: 训练集比例
        random_seed: 随机种子，用于保证结果可重现

    Returns:
        Tuple[Subset, Subset]: 训练集和验证集
    """
    # 设置随机种子
    random.seed(random_seed)
    torch.manual_seed(random_seed)

    # 获取数据集大小
    dataset_size = len(dataset)

    # 生成索引列表
    indices = list(range(dataset_size))

    # 随机打乱索引
    random.shuffle(indices)

    # 计算分割点
    split_point = int(dataset_size * train_ratio)

    # 分割索引
    train_indices = indices[:split_point]
    val_indices = indices[split_point:]

    # 创建训练集和验证集
    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)

    print(f"Dataset split completed:")
    print(f"- Total samples: {dataset_size}")
    print(f"- Training samples: {len(train_dataset)} ({len(train_dataset) / dataset_size:.2%})")
    print(f"- Validation samples: {len(val_dataset)} ({len(val_dataset) / dataset_size:.2%})")

    return train_dataset, val_dataset


def create_dataloader(
    data_path: str, batch_size: int,
    max_length: Optional[int] = None,
    shuffle: bool = True,
    split_ratio: Optional[float] = None) -> Tuple[
        Union[DataLoader, Tuple[DataLoader, DataLoader]],
        Union[TranslationDataset, Subset]
    ]:
    """
    创建数据加载器，可选地将数据集分割为训练集和验证集

    Args:
        data_path: 数据文件路径
        batch_size: 批次大小
        max_length: 最大序列长度
        shuffle: 是否打乱数据
        split_ratio: 分割比例，如果为None则不分割

    Returns:
        Tuple[Union[DataLoader, Tuple[DataLoader, DataLoader]], Union[TranslationDataset, Subset]]:
            数据加载器（或训练和验证数据加载器的元组）和数据集（或子集）
    """
    # 创建数据集
    dataset = TranslationDataset(data_path, max_length)

    # 创建数据加载器，使用自定义的collate函数
    def collate_fn(batch):
        return translation_collate_fn(batch, dataset)

    if split_ratio is not None:
        # 分割数据集为训练集和验证集
        train_dataset, val_dataset = split_dataset(dataset, train_ratio=split_ratio)

        # 创建训练数据加载器
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=collate_fn
        )

        # 创建验证数据加载器
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,  # 验证集通常不需要打乱
            collate_fn=collate_fn
        )

        return (train_loader, val_loader), dataset
    else:
        # 创建单个数据加载器
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=collate_fn
        )

        return dataloader, dataset


# 示例用法
if __name__ == "__main__":
    data_path = "./data/eng-fra-v2.txt"

    # 创建分割后的数据集加载器（8:2分割）
    (train_loader, val_loader), dataset = create_dataloader(
        data_path,
        batch_size=32,
        max_length=50,
        shuffle=True,
        split_ratio=0.8
    )

    # 打印一些信息
    print(f"English vocabulary size: {dataset.src_vocab.n_words}")
    print(f"French vocabulary size: {dataset.tgt_vocab.n_words}")
    print(f"Number of training batches: {len(train_loader)}")
    print(f"Number of validation batches: {len(val_loader)}")

    # 获取一个训练批次的数据
    for batch in train_loader:
        src_tensor, tgt_tensor, src_lens, tgt_lens, indices = batch
        print(f"Training batch - Source tensor shape: {src_tensor.shape}")
        print(f"Training batch - Target tensor shape: {tgt_tensor.shape}")
        break

    # 获取一个验证批次的数据
    for batch in val_loader:
        src_tensor, tgt_tensor, src_lens, tgt_lens, indices = batch
        print(f"Validation batch - Source tensor shape: {src_tensor.shape}")
        print(f"Validation batch - Target tensor shape: {tgt_tensor.shape}")
        break