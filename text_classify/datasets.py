import torch
import pandas as pd
import string

from typing import List, Tuple, Optional
from torch.utils.data import Dataset

ONEHOT_LETTER = string.ascii_letters + " .,;'"
ONEHOT_DIM = len(ONEHOT_LETTER)
CATEGORIES = [
    'Italian', 'English', 'Arabic', 'Spanish', 'Scottish', 'Irish', 
    'Chinese', 'Vietnamese', 'Japanese', 'French', 'Greek', 'Dutch', 
    'Korean', 'Polish', 'Portuguese', 'Russian', 'Czech', 'German'
]
CATEGORIES_DIM = len(CATEGORIES)


class NameClassDataset(Dataset):
    def __init__(self, x: Optional[List[str]]=None, y: Optional[List[str]] = None):
        super(NameClassDataset, self).__init__()
        self.x = x
        self.y = y

    @staticmethod
    def read_data(path: str):
        df = pd.read_csv(path, sep='\t', names=['name', 'category'], header=None)
        return df['name'].tolist(), df['category'].tolist()
    
    def transform(self, x: str, y: str) -> Tuple[torch.Tensor, torch.Tensor]:
        # 如果传入的数据是 str，则需要将其转化为 one-hot 编码
        x_tensor = self.str_to_onehot(x)
        y_tensor = torch.tensor(
            CATEGORIES.index(y), dtype=torch.long
        )

        return x_tensor, y_tensor
        
    def str_to_onehot(self, x: str):
        # 首先创建一个空的 tensor，用于存储 one-hot 编码
        onehot = torch.zeros(len(x), ONEHOT_DIM)
        # 遍历人名 的 每个字母 做成 one-hot 编码
        for i, char in enumerate(x):
            onehot[i][ONEHOT_LETTER.index(char)] = 1

        return onehot

    def __len__(self):
        return len(self.x) if self.x is not None else 0

    def __getitem__(self, idx):
        if self.x is None or self.y is None:
            raise ValueError("Dataset not initialized with data. Please provide x and y when creating the dataset.")
        tensor_x, tensor_y = self.transform(self.x[idx], self.y[idx])
        return tensor_x, tensor_y
        
        
if __name__ == '__main__':
    x_list, y_list = NameClassDataset.read_data("./data/name_classfication.txt")
    dataset = NameClassDataset(x_list, y_list)
    print(dataset[0])