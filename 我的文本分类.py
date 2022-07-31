import torch
from datasets import load_dataset

# 加载训练数据dataset
# 定义数据集
class Dataset(torch.utils.data.Dataset):
    def __init__(self, split):
        self.dataset = load_dataset(path='seamew/ChnSentiCorp', split=split)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, i):
        text = self.dataset[i]['text']
        label = self.dataset[i]['label']
        return text, label


dataset = Dataset('train')

len(dataset), dataset[0]

# dataloader及fn函数定义，返回token信息


# 模型训练


# 模型预测
