import torch
from datasets import load_dataset
import pandas as pd


#定义数据集
class MyDataset(torch.utils.data.Dataset):
    def __init__(self):
        path = "/Users/xusijun/Documents/MY_NLP_001/transformers-main/examples/pytorch/translation/label_test.csv"
        data = pd.read_csv(path)
        dataset = []
        for index, row in data.iterrows():
            item = {}
            item['content'] = row['content']
            item['label_code'] = row['label_code']
            dataset.append(item)

        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, i):
        text = self.dataset[i]['content']
        label = self.dataset[i]['label_code']
        return text, label


dataset = MyDataset()

print(len(dataset), dataset[0])
