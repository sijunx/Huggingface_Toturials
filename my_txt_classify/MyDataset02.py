import torch
import json
from datasets import load_dataset
from datasets import load_from_disk


# 加载训练数据dataset
# 定义数据集
class MyDataset(torch.utils.data.Dataset):
    def __init__(self, split):
        if split == 'train':
            self.my_data = load_data(
                '/Users/zard/Documents/nlp001/bert4keras/examples/datasets/myClsData/train.json')

        if split == 'test':
            self.my_data = load_data(
                '/Users/zard/Documents/nlp001/bert4keras/examples/datasets/myClsData/test.json')

        if split == 'dev':
            self.my_data = load_data(
                '/Users/zard/Documents/nlp001/bert4keras/examples/datasets/myClsData/dev.json')


    def __len__(self):
        return len(self.my_data)

    def __getitem__(self, i):
        text = self.my_data[i][0]
        label = self.my_data[i][1]
        return text, label


def load_data(filename):
    """加载数据
    单条格式：(文本, 标签id)
    """
    D = []
    with open(filename) as f:
        for i, l in enumerate(f):
            l = json.loads(l)
            text, label = l['sentence'], l['label']
            D.append((text, int(label)))
    print("D:", D)
    return D


myDataset = MyDataset('train')

x = myDataset[0]

print("x:", x)

print(len(myDataset), myDataset[1])
