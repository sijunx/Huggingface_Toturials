import torch
from datasets import load_dataset
from datasets import load_from_disk
import random


# 定义数据集
class Dataset(torch.utils.data.Dataset):
    # def __init__(self, split):
    def __init__(self):
        # dataset = load_dataset(path='seamew/ChnSentiCorp', split=split)
        # def f(data):
        #     return len(data['text']) > 40
        #
        # self.dataset = dataset.filter(f)
        # self.dataset = load_from_disk('./data/ChnSentiCorp')
        self.dataset = load_from_disk(
            '/Users/xx/Documents/MY_NLP_001/Huggingface_Toturials-main/data/ChnSentiCorp')
        print(self.dataset)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, train_type, i):
        text = self.dataset[train_type]['text'][i]
        print("完整的句子:", text)
        # 切分一句话为前半句和后半句
        sentence1 = text[:20]
        sentence2 = text[20:40]
        label = 0
        print("切分第一句:", sentence1)
        print("切分第二句:", sentence2)

        # 有一半的概率把后半句替换为一句无关的话
        if random.randint(0, 1) == 0:
            j = random.randint(0, len(self.dataset) - 1)
            # sentence2 = self.dataset[j]['text'][20:40]
            sentence2 = self.dataset["train"]['text'][j]

            label = 1

        print("第一句：", sentence1)
        print("第二句：", sentence2)
        print("标签：", label)
        return sentence1, sentence2, label


dataset = Dataset()

print("-----------------")
result = dataset.__getitem__("train", 0)
print("result:", result)
print("result[0]:", result[0])
