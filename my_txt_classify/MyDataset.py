# import torch
# from datasets import load_dataset
# from datasets import load_from_disk
#
# # 加载训练数据dataset
# # 定义数据集
# class MyDataset(torch.utils.data.Dataset):
#     def __init__(self, split):
#         # self.dataset = load_dataset(path='seamew/ChnSentiCorp', split=split)
#         self.dataset = load_from_disk('/Users/zard/Documents/nlp002/Huggingface_Toturials/data/ChnSentiCorp')
#         self.dataset = self.dataset[split]
#
#     def __len__(self):
#         return len(self.dataset)
#
#     def __getitem__(self, i):
#         text = self.dataset[i]['text']
#         label = self.dataset[i]['label']
#         return text, label
#
# myDataset = MyDataset('train')
#
# x = myDataset[0]
#
# print("x:", x)
#
# print(len(myDataset), myDataset[1])
#
#
