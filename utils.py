import json
from tqdm import tqdm
import numpy as np
import torch
from transformers import AutoTokenizer
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold


# ***************************************************
# 读取大尺寸.json格式的数据，返回格式大致为：[{}, {}, ...]
# ***************************************************
def read_json(path):
    tempt = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            dic = json.loads(line)
            tempt.append(dic)
    return tempt


# ***************************************************
# 从训练集中划分出：1.训练集（4/5） 2.开发验证集（1/5）
# 得到训练集和开发验证集的id列表：train_data_ids = valid_data_ids = []
# ***************************************************
def divide_data(data, split_k):
    fact_list = [x['fact'].strip for x in data]
    label_list = [x['meta']['accusation'][0] for x in data]
    skf = StratifiedKFold(n_splits=split_k)
    for train_data_ids, valid_data_ids in skf.split(fact_list, label_list):
        return train_data_ids, valid_data_ids


# ***************************************************
# 覆写Dataset类，用于得到模型输入
# ***************************************************
class MyDataset(Dataset):
    def __init__(self, data, unique_labels):  # data=[{},{},{}...]  unique_labels=[]
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        self.data = data
        self.unique_labels =  unique_labels  # str格式标签到int的映射

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        fact = self.data[idx]['fact'].strip()
        label = self.unique_labels.index(self.data[idx]['meta']['accusation'][0]) # 将str格式标签映射为int格式

        inputa = self.tokenizer(fact,
                                truncation=True,
                                add_special_tokens=True,
                                max_length=200,
                                padding='max_length',
                                return_tensors='pt')

        # inputa.keys() = input_ids, attention_mask, token_type_ids
        # inputa['input_ids'].shape = [batch_size, 1, max_length]
        return inputa, label


# ************************************************************
# 利用 1.read_json函数 2.MyDataset类 返回DataLoader类型的数据
# ************************************************************
def get_train_valid_dataloader(path, train_batch_size, valid_batch_size, split_k):
    train_data = read_json(path)
    with open(r'./data_cail2019/unique_labels') as f:
        unique_labels = json.load(f)  # 获取标签列表，共159个不同的标签

    train_data_ids, valid_data_ids = divide_data(train_data, split_k)

    train_ds = MyDataset([train_data[i] for i in train_data_ids], unique_labels)
    train_dl = DataLoader(train_ds, batch_size=train_batch_size, shuffle=True)

    valid_ds = MyDataset([train_data[i] for i in valid_data_ids], unique_labels)
    valid_dl = DataLoader(valid_ds, batch_size=valid_batch_size, shuffle=True)
    return train_dl, valid_dl


# ************************************************************
# 利用 1.read_json函数 2.MyDataset类 返回DataLoader类型的测试数据
# ************************************************************
def get_test_dataloader(path, test_batch_size):
    test_data = read_json(path)
    with open(r'./data_cail2019/unique_labels') as f:  # 获取标签列表，共159个不同的标签
        unique_labels = json.load(f)

    test_ds = MyDataset(test_data, unique_labels)
    test_dl = DataLoader(test_ds, batch_size=test_batch_size, shuffle=True)
    return test_dl


# ********************************************************
# 选择学习率调整策略 1.（默认）余弦模拟退火 2.余弦模拟退火热重启
# ********************************************************
def fetch_scheduler(optimizer, schedule='CosineAnnealingLR'):
    if schedule == 'CosineAnnealingLR':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=500,
                                                   eta_min=1e-6)
    elif schedule == 'CosineAnnealingWarmRestarts':
        scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=100,
                                                             eta_min=1e-6)
    elif schedule == None:
        return None

    return scheduler


# *********************************************
# 设置可人工赋值的随机种子，以保证结果可复现
# *********************************************
def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ***************************************************
# 读取全部数据中的label，得到unique的label集合，并以list形式返回
# ***************************************************
def save_unique_labels(train_path, test_path):
    train_data = []
    with open(train_path, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            dic = json.loads(line)
            train_data.append(dic)
    print('train_data read complete !!!')

    test_data = []
    with open(test_path, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            dic = json.loads(line)
            test_data.append(dic)
    print('test_data read complete !!!')

    labels = []
    for x in tqdm(train_data + test_data):
        labels.append(x['meta']['accusation'][0])
    unique_labels = list(set(labels))

    with open('./data_cail2019/unique_labels', 'w', encoding='utf-8') as f:
        json.dump(unique_labels, f)

if __name__ == '__main__':
    # save_unique_labels(r'./data_cail2019/train.json', r'./data_cail2019/test.json')
    with open(r'./data_cail2019/unique_labels') as f:
        unique_labels = json.load(f)
    print(len(unique_labels))
    print(len(set(unique_labels)))

