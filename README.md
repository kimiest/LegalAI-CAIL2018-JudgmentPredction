# CAIL2018比赛任务-法律判决预测-BERT基础模型
 
================全代码包含详细的中文注释===============

1. 使用Pytorch和Transformers框架实现

2. 模型网络结构为：BERT+全连接分类头

3. 为了便于理解和学习，代码没有过度集成

===============全代码包含详细的中文注释=================
 
## 文件介绍
 
data_cail2018文件夹   --->   包含CAIL2018官方发布的训练集train.json和测试集test.json，以及罪名标签列表unique_labels.json

models文件夹   --->   包含模型网络结构BERT.py

utils.py   --->   包含数据读取、预处理、数据加载等工具类和函数

train.py   --->   训练一个epoch的代码

valid.py   --->   验证一个epoch的代码

run.py   --->   执行模型训练

test.py   --->   执行模型测试
 
### 模型训练
 
直接运行run.py文件，可以执行模型训练，记得修改随机数种子
 
```
python run.py
```
 
### 模型测试
 
执行test.py文件，可以执行模型测试
 
```
python test.py
```

