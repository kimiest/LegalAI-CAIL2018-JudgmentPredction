## CAIL2018比赛任务-法律判决预测-BERT基础模型
 
### 1. 说明

全代码包含详细的中文注释！！！

使用Pytorch和Transformers框架实现

模型网络结构为：BERT+全连接分类头

为了便于理解和学习，代码没有过度集成

使用了：1.lr_scheduler='CosineAnnealingLR'  
未来优化：2.warm_up   2.分组学习率  3.pin_memory=True  4.自动混合精度AMP
 
### 2. 文件介绍
 
data_cail2018文件夹   --->   包含CAIL2018官方发布的训练集train.json和测试集test.json，以及罪名标签列表unique_labels.json

models文件夹   --->   包含模型网络结构BERT.py

utils.py   --->   包含数据读取、预处理、数据加载等工具类和函数

train.py   --->   训练一个epoch的代码

valid.py   --->   验证一个epoch的代码

run.py   --->   执行模型训练

test.py   --->   执行模型测试
 
### 3. 模型训练
 
直接运行run.py文件，可以执行模型训练，记得修改随机数种子
 
```
python run.py
```
 
### 4. 模型测试
 
执行test.py文件，可以执行模型测试
 
```
python test.py
```

### 5. 标签映射

为了将str类型的罪名标签映射为int类型，需要执行utils.py中的return_unique_labels()函数，得到全部罪名标签的列表，并保存为.json文件：'./data_cail2018/unique_labels.json' 。
数据加载过程中，将根据该列表对罪名标签进行映射，例如“偷窃”是列表中的第12个元素，则该罪名标签映射为int类型的12。
共有196个不同的罪名标签



