# 数据分析及实践-实验四

PB20000326 &nbsp; 徐海阳

---

[toc]

## Part 1 分类算法实践

###  1.1 算法主要流程

Step 1.  将lab3中特征工程得到的DIY.csv转换为.npy文件，并进行5折交叉验证所需要的train/test数据集划分

```python
data = np.load('./data/DIY.npy')

data_part = cross_validation(data, 5)

for i in range(5):
	print('part: %d' % (i+1))
	train, test = split_train_test(data_part, i)
```

Step 2.  用logistic回归进行二分类

```python
class MLP(object):
    # ...
for i in range(5):
    print('part: %d' % (i+1))
	train, test = split_train_test(data_part, i)
	model = MLP(train, test, train.shape[1]-1, 1, 0.1, 5001)
	model.train_model()
```

### 1.2 算法关键技术

Step 1: 将DIY.npy数据分为5个part，其中四个concatenate成为train，剩余一个为test。

Step 2: 构造class: MLP，实现sigmoid，sigmoid_derivative（对sigmoid求导），forward（前向传播），judge（judge>0则代表predict结果正确），backward（反向传播），test_model（在test集上计算正确率），train_model （在train集上计算正确率）方法。

### 1.3 算法实现

关键代码结构如下。完整代码见附录（1）。

```python
import numpy as np

def cross_validation(data, k):
	# split the data into k parts
	# return the k parts
	return data.reshape(k, -1, data.shape[1])

def split_train_test(data_part, k):
	# split the data_part into train and test
	# return the train and test
	test = data_part[k, :, :]
	train = np.delete(data_part, k, axis=0)
	train = np.concatenate(train, axis=0)
	return train, test

class MLP(object):
	def __init__(self, train, test, n_in, n_out, lr, epoch):
		# ...

	def sigmoid(self, x): 
		# ...

	def sigmoid_derivative(self, x): 
		# ...

	def forward(self, x): # (bsz, n_in)
		# ...

	def judge(self, y_hat, y):
		"""
		if (y_hat > 0.5 and y = 1) or (y_hat < 0.5 and y = 0),then return positive: right predict!
        create a function to judge whether the prediction is correct
		"""
		# ...

	def backward(self, x, y): # (bsz, n_in) , (bsz, n_out)
		# ...
	
  
	def test_model(self, data, label):
		# ...

	def train_model(self):
		# ...
        
def main():
	data = np.load('./data/DIY.npy')

	data_part = cross_validation(data, 5)

	for i in range(5):
		print('part: %d' % (i+1))
		train, test = split_train_test(data_part, i)
		model = MLP(train, test, train.shape[1]-1, 1, 0.1, 5001)
		model.train_model()
		print('\n\n')
  
if __name__ == '__main__':
	main()
```

### 1.4 实验记录

（k折交叉验证，4:1比例，共有5折）

在learning_rate=0.1，epoch=5001时的结果如下：

<img src="pic/p1.png" style="zoom:75%;" />

|k|ACC|
|:---|:---|
|1|0.917989|
|2|0.929038|
|3|0.922502|
|4|0.931684|
|5|0.929194|
|平均值|0.926081|

## Part 2 预测算法实践

1. 尝试了lightgbm库
2. 尝试了sklearn库中的12种回归算法

### 2.1 lightgbm







## 附录

1. 见 lab4\lab4_part1\lab4_part1.py
2. 