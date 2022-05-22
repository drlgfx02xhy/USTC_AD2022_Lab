# 数据分析及实践-实验四

PB20000326 &nbsp; 徐海阳

---

[toc]

## Part 1 分类算法实践

### 文件树

![](pic/lab4_part1.png)

###  1.1 算法主要流程

#### Step 1

将lab3中特征工程得到的DIY.csv转换为.npy文件，并进行5折交叉验证所需要的train/test数据集划分

```python
data = np.load('./data/DIY.npy')

data_part = cross_validation(data, 5)

for i in range(5):
	print('part: %d' % (i+1))
	train, test = split_train_test(data_part, i)
```

#### Step 2

用logistic回归进行二分类

选用理由：实现的逻辑简单，并且我认为在lab3中我自己提取的特征DIY.npy是不错的，可以简单地认为这些因素加权求和后经过一个非线性变换就可以得到分类结果。

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

#### Step 1

将DIY.npy数据分为5个part，其中四个concatenate成为train，剩余一个为test。

#### Step 2

构造class: MLP，实现sigmoid，sigmoid_derivative（对sigmoid求导），forward（前向传播），judge（judge>0则代表predict结果正确），backward（反向传播），test_model（在test集上计算正确率），train_model （在train集上计算正确率）方法。

### 1.3 算法实现

关键代码结构如下。完整代码见附录。

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

5折交叉验证，4:1比例

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

观察训练后网络的w可知，“过往是否复读”和“知识掌握程度”的weight最大，这符合我在lab3特征工程中的数据分析。

## Part 2 预测算法实践

### Method 1 ：sklearn 文件树

<img src="pic/lab4_part2_sk.png" style="zoom:70%;" />

使用sklearn工具包的算法，如：

Adaboost, Bagging, DecisionTree, ExtraTree, GradientBoosting, KNN, Lasso, LinearReg, MLP, RandomForest, Ridge, SVM

### Method 2 ：lightgbm 文件树

<img src="pic/lab4_part2.png" style="zoom:50%;" />

使用lightgbm工具包

### 2.1 算法主要流程

#### Step 1

lab4_p2_preprocess.ipynb预处理: 

1. 测试集部分存入“data\D_test.npy”

2. 训练集部分存入“data\DatawithrawL.npy”

3. 取这些特征中与MATH相关系数最大的100个特征，存入“data\T100withrawL.npy”

#### Step 2

使用不同模型在训练集上进行kfold交叉验证

#### Step 3

挑选表现最好的模型在测试集上进行predict，保存结果

### 2.2 实验记录

#### 2.2.1 Method 1 sklearn

sklearn是简单常用的机器学习库。

针对回归问题，我选择了sklearn.ensemble集成学习库和sklearn.linear_model线性模型库进行模型训练。针对回归问题，最经典的LinearRegression的效果应该不会差，可以将它的MSEloss作为baseline。同时，集成学习（Bagging，Adaboost等）应该也有不错的效果。

首先，我在DatawithrawL训练集上使用默认参数，对各种算法都简单地跑了一遍：（Adaboost的例子，其余代码结构相同）

<img src="pic/Adaboost.png" style="zoom:50%;" />

结果如下图所示：

<img src="pic/sklearn.png" style="zoom:50%;" />

首先，考虑非线性模型。实验过程中，我发现Tree模型耗时长且效果不好。这一定程度上可能是因为没有调参。于是我调了max_depth, num_iterations等参数，发现速度仍然很慢，而且效果仍然很差。我觉得这可能是因为特征数量太多，导致决策树更倾向于划分那些分的子类更多的子节点，这有害决策树的学习。于是我将训练集换为了T100withrawL（前100个最相关的特征），发现效果略有提升，但是速度仍旧很慢而且效果很难进一步提升。同时，可以看出集成算法的效果确实不错。由于我在下一部分将用到lightgbm，它作为GBDT算法，效果优于仅仅利用Boost或Bagging的算法，也优于简单的单棵Tree算法，因此我决定在sklearn中不再对Tree和集成算法做调参，在下一部分lightgbm中详细调参。MLP在手动调参过后hidden_layer_sizes=(128,8),activation='relu',solver='adam',batch_size=1024,learning_rate_init=0.1效果变为1300左右，略优于LinearRegression。

再说线性模型：

1. LinearReg：线性回归，效果不错，作为baseline。
2. KNN：效果不好。我认为也是因为训练集中特征个数太多，并且有部分特征事实上对于MATH没有太多影响，所以在进行distance度量时影响结果。于是我将训练集换成了T100withrawL（前100个最相关的特征），并且在axis=1中只取[:5]，也就是只考虑5个影响因素最大的特征。然后手动调参，发现在n_neighbors=10时效果最好，达到了1800，不如baseline。
3. Lasso：默认情况下正则项alpha=1效果很差。手动调参，在alpha=0.0001时效果最好，达到1600左右，不如baseline。
4. Ridge：同样的，手动调参，在alpha=0.0001时效果最好，达到1300左右，略优于baseline。

总结，在不使用集成学习的情况下，线性方法中Lasso的效果最好，非线性方法中MLP的效果最好。都达到了1300左右。



#### 2.2.2 Method 2 lightgbm

lightgbm使用GBDT算法，拥有集成学习的优点，同时相比Xgboost速度快很多。

##### 第一次调参：learning_rate, feature_fraction, bagging_fraction, bagging_freq

在如下cfggen_1.py中生成超参数的.yaml文件：<img src="pic/cfg1.png" style="zoom:67%;" />

存入如下./config_1文件夹：

<img src="pic/config.png" style="zoom:50%;" />

在lgb_1.py中定义函数：1. get_data() 2. get_config(config_file)（根据yaml文件中更新超参）

train_lgb_1.py批量跑这81个参数配置下的模型：

```python
import pandas as pd
import numpy as np
import lightgbm as lgb
from lightgbm import early_stopping,log_evaluation
from sklearn.metrics import mean_squared_error
from defaults_1 import default_config
import os
from lgb_1 import get_data, get_config


dir_path = './config_1'
save_dir = './lgbmodels'

file_list = [f for f in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, f))]

test_x, test_label, lgb_train, lgb_eval = get_data()

for filename in file_list:
	params = get_config(filename)

	# 调用LightGBM模型，使用训练集数据进行训练（拟合）
	# my_model = lgb.train(params, lgb_train, num_boost_round=100000, valid_sets=lgb_eval, early_stopping_rounds=200, verbose_eval=-1)
	my_model = lgb.train(params, lgb_train, num_boost_round=100000, valid_sets=lgb_eval, \
						callbacks=[early_stopping(stopping_rounds=200), log_evaluation(period=200)])
	file = filename.split('.yaml')[0]
	save_path = os.path.join(save_dir, file)
	my_model.save_model(save_path)
	# 使用模型对测试集数据进行预测
	predictions = my_model.predict(test_x, num_iteration=my_model.best_iteration)
	print('\n')
	# log results in '/log/file.txt'
	with open('./log/result_1.txt', 'a') as f:
		f.write('Config : {}'.format(file))
		f.write('    Mean Squared Error : {}\n'.format(mean_squared_error(predictions, test_label)))
```

结果写入result_1.txt中。部分结果如下：

<img src="pic/result1.png" style="zoom: 33%;" />

发现learning_rate=0.01时最优，同时feature_fraction和bagging_fraction在0.8时优于0.9和1.0，这说明这两个超参并不在最有区间内取值，取值范围还可以更小，小于0.8。根据这些信息进行第二次调参。

##### 第二次调参: 调小feature_fraction和bagging_fraction的取值区间

在如下cfggen_2.py中生成超参数的.yaml文件，存入./config_2文件夹。

调小了feature_fraction和bagging_fraction，取值范围[0.5,0.7]

<img src="pic/cfg2.png" style="zoom:67%;" />

在lgb_2.py中定义函数：1. get_data() 2. get_config(config_file)（根据yaml文件中更新超参）

train_lgb_2.py批量跑这81个参数配置下的模型（代码与train_lgb_1.py类似），结果写入result_2.txt中。部分结果如下：

<img src="pic/result2.png" style="zoom: 33%;" />

发现效果优于第一次调参的结果。仍然是在learning_rate=0.01时最优。同时大部分feature_fraction和bagging_fraction在0.6要优于0.5，也有部分在0.7时更好，这说明这两个超参的取值区间已经可以大致确定在[0.5, 0.6, 0.7]。

这些参数的最优取值区间确定后，开始对最重要的max_depth（梯度提升树的深度）进行调参。

##### 第三次调参: 调试max_depth

在如下cfggen_3.py中生成超参数的.yaml文件，存入./config_3文件夹。

尝试max_depth取值[2,3,4]（一般树的深度小于叶子结点的log值，叶子结点默认为31）

<img src="pic/cfg3.png" style="zoom:67%;" />

在lgb_3.py中定义函数：1. get_data() 2. get_config(config_file)（根据yaml文件中更新超参）

train_lgb_3.py批量跑这81个参数配置下的模型（代码与train_lgb_1.py类似），结果写入result_3.txt中。部分结果如下：

<img src="pic/result3.png" style="zoom: 33%;" />

发现效果大大优于第二次调参的结果。仍然是在learning_rate=0.01时最优。树的深度在2时效果较好，也有些情况在3时最优。

此时，全部参数的最优取值区间确定下来：

```Python
learning_rate = 0.01
max_depth in [2,3]
feature_fraction in [0.5,0.7]
bagging_fraction in [0.5,0.7]
```

下一步根据kfold交叉验证进行细致实验。

##### 在参数的最优取值区间上进行kfold交叉预测

取定k=6（5不整除训练集样本数）。在train_lgb_3.py的基础上加入kfold的代码得到train_lgb_3kfold.py（代码与train_lgb_1.py类似），计算MSE的均值和方差。

结果写入kfold.txt。部分结果如下：

<img src="pic/kfold.png" style="zoom: 33%;" />

然后在bestiter.py中取MSE最小的5个模型，在k=12时进行一次12折交叉验证，目的是得到best_iteration，然后取平均，作为模型在整个训练集上训练的轮次。

（代码与train_lgb_1.py类似），加上提取best_iter的代码：

```python
best_iter.append(my_model.best_iteration)
best_iter = np.array(best_iter)
print('Best Iteration : {}\n'.format(best_iter.mean(), best_iter.std()))
```

结果如下：

<img src="pic/best_iter.png" style="zoom:80%;" />

最后在predict.py中用这5个模型在整个训练集上进行训练，然后在测试集D_test上做predict：

```python
import pandas as pd
import numpy as np
import lightgbm as lgb
from lightgbm import early_stopping,log_evaluation
from sklearn.metrics import mean_squared_error
from defaults_3 import default_config
import os
from lgb_3 import get_kfold_data, get_config
from bestmodel import get_best_cfg
import re

def get_train_test_data():
	D_test = np.load('./data/D_test.npy')
	test_x = D_test[:,:-1]
	D_train = np.load('./data/DatawithrawL.npy')
	train_x = D_train[:,:-1]
	train_label = D_train[:,-1]
	lgb_train = lgb.Dataset(train_x, train_label)
	return lgb_train, test_x
	

def get_best_iter():
	with open('./log/best_iter.txt', 'r') as f:
		lines = f.readlines()
		cfg = []
		best_iter = []
		for line in lines:
			cfg.append(line[9:20]+'.yaml')
			pos = re.search('Best Iteration : ', line).span()[1]
			best_iter.append(int(line[pos:pos+5]))
	f.close()
	return cfg, best_iter

def run_each_model(lgb_train, test_x, filename, best_iter):
	params = get_config(filename)
	my_model = lgb.train(params, lgb_train, num_boost_round=best_iter)
	prediction = my_model.predict(test_x, num_iteration=my_model.best_iteration)
	return prediction

def predict(lgb_train, test_x, cfgs, best_iters):
	predictions = []
	for cfg, best_iter in zip(cfgs, best_iters):
		prediction = run_each_model(lgb_train, test_x, cfg, best_iter)
		print(prediction)
		predictions.append(prediction)
	predictions = np.array(predictions)
	predictions = np.mean(predictions, axis=0)
	return predictions
 
if __name__ == '__main__':
	lgb_train, test_x = get_train_test_data()
	cfgs, best_iters = get_best_iter()
	print(best_iters)
	predictions = predict(lgb_train, test_x, cfgs, best_iters)
	np.save('./pred/pred.npy', predictions)
```

对得到的5个predictions做平均作为最终的测试集预测结果，将结果按格式要求存入csv文件。

<img src="pic/pred.png" style="zoom:50%;" />



## 3. 完成内容汇总

<img src="pic/require.png" style="zoom:50%;" />

1，2，3：完成

4：在part2中尝试了多种算法（sklearn的多种和lightgbm），并且详细调参

5：在part2中尝试了不同特征组合（Tree模型和KNN模型等用T100withrawL训练集，是取原始训练集中与MATH相关系数最大的100个特征组成的训练集）

6，7：完成

## 附录: part1完整代码

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
		# self.bsz = train.shape[0]
		# self.train = train
		# self.test = test
		self.n_in = n_in
		self.n_out = n_out
		self.lr = lr
		self.epoch = epoch
		self.train_data = train[:, :-1]
		train_label = train[:, -1]
		self.train_label = train_label.reshape(train_label.shape[0], 1)
		self.test_data = test[:, :-1]
		test_label = test[:, -1]
		self.test_label = test_label.reshape(test_label.shape[0], 1)
		self.w = np.random.randn(self.n_in, self.n_out)
		self.b = np.random.randn(self.n_out)

	def sigmoid(self, x): 
		return 1 / (1 + np.exp(-x))

	def sigmoid_derivative(self, x): 
		return self.sigmoid(x) * (1 - self.sigmoid(x))

	def forward(self, x): # (bsz, n_in)
		o = np.dot(x, self.w) + self.b # (bsz, n_out)
		y_hat = self.sigmoid(o) # (bsz, n_out)
		return o, y_hat

	def judge(self, y_hat, y):
		"""
		if (y_hat > 0.5 and y = 1) or (y_hat < 0.5 and y = 0),then return positive: right predict!
        create a function to judge whether the prediction is correct
		"""
		s = (y_hat - 0.5) * (y - 0.5)
		return s

	def backward(self, x, y): # (bsz, n_in) , (bsz, n_out)
		bsz = x.shape[0]
		o, y_hat = self.forward(x) # (bsz, n_out)
  
		# 链式法则
		d_L_d_y_hat = -y/y_hat + (np.ones_like(y)-y)/(np.ones_like(y)-y_hat) # (bsz, n_out)
		d_y_hat_d_o = self.sigmoid_derivative(y_hat) # (bsz, n_out)
		d_o_d_w = x # (bsz, n_in)
		d_o_d_b = np.ones((bsz, 1)) # (bsz, 1)
	
		d_L_d_w = np.mean(d_L_d_y_hat * d_y_hat_d_o * d_o_d_w, axis=0) # (n_in,)
		d_L_d_w = d_L_d_w.reshape(self.n_in, 1) # (n_in, 1)
		d_L_d_b = np.mean(d_L_d_y_hat * d_y_hat_d_o * d_o_d_b, axis=0) # (1,)
		self.w = self.w - self.lr * d_L_d_w
		self.b = self.b - self.lr * d_L_d_b
	
  
	def test_model(self, data, label):
		# pdb.set_trace()
		o, y_hat = self.forward(data)
		total = data.shape[0]
		"""
		correct are those whose prediction is correct, i.e. y_hat > 0.5 and y = 1
  		"""
		correct = np.sum(self.judge(y_hat, label)>0)
		accs = correct / total
		return accs

	def train_model(self):
		for i in range(self.epoch):
			self.backward(self.train_data, self.train_label)
			if i!=0 and i%5000 == 0:
				accs = self.test_model(self.test_data, self.test_label)
				print('epoch: %d, accs: %f' % (i, accs))

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

