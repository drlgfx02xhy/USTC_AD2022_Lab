import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import mean_squared_error
from defaults_3 import default_config
import os

def get_data():
	# 划分数据集
	data = np.load('./data/DatawithrawL.npy')
	train = data[:int(data.shape[0]*0.8), :]
	test = data[int(data.shape[0]*0.8):, :]
	train_x = train[:, :-1]
	train_label = train[:, -1]
	test_x = test[:, :-1]
	test_label = test[:, -1]

	# 转换为Dataset数据格式
	lgb_train = lgb.Dataset(train_x, train_label)
	lgb_eval = lgb.Dataset(test_x, test_label, reference=lgb_train)
	return test_x, test_label, lgb_train, lgb_eval

def get_config(config_file):
	FIXED_params = {
    'task': 'train',
    'boosting_type': 'gbdt',  # 设置提升类型
    'objective': 'regression',  # 目标函数
    'metric': 'l2',  # 评估函数
	'learning_rate': 0.01,
	'num_leaves': 31,  # 叶子节点数
    'verbose': -1  # <0 显示致命的, =0 显示错误 (警告), >0 显示信息
	}
	SEARCH_params = default_config()
	SEARCH_params.merge_from_file(os.path.join('./config_3', config_file))
	params = {**FIXED_params, **SEARCH_params}
	return params

def get_single_data():
	# 划分数据集
	data = np.load('./data/DatawithrawL.npy')
	test = data[:int(data.shape[0]*0.2), :]
	train = data[int(data.shape[0]*0.2):, :]
	train_x = train[:, :-1]
	train_label = train[:, -1]
	test_x = test[:, :-1]
	test_label = test[:, -1]

	# 转换为Dataset数据格式
	lgb_train = lgb.Dataset(train_x, train_label)
	lgb_eval = lgb.Dataset(test_x, test_label, reference=lgb_train)
	return test_x, test_label, lgb_train, lgb_eval