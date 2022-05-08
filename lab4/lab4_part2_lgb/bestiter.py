import pandas as pd
import numpy as np
import lightgbm as lgb
from lightgbm import early_stopping,log_evaluation
from sklearn.metrics import mean_squared_error
from defaults_3 import default_config
import os
from lgb_3 import get_kfold_data, get_config
from bestmodel import get_best_cfg

def cal_each_cfg(filename, train_x, train_label, test_x, test_label):
	params = get_config(filename)
	# 调用LightGBM模型，使用训练集数据进行训练（拟合）
	file = filename.split('.yaml')[0]
	print('Config : {}'.format(file))
	mse = []
	best_iter = []
	for fold in range(12):
		print('Fold {}'.format(fold+1))
		train_x_fold = train_x[fold]
		train_label_fold = train_label[fold]
		test_x_fold = test_x[fold]
		test_label_fold = test_label[fold]
		lgb_train = lgb.Dataset(train_x_fold, train_label_fold)
		lgb_eval = lgb.Dataset(test_x_fold, test_label_fold, reference=lgb_train)
		my_model = lgb.train(params, lgb_train, num_boost_round=100000, valid_sets=lgb_eval, \
						callbacks=[early_stopping(stopping_rounds=2000), log_evaluation(period=2000)])
		# 使用模型对测试集数据进行预测
		prediction = my_model.predict(test_x_fold, num_iteration=my_model.best_iteration)
		mse.append(mean_squared_error(prediction, test_label_fold))
		best_iter.append(my_model.best_iteration)
		print('\n')
	mse = np.array(mse)
	best_iter = np.array(best_iter)
	print('Mean Squared Error : {}+/-{}\n'.format(mse.mean(), mse.std()))
	print('Best Iteration : {}\n'.format(best_iter.mean(), best_iter.std()))
	with open('./log/best_iter.txt', 'a') as f:
		f.write('Config : {}'.format(file))
		f.write('    Mean Squared Error : {}+/-{}'.format(mse.mean(), mse.std()))
		f.write('    Best Iteration : {}+/-{}\n'.format(best_iter.mean(), best_iter.std()))


if __name__ == '__main__':
	dir_path = './config_3'

	file_list = get_best_cfg(5)

	train_x, train_label, test_x, test_label = get_kfold_data(k=12)
	
	for filename in file_list:
		cal_each_cfg(filename, train_x, train_label, test_x, test_label)