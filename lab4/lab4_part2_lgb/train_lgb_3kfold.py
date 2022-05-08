import pandas as pd
import numpy as np
import lightgbm as lgb
from lightgbm import early_stopping,log_evaluation
from sklearn.metrics import mean_squared_error
from defaults_3 import default_config
import os
from lgb_3 import get_kfold_data, get_config

def experiment(filename, train_x, train_label, test_x, test_label):
	params = get_config(filename)
	# print(params)
	# 调用LightGBM模型，使用训练集数据进行训练（拟合）
	file = filename.split('.yaml')[0]
	print('Config : {}'.format(file))
	mse = []
	for fold in range(6):
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
		print('\n')
	mse = np.array(mse)
	print('Mean Squared Error : {}+/-{}\n'.format(mse.mean(), mse.std()))
	with open('./log/kfold.txt', 'a') as f:
			f.write('Config : {}'.format(file))
			f.write('    Mean Squared Error : {}+/-{}\n'.format(mse.mean(), mse.std()))


if __name__ == '__main__':
	dir_path = './config_3'

	file_list = [f for f in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, f))]

	train_x, train_label, test_x, test_label = get_kfold_data(k=6)
	
	for filename in file_list[44:]:
		experiment(filename, train_x, train_label, test_x, test_label)