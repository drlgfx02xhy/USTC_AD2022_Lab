import pandas as pd
import numpy as np
import lightgbm as lgb
from lightgbm import early_stopping,log_evaluation
from sklearn.metrics import mean_squared_error
from defaults import default_config
import os
from add_lgb import get_data, get_config


dir_path = './add_config'
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
	with open('./log/add_result.txt', 'a') as f:
		f.write('Config : {}'.format(file))
		f.write('    Mean Squared Error : {}\n'.format(mean_squared_error(predictions, test_label)))