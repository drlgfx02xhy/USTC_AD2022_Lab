import pandas as pd
import numpy as np
import lightgbm as lgb
from lightgbm import early_stopping,log_evaluation
from sklearn.metrics import mean_squared_error
from defaults_3 import default_config
import os
from lgb_3 import get_single_data, get_config


dir_path = './config_3'
save_dir = './lgbmodels'

test_x, test_label, lgb_train, lgb_eval = get_single_data()

for filename in ['2_0.5_0.7_5.yaml']:
	params = get_config(filename)
	print(params)
	# 调用LightGBM模型，使用训练集数据进行训练（拟合）
	# my_model = lgb.train(params, lgb_train, num_boost_round=100000, valid_sets=lgb_eval, early_stopping_rounds=200, verbose_eval=-1)
	my_model = lgb.train(params, lgb_train, num_boost_round=100000, valid_sets=lgb_eval, \
						callbacks=[early_stopping(stopping_rounds=200), log_evaluation(period=2000)])
	file = filename.split('.yaml')[0]
	save_path = os.path.join(save_dir, file)
	my_model.save_model(save_path)
	# 使用模型对测试集数据进行预测
	predictions = my_model.predict(test_x, num_iteration=my_model.best_iteration)
	print(mean_squared_error(predictions, test_label))