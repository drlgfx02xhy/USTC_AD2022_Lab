import pandas as pd
import numpy as np
import lightgbm as lgb
from lightgbm import early_stopping,log_evaluation
from sklearn.metrics import mean_squared_error
from defaults_1 import default_config
import os
from lgb_1 import get_data, get_config


test_x, test_label, lgb_train, lgb_eval = get_data()

FIXED_params = {
    'task': 'train',
    'boosting_type': 'gbdt',  # 设置提升类型
    'objective': 'regression',  # 目标函数
    'metric': 'l2',  # 评估函数
    'num_leaves': 31,  # 叶子节点数
    'verbose': -1,  # <0 显示致命的, =0 显示错误 (警告), >0 显示信息
    'max_depth': 1,  # 树的最大深度
	}
SEARCH_params = {
	'bagging_fraction': 0.7,
	'bagging_freq': 2,
	'feature_fraction': 0.6,
	'learning_rate': 0.01
	}
params = {**FIXED_params, **SEARCH_params}

# 调用LightGBM模型，使用训练集数据进行训练（拟合）
# my_model = lgb.train(params, lgb_train, num_boost_round=100000, valid_sets=lgb_eval, early_stopping_rounds=200, verbose_eval=-1)
my_model = lgb.train(params, lgb_train, num_boost_round=100000, valid_sets=lgb_eval, \
						callbacks=[early_stopping(stopping_rounds=200), log_evaluation(period=1000)])
# 使用模型对测试集数据进行预测
predictions = my_model.predict(test_x, num_iteration=my_model.best_iteration)
print(mean_squared_error(predictions, test_label))
