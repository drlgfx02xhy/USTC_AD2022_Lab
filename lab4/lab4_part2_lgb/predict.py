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