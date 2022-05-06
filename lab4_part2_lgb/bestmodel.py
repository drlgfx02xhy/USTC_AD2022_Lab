import os
import lightgbm as lgb

def get_bestmodel():
	log_path = './log/all_result.txt'
	with open(log_path, 'r') as f:
		lines = f.readlines()

	mse_list = []
	for i in range(len(lines)):
		mse = float(lines[i][48:-1])
		mse_list.append(mse)
	mse_list.sort()

	for i in range(len(lines)):
		if mse_list[0] == float(lines[i][48:-1]):
			hyper_param1 = lines[i][9:23]
		if mse_list[1] == float(lines[i][48:-1]):
			hyper_param2 = lines[i][9:23]
		if mse_list[2] == float(lines[i][48:-1]):
			hyper_param3 = lines[i][9:23]

	model_file1 = os.path.join('./lgbmodels', hyper_param1)
	model_file2 = os.path.join('./lgbmodels', hyper_param2)
	model_file3 = os.path.join('./lgbmodels', hyper_param3)
	model1 = lgb.Booster(model_file=model_file1)
	model2 = lgb.Booster(model_file=model_file2)
	model3 = lgb.Booster(model_file=model_file3)
	model_file = [model_file1, model_file2, model_file3]
	model = [model1, model2, model3]
	return model_file, model



