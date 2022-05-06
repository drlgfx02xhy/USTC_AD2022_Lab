import os
import lightgbm as lgb

def get_bestmodel():
	log_path = './log/result.txt'
	with open(log_path, 'r') as f:
		lines = f.readlines()

	mse_list = []
	for i in range(len(lines)):
		mse = float(lines[i][48:-1])
		mse_list.append(mse)
	mse_list.sort()

	for i in range(len(lines)):
		if mse_list[0] == float(lines[i][48:-1]):
			hyper_param = lines[i][9:23]
			break

	model_file = os.path.join('./lgbmodels', hyper_param)
	model = lgb.Booster(model_file=model_file)
	return model



