from yacs.config import CfgNode as CN

__C = CN()
__C.bagging_fraction = 0.8
__C.bagging_freq = 3
__C.feature_fraction = 0.8
__C.learning_rate = 0.01

def default_config():
	return __C.clone()
