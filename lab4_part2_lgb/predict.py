import lightgbm as lgb
import numpy as np
import pandas as pd
import os
from bestmodel import get_bestmodelpath, get_bestmodel


paths = get_bestmodelpath(5)
models = get_bestmodel(paths)

D_test = np.load('./data/D_test.npy')
testset = D_test[:,:-1]

pred = []
for i in range(len(models)):
	pred.append(models[i].predict(testset, num_iteration=models[i].best_iteration))

pred = np.array(pred)
math = np.mean(pred, axis=0)

df = pd.DataFrame(math, columns=['MATH'])

df.to_csv('./pred/pred.csv', header=True, index=True)

print("Prediction done!")