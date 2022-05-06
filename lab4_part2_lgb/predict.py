import lightgbm as lgb
import numpy as np
import pandas as pd
import os
from bestmodel import get_bestmodel

bestpath, bestmodel = get_bestmodel()
print("Best model path: \n")
print(bestpath[0])
print(bestpath[1])
print(bestpath[2])

D_test = np.load('./data/D_test.npy')

testset = D_test[:,:-1]

pred = []
for i in range(len(bestmodel)):
	pred.append(bestmodel[i].predict(testset, num_iteration=bestmodel[i].best_iteration))

math = ((pred[0]+pred[1]+pred[2])/3).reshape(-1,1)

df = pd.DataFrame(math, columns=['MATH'])

df.to_csv('./pred/pred.csv', header=True, index=True)

print("Prediction done!")