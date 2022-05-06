import lightgbm as lgb
import numpy as np
import pandas as pd
import os
from bestmodel import get_bestmodel

bestmodel = get_bestmodel()

D_test = np.load('./data/D_test.npy')

testset = D_test[:,:-1]

pred = bestmodel.predict(testset, num_iteration=bestmodel.best_iteration)

math = pred.reshape(-1,1)

df = pd.DataFrame(math, columns=['MATH'])

df.to_csv('./pred/pred.csv', header=True, index=True)