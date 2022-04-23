import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import json
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

sns.set_context("paper")  
sns.set(rc={'figure.figsize': (10, 8)})  # 设置画板大小
sns.set_style('whitegrid')
raw_path = "data/pica2015.csv"
cleaned_path = "C:/Users/x/Desktop/twodown/USTC_AD2022_Lab/lab3/data/pica2015_cleaned.csv"

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 500)

df = pd.read_csv(cleaned_path)

# first, see the distribution of `REPEAT`
rpt = df['REPEAT'].value_counts()
rpt.plot(kind='pie', autopct='%1.1f%%')
# extract the data of `REPEAT`
final_rpt = pd.DataFrame(df['REPEAT'], columns=['REPEAT'])

# make new dataframe from old dataframe according to index_list
def turn_to_list(str):
	begin, end = str.split('~')
	list = [int(i) for i in range(int(begin), int(end)+1)]
	return list

def make_df(df, index_list):
	df_new = pd.DataFrame()
	for i in range(len(index_list)):
		index = index_list[i]
		if(type(index) == int):
			col = df.columns[index]
		else:
			col = index
		df_new[col] = df[col]
	df_new['REPEAT'] = df['REPEAT']
	return df_new

# simply sum all the features with relation to `REPEAT` and standardlize the value
def simplesum_stdlz(df, name):
	if(type(df) == pd.core.frame.DataFrame):
		df = df.to_numpy()
	scaler = StandardScaler()
	df_new = scaler.fit_transform(np.sum(df, axis=1).reshape(-1,1))
	df_new_pd = pd.DataFrame(df_new).rename(columns={0: name})
	return df_new_pd

# sum all the features with relation to `REPEAT` and standardlize the value
def sum_stdlz(map, df, name):
	if(type(df) == pd.core.frame.DataFrame):
		df = df.to_numpy()
	scaler = StandardScaler()
	pn = map.iloc[-2]
	sign = map.iloc[-1][0]
	for i in range(len(pn)-1):
		if(i==0):
			if(sign>0):
				temp = df[:,i]
			else:
				temp = -df[:,i]
		else:
			if ((pn[i]>0 and sign > 0)or(pn[i]<0 and sign < 0)):
				temp = temp + df[:,i]
			else:
				temp = temp - df[:,i]
	df_new = scaler.fit_transform(temp.reshape(-1,1))
	df_new_pd = pd.DataFrame(df_new).rename(columns={0: name})
	return df_new_pd

# draw corr hot map
def draw_corr_map(df,method='pearson'):
	map = df.corr(method=method)
	sns.heatmap(map, annot=True, cmap='coolwarm')
	return map

tchr_pd = make_df(df, turn_to_list('79~84'))

map = draw_corr_map(tchr_pd, method='pearson')

final_tchr = sum_stdlz(map, tchr_pd, 'TCHR')