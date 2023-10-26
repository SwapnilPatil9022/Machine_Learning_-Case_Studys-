# Requre librarys

import pandas as pd
import matplotlib
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

###################################################
# Project_Name : House Price Prediction
# Input_CSV_File : kc_house_data.csv
# Author : Swapnil Ashok Patil
# Date : 15/02/2022
###################################################

def House_Prediction():
	# import csv using pandas
	data=pd.read_csv(r'kc_house_data.csv')

	data.head()

	# Exploratory Data Analysis
	print("=================================================")
	D1 = data.shape
	print(D1)
	print("=================================================")

	print(data)
	print("=================================================")

	pd.set_option('display.float_format', lambda x: '%.5f' % x)

	print("=================================================")

	D = data.describe()
	print(D)

	print("=================================================")

	data.info()
	print(data)
	 
	D2 = data.isnull().sum()

	print(D2)

	print("=================================================")

	D3 = data.columns

	print(D3)

	print("=======================================================")

	X = data[['bedrooms', 'bathrooms', 'sqft_living',
	       'sqft_lot', 'floors', 'waterfront', 'view', 'condition', 'grade',
	       'sqft_above', 'sqft_basement',
	        'sqft_living15', 'sqft_lot15']]

	print(X)

	Y = data['price']

	print(Y)

	print("=========================================================")

	X_train, X_test , Y_train , Y_test = train_test_split(X,Y,test_size=0.25,random_state=101)

	print(X_train)

	print(Y_train)

	print(X_test)

	print(Y_test)

	print("=======================================================")

	# Normalization basically Standarization

	std=StandardScaler()

	X_train_std=std.fit_transform(X_train)
	X_test_std=std.transform(X_test)

	print(X_train)

	print(X_train_std)

	print(X_test_std)

	print(Y_train)

	print(Y_test)

	print("========================================================")

	# Model training

	lr=LinearRegression()

	lr.fit(X_train_std,Y_train)

	Y_pred=lr.predict(X_test_std)

	print(Y_pred)

	print(Y_test)

	print("=========================================================")

	from sklearn.metrics import mean_absolute_error,r2_score

	print("#########################")
	mean_absolute_error(Y_test,Y_pred)
	print("#########################")

	print(X_test)

	X_test.loc[7148]

	print("==========================================")
	D5 = r2_score(Y_test,Y_pred)
	print(D5)
	print("===========================================")

	# Lets predict for Single House

	new_house = [[3,1,1520,5000,1,0,0,3,8,1000,1,2000,5000]]

	new_house_std = std.transform(new_house)

	print(new_house_std)

	print("############################")

	print(int(lr.predict(new_house_std)))

def main():
	House_Prediction()

if __name__ == "__main__":
	main()