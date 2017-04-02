'''

Convergence
2. Assign each point to a cluster belonging to the nearest mean
3. Find the new means/ centroid of the clusters


Predicting relationships



Clustering large dataset ino meaninfull groups



'''

import pandas as pd					# pandas is a dataframe library
import matplotlib.pyplot as plt		# matplotlib.pyplot plots data
import numpy as np 					# numpy provides N-dim object support
import os							# loading relative paths


trainFile = "C:/Users/Administrator/Documents/GitHub/recruitment_challenge/ML_201703/sample.csv" # adjust path if necessary

# 01-Load and review data step


pwd = os.getcwd()
os.chdir(os.path.dirname(trainFile))
trainData = pd.read_csv(os.path.basename(trainFile), header=None)  # loading data.
os.chdir(pwd)



trainData.shape
# (66136, 296)

trainData.head(5)
#    0  0.1  0.2    20000  0.3  0.4  1  0.5  0.6  0.7 ...  0.272  0.273  0.274  0.275  0.276  1.10  0.277  0.278  259.227165  B
# 0  0    0    0   7059.0    0    0  1    0    0    0 ...      0      0      0      0      0     0      1      0  271.983584  E
# 1  0    0    0   3150.0    0    0  1    0    0    0 ...      0      0      0      0      0     1      0      0  235.233437  D
# 2  0    0    0  24000.0    0    0  1    0    0    0 ...      0      0      0      0      0     0      1      0  415.104389  C
# 3  0    0    0   5600.0    0    0  1    0    0    0 ...      0      0      0      0      0     0      0      1  462.230610  D
# 4  0    0    0  16507.0    1    0  1    0    0    0 ...      0      0      0      0      1     0      0      0  824.520326  C
# [5 rows x 296 columns]


trainData.tail(5)
#        0  0.1  0.2    20000  0.3  0.4  1  0.5  0.6  0.7 ...  0.272  0.273  0.274  0.275  0.276  1.10  0.277  0.278  259.227165  B
# 66131  0    0    0   5000.0    0    0  1    0    0    0 ...      0      0      0      0      0     0      1      0  754.125582  C
# 66132  0    0    0  13759.0    0    0  1    0    0    0 ...      0      0      0      0      1     0      0      0  521.998666  D
# 66133  0    0    0  12100.0    2    0  1    0    0    0 ...      0      0      0      0      0     1      0      0  430.970745  C
# 66134  0    0    1  17280.0    0    0  1    0    0    0 ...      0      0      0      0      0     0      0      1  588.470479  D
# 66135  0    0    0   1047.0    0    0  0    0    0    0 ...      0      0      0      0      0     0      1      0  377.895620  C


trainData.isnull().values.any() # check for null values
#False


def plot_corr(trainData, size=11):
	"""
	Function plots a graphical correlation matrix for each pair of columns in the dataframe.
		Input:
		df: pandas DataFrame
		size: vertical and horizontal size of the plot

	Displays:
		matrix of correlation between columns. 	

		Blue-cyan-yellow-red-darkred -> less to more correlated
		0 ---------------------> 1
		Expect a darkred line running from top left to bottom right
	"""
	corr = trainData.corr()	# data frame correlation function
	fig, ax = plt.subplots(figsize = (size, size))
	ax.matshow(corr)	# color code the rectangles by correlation value
	plt.xticks(range(len(corr.columns)), corr.columns)	# draw x tick marks
	plt.yticks(range(len(corr.columns)), corr.columns)	# draw y tick marks


# Algorithms are largely mathematical models. As such they work best with numeric quantities

class_map = {'A': 1, 'B' : 2, 'C' : 3, 'D' : 4, 'E' : 5} # defining map dicionary


df['class'] = df['class'].map(class_map)






num_trueA = len (df.loc[df['A'] == True])
num_trueB = len (df.loc[df['B'] == True])
num_trueC = len (df.loc[df['C'] == True])
num_trueD = len (df.loc[df['D'] == True])
num_trueE = len (df.loc[df['E'] == True])


class_col = trainData[trainData.columns[295]]
dff = class_col.apply(pd.value_counts)
newTable = dff.fillna(0)


class_map = {True : 1, False : 0} # defining map dicionary 
newTable['A'] = newTable['A'].map(class_map)
newTable['B'] = newTable['B'].map(class_map)
newTable['C'] = newTable['C'].map(class_map)
newTable['D'] = newTable['D'].map(class_map)
newTable['E'] = newTable['E'].map(class_map)

newTable.head(5)

num_trueA = len(newTable.loc[newTable['A'] == True])
num_falseA = len(newTable.loc[newTable['A'] == False])

num_trueB = len(newTable.loc[newTable['B'] == True])
num_falseB = len(newTable.loc[newTable['B'] == False])

num_trueC = len(newTable.loc[newTable['C'] == True])
num_falseC = len(newTable.loc[newTable['C'] == False])

num_trueD = len(newTable.loc[newTable['D'] == True])
num_falseD = len(newTable.loc[newTable['D'] == False])

num_trueE = len(newTable.loc[newTable['E'] == True])
num_falseE = len(newTable.loc[newTable['E'] == False])

print("Number of A classes: {0}".format(num_trueA)) # 1,3%
print("Number of B classes: {0}".format(num_trueB)) # 10,0%
print("Number of C classes: {0}".format(num_trueC)) # 70,9%
print("Number of D classes: {0}".format(num_trueD)) # 14,0%
print("Number of E classes: {0}".format(num_trueE)) # 3,8%

'''
Only a really small amount of data is classified as "A" (1,3%). Thus, a standard learning technique might not work very well. In this particular case, we need to use a special advanced technique.

Around 10,0% of the data is classified as B. in these case, falls into the same problem as class A which has not many samples

The Class C represents the vast majority of the cases covering around 70,9% of the total amount of classes.

D -> 14,0%
E -> 3,8%

Standards prediction techniques can be used to classify what is C and what is NOT C
'''


# msk = np.random.rand(len(df)) < 0.7
# trainData = df[msk]
# testData = df[~msk]

# print("Shape of train data: {0}".format(trainData.shape))
# print("Shape of test data: {0}".format(testData.shape))

from sklearn.cross_validation import train_test_split

feature_col_names = range(0, 294)
predicted_class_name = [295]

X = df[feature_col_names].values # predictor feature columns (8 X m)
Y = df[predicted_class_name].values # predicted class (1=true, 0=false) column (1 X m)
split_test_size = 0.30

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size= split_test_size, random_state=42) # test_size = 0.3 is 30%, 42 is the answer to everything




