
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from pandas.tseries.offsets import Hour, YearBegin
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.pipeline import make_pipeline
from imblearn.pipeline import make_pipeline as imbalanced_make_pipeline
from sklearn.metrics import recall_score, precision_score, f1_score, accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix


df2 = pd.read_csv('creditcard.csv')
df = pd.read_csv('creditcard.csv')

data['step'] = pd.to_datetime(data['step']*1000000000*3600)
#data preprocessing
data.info()
data.describe()
data.head()
data.step.info()
df = data.drop(['step', 'nameOrig', 'nameDest'], axis=1) #drop irrelevant features
df.isnull().values.any() #look for missing values
df.head()
df.describe()

pd.Series(data['nameOrig'].unique()).count()



df_corr = df[['amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest']].corr()


#df2 ANALYSIS
df2.info()
df2.describe()
df2.head()

print('distribution of class',df2.Class.unique())

print('No. of missing values in data:', df2.isnull().sum().max()) #check for missing values


#Visualization
sns.countplot('Class', data = df2)
plt.title('Target variable class distribution - IMBALANCED!', size = 14)

sns.boxplot(x = 'Class', y = 'Amount', data = df2, fliersize=5)
plt.title('Amount vs Class Boxplots', size = 20)
plt.ylim((-10,500))



#PCs are scaled but need to scale remaining features
sc = StandardScaler()
df2['time_scaled'] = sc.fit_transform(df2['Time'].values.reshape(-1,1))
df2['amount_scaled'] = sc.fit_transform(df2['Amount'].values.reshape(-1,1))
df2.drop(['Amount','Time'], axis= 1, inplace = True)
df2 = df2[['time_scaled', 'amount_scaled', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7',\
 'V8', 'V9', 'V10', 'V11','V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', \
 'V19', 'V20', 'V21','V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'Class',]]

#more visualization
plt.figure(figsize = (24,20))
df_corr = df2.drop(['Class'], axis = 1).corr()
sns.heatmap(df_corr, cmap = 'coolwarm')
plt.xticks(rotation=90)
plt.yticks(rotation=0)
plt.title('Correlation Heatmap of dataset', size = 14)
plt.tight_layout()



dv1 = sns.lmplot(data=df, x = 'V2', y = 'Amount', fit_reg=False, \
                 hue = 'Class', size = 7)

dv2 = sns.lmplot(data = df, x = 'V7', y = 'Amount', fit_reg=False, \
                 hue = 'Class', size = 7)


#predictor/target variable split
X = df2.drop(['Class'], axis = 1)
y = df2['Class']

#severely imbalanced dataset!
print('No Fraud =', round(df2['Class'].value_counts()[0]/len(df2) * 100, 2), '% from dataset')
print('Fraud =', round(df2['Class'].value_counts()[1]/len(df2) * 100, 2), '% from dataset')

sss = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=42)

for train_index, test_index in sss.split(X, y):
    orig_Xtrain, orig_Xtest = X.iloc[train_index], X.iloc[test_index]
    orig_ytrain, orig_ytest = y.iloc[train_index], y.iloc[test_index]

#convert into numpy arrays
orig_Xtrain = orig_Xtrain.values
orig_Xtest = orig_Xtest.values
orig_ytrain = orig_ytrain.values
orig_ytest = orig_ytest.values

train_unique_label, train_count_label = np.unique(orig_ytrain, return_counts = True)
test_unique_label, test_count_label = np.unique(orig_ytest, return_counts = True)

print('Target variable distribution in train and test dataset')
print(train_count_label/len(orig_ytrain))
print(test_count_label/len(orig_ytest))


test_count_label.shape()
df = df2.sample(frac = 1)

df.head()
df2.head()




len(orig_Xtrain)


print('Length of X (train): {} | Length of y (train): {}'.format(len(orig_Xtrain), len(orig_ytrain)))
print('Length of X (test): {} | Length of y (test): {}'.format(len(orig_Xtest), len(orig_ytest)))



# List to append the score and then find the average
accuracy_lst = []
precision_lst = []
recall_lst = []
f1_lst = []
auc_lst = []

# Classifier with optimal parameters
# log_reg_sm = grid_log_reg.best_estimator_
log_reg_sm = LogisticRegression()

log_reg_params = {"penalty": ['l1', 'l2'], 'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}



rand_log_reg = RandomizedSearchCV(LogisticRegression(), log_reg_params, n_iter=4)


# Implementing SMOTE Technique 
# Cross Validating the right way
# Parameters

for train, test in sss.split(orig_Xtrain, orig_ytrain):
    pipeline = imbalanced_make_pipeline(SMOTE(ratio='minority'), rand_log_reg) # SMOTE happens during Cross Validation not before..
    model = pipeline.fit(orig_Xtrain[train], orig_ytrain[train])
    best_est = rand_log_reg.best_estimator_
    prediction = best_est.predict(orig_Xtrain[test])
    
    accuracy_lst.append(pipeline.score(orig_Xtrain[test], orig_ytrain[test]))
    precision_lst.append(precision_score(orig_ytrain[test], prediction))
    recall_lst.append(recall_score(orig_ytrain[test], prediction))
    f1_lst.append(f1_score(orig_ytrain[test], prediction))
    auc_lst.append(roc_auc_score(orig_ytrain[test], prediction))
    
print('---' * 45)
print('')
print("accuracy: {}".format(np.mean(accuracy_lst)))
print("precision: {}".format(np.mean(precision_lst)))
print("recall: {}".format(np.mean(recall_lst)))
print("f1: {}".format(np.mean(f1_lst)))
print('---' * 45)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
smote = SMOTE(random_state=42, ratio = 'minority')
os_Xtrain, os_ytrain = smote.fit_sample(orig_Xtrain, orig_ytrain)
len(os_ytrain[os_ytrain == 1])
len(os_ytrain[os_ytrain == 0])

#check if target variable is balanced!
np.unique(os_ytrain, return_counts = True)

clf = RandomForestClassifier(random_state=0)
clf.fit(os_Xtrain,os_ytrain)
y_pred = clf.predict(orig_Xtest)

cm = confusion_matrix(orig_ytest, y_pred)
accuracy_score(orig_ytest, y_pred)
recall_score(orig_ytest, y_pred)
precision_score(orig_ytest, y_pred)


np.unique(orig_ytest, return_counts = True)

from sklearn.metrics import roc_curve, auc

false_positive_rate, true_positive_rate, thresholds = roc_curve(orig_ytest, y_pred)
roc_auc = auc(false_positive_rate, true_positive_rate)
print (roc_auc) 

import matplotlib.pyplot as plt
plt.title('Receiver Operating Characteristic')
plt.plot(false_positive_rate, true_positive_rate, 'b', label='AUC = %0.2f'% roc_auc)
plt.legend(loc='lower right')
plt.plot([0,1],[0,1],'r--')
plt.xlim([-0.1,1.2])
plt.ylim([-0.1,1.2])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')

np.unique(y_train, return_counts = True)
