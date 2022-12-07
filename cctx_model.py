import pandas as pd
from sklearn import svm
import numpy as np
# import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_fscore_support

#np不要以科學符號顯示
np.set_printoptions(suppress=True)

#讀取資料
train_data = pd.read_csv('cctx_dataset.csv')


features = np.array(train_data.drop(['alert_key','sar_flag','cust_id','date'],axis = 1))
# features = np.array(train_data[['risk_rank','tw','n_tw','ntd','n_ntd','amt']])
target = np.array(train_data.sar_flag)

# print(train_data.drop(['sar_flag','cust_id'],axis = 1))
# print(features)
# print(target)

#資料分割
X_train, X_test, y_train, y_test = train_test_split(features,target,test_size=0.7,random_state=0)
np.savetxt("X_train.csv", X_train, delimiter=",")
np.savetxt("X_test.csv", X_test, delimiter=",")

sc = StandardScaler().fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)
# print(X_train.shape)
# print(y_train.shape)

#建模訓練
clf = svm.SVC(kernel='poly' , gamma = 'auto', C = 1 , probability=True,class_weight='balanced')
clf.fit(X_train_std,y_train)

clf2 = DecisionTreeClassifier(criterion = 'gini',max_depth=5, random_state=0,class_weight='balanced') #max_depth=1,3,12
clf2.fit(X_train_std,y_train)

forest = RandomForestClassifier(criterion='gini', n_estimators=8,random_state=0,n_jobs=6,class_weight='balanced')
forest.fit(X_train_std,y_train)

#預測
pred  = clf.predict(X_test_std)
pred2  = clf2.predict(X_test_std)
pred3  = forest.predict(X_test_std)
np.savetxt("cctx_pred.csv", pred, delimiter=",")
np.savetxt("cctx_pred2.csv", pred2, delimiter=",")
np.savetxt("cctx_pred3.csv", pred3, delimiter=",")

#模型評分
# print('svm training score:',clf.score(X_train_std,y_train))
# print('svm test score:',clf.score(X_test_std,y_test))
# print('decsiontree training score:',clf2.score(X_train_std,y_train))
# print('decsiontree test score:',clf2.score(X_test_std,y_test))
# print('randomforest training score:',forest.score(X_train_std,y_train))
# print('randomforest test score:',forest.score(X_test_std,y_test))

s1 = precision_recall_fscore_support(y_test,pred)
s2 = precision_recall_fscore_support(y_test,pred2)
s3 = precision_recall_fscore_support(y_test,pred3)

# print('precision_svm major:%f  miner:%f '%(s1[0][0],s1[0][1]))
# print('recall_svm major:%f  miner:%f '%(s1[1][0],s1[1][1]))
# print('precision_decsiontree major:%f  miner:%f '%(s2[0][0],s2[0][1]))
# print('recall_decsiontree major:%f  miner:%f '%(s2[1][0],s2[1][1]))
# print('precision_randomforest major:%f  miner:%f '%(s3[0][0],s3[0][1]))
# print('recall_randomforest major:%f  miner:%f '%(s3[1][0],s3[1][1]))

#public dataset predict

#read data
row_data = pd.read_csv('cctx_public_dataset.csv')
public_data = np.array(row_data.drop(['alert_key','cust_id','date'],axis = 1))

#data standarized
public_data_std = sc.transform(public_data)

#predict
pred  = clf.predict(public_data_std)
a = clf.predict_proba(public_data_std)
print(a)
np.savetxt("cctx_public_proba.csv", a, delimiter=",")