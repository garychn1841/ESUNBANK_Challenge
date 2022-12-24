import pandas as pd
from sklearn import svm
import numpy as np
# import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import precision_recall_fscore_support
from imblearn.over_sampling import SMOTE, ADASYN,SVMSMOTE
from sklearn.naive_bayes import GaussianNB , MultinomialNB , ComplementNB ,CategoricalNB
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2,f_classif

#np不要以科學符號顯示
np.set_printoptions(suppress=True)

#讀去資料
train_data = pd.read_csv('remit_dataset.csv')
# print(train_data)

dirty_X = train_data.drop(['alert_key','sar_flag','cust_id','date','risk_rank'],axis = 1)
dirty_Y = train_data.sar_flag

#資料標準化
sc = MinMaxScaler().fit(dirty_X)
X_dirty_std = sc.transform(dirty_X)
X_dirty_std = pd.DataFrame(X_dirty_std,columns=dirty_X.columns)
print(X_dirty_std)

#特徵選取
Select = SelectKBest(f_classif,k=3)
features = Select.fit_transform(X_dirty_std,dirty_Y)
features_names = Select.get_feature_names_out()
print(features_names)

# features = np.array(train_data.drop(['alert_key','sar_flag','cust_id','date'],axis = 1))
features = np.array(X_dirty_std[features_names])
target = np.array(train_data.sar_flag)

# print(train_data.drop(['sar_flag','cust_id'],axis = 1))
# print(features)
# print(target)

#資料分割
X_train_std, X_test_std, y_train, y_test = train_test_split(features,target,test_size=0.3,random_state=0)

print(X_train_std.shape)

# SMOTE
# X_train_std , y_train = SVMSMOTE(random_state = 0).fit_resample(X_train_std,y_train)
# print(X_train_std.shape)

#建模訓練

weight_minority_class = np.sum(y_train == 0)/np.sum(y_train == 1)
print(weight_minority_class)

clf = svm.SVC(kernel='rbf' , gamma = 'auto',C=100 , probability=True)
clf.fit(X_train_std,y_train)

clf2 = GaussianNB()
clf2.fit(X_train_std,y_train)

clf3 = LogisticRegression()
clf3.fit(X_train_std,y_train)


#預測
pred  = clf.predict(X_test_std)
pred2  = clf2.predict(X_test_std)
pred3  = clf3.predict(X_test_std)


#模型評分
print('svm training score:',clf.score(X_train_std,y_train))
print('svm test score:',clf.score(X_test_std,y_test))
print('decsiontree training score:',clf2.score(X_train_std,y_train))
print('decsiontree test score:',clf2.score(X_test_std,y_test))
print('LogisticRegression training score:',clf3.score(X_train_std,y_train))
print('LogisticRegression test score:',clf3.score(X_test_std,y_test))

s1 = precision_recall_fscore_support(y_test,pred)
s2 = precision_recall_fscore_support(y_test,pred2)
s3 = precision_recall_fscore_support(y_test,pred3)

print('precision_svm major:%f  miner:%f '%(s1[0][0],s1[0][1]))
print('recall_svm major:%f  miner:%f '%(s1[1][0],s1[1][1]))
print('precision_decsiontree major:%f  miner:%f '%(s2[0][0],s2[0][1]))
print('recall_decsiontree major:%f  miner:%f '%(s2[1][0],s2[1][1]))
print('precision_decsiontree major:%f  miner:%f '%(s3[0][0],s3[0][1]))
print('recall_decsiontree major:%f  miner:%f '%(s3[1][0],s3[1][1]))



#public dataset predict

#read data
row_data = pd.read_csv('remit_public_dataset.csv')
public_data = row_data.drop(['alert_key','cust_id','date','risk_rank'],axis = 1)
#data standarized
public_data_std = sc.transform(public_data)
public_data_std = pd.DataFrame(public_data_std,columns=public_data.columns)

public_data_std = np.array(public_data_std[features_names])



#predict
pred  = clf.predict(public_data_std)
a = clf.predict_proba(public_data_std)

pred2  = clf2.predict(public_data_std)
a2 = clf2.predict_proba(public_data_std)

pred3  = clf3.predict(public_data_std)
a3 = clf3.predict_proba(public_data_std)
# print(a)
# np.savetxt("dp_public_proba.csv", a, delimiter=",")

df_pred3 = pd.DataFrame(a3,columns=['1-probability','probability']).drop('1-probability',axis=1)
df_remit3 = pd.concat([row_data,df_pred3],axis=1)
df_remit3= df_remit3[['alert_key','probability']]
df_remit3.to_csv('remit3_final_public_proba.csv',index=False)

df_pred2 = pd.DataFrame(a2,columns=['1-probability','probability']).drop('1-probability',axis=1)
df_remit2 = pd.concat([row_data,df_pred2],axis=1)
df_remit2 = df_remit2[['alert_key','probability']]
df_remit2.to_csv('remit2_final_public_proba.csv',index=False)

df_pred = pd.DataFrame(a,columns=['1-probability','probability']).drop('1-probability',axis=1)
df_remit = pd.concat([row_data,df_pred],axis=1)
df_remit = df_remit[['alert_key','probability']]
df_remit.to_csv('remit_final_public_proba.csv',index=False)