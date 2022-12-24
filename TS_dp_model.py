import pandas as pd
import numpy as np
from sklearn import svm
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import MaxAbsScaler,MinMaxScaler,Normalizer,RobustScaler,StandardScaler


#np不要以科學符號顯示
np.set_printoptions(suppress=True)

#讀去資料
train_data = pd.read_csv('dp_dataset.csv')
# print(train_data)


X = np.array(train_data.drop(['alert_key','sar_flag','cust_id','date','sar_flag','occupation_code'],axis = 1))

transformer_MaxAbsScaler   = MaxAbsScaler().fit(X)
transformer_MinMaxScaler   = MinMaxScaler().fit(X)
transformer_Normalizer     = Normalizer().fit(X)
transformer_RobustScaler   = RobustScaler().fit(X)
transformer_StandardScaler = StandardScaler().fit(X)


X_MaxAbsScaler = transformer_MaxAbsScaler.transform(X)
X_MinMaxScaler = transformer_MinMaxScaler.transform(X)
X_Normalizer = transformer_Normalizer.transform(X)
X_RobustScaler = transformer_RobustScaler.transform(X)
X_StandardScaler = transformer_StandardScaler.transform(X)


###########linear###########
# clf = OneClassSVM(kernel = 'linear', gamma = 'auto').fit(X)
# outlier_MaxAbsScaler = clf.predict(X_MaxAbsScaler)
# np.savetxt("OneClassSVM_linear_MaxAbsScaler.csv", outlier_MaxAbsScaler, delimiter=",")
# 
# clf2 = OneClassSVM(kernel = 'linear', gamma = 'auto').fit(X)
# outliers_MinMaxScale = clf2.predict(X_MinMaxScaler)
# np.savetxt("OneClassSVM_linear_MinMaxScale.csv", outliers_MinMaxScale, delimiter=",")
# 
# clf3 = OneClassSVM(kernel = 'linear', gamma = 'auto').fit(X)
# outliers_Normalizer = clf3.predict(X_Normalizer)
# np.savetxt("OneClassSVM_linear_Normalizer.csv", outliers_Normalizer, delimiter=",")
# 
# clf4 = OneClassSVM(kernel = 'linear', gamma = 'auto').fit(X)
# outliers_RobustScaler = clf4.predict(X_RobustScaler)
# np.savetxt("OneClassSVM_linear_RobustScaler.csv", outliers_RobustScaler, delimiter=",")
# 
# clf5 = OneClassSVM(kernel = 'linear', gamma = 'auto').fit(X)
# outliers_StandardScaler = clf5.predict(X_StandardScaler)
# np.savetxt("OneClassSVM_linear_StandardScaler.csv", outliers_StandardScaler, delimiter=",")
# 
# 
# 
# ###########poly###########
# clf = OneClassSVM(kernel = 'rbf', gamma = 'auto').fit(X)
# outliers_MaxAbsScaler = clf.predict(X_MaxAbsScaler)
# np.savetxt("OneClassSVM_rbf_MaxAbsScaler.csv", outliers_MaxAbsScaler, delimiter=",")
# 
# clf2 = OneClassSVM(kernel = 'rbf', gamma = 'auto').fit(X)
# outliers_MinMaxScaler = clf2.predict(X_MinMaxScaler)
# np.savetxt("OneClassSVM_rbf_MinMaxScale.csv", outliers_MinMaxScaler, delimiter=",")
# 
# clf3 = OneClassSVM(kernel = 'rbf', gamma = 'auto').fit(X)
# outliers_Normalizer = clf3.predict(X_Normalizer)
# np.savetxt("OneClassSVM_rbf_Normalizer.csv", outliers_Normalizer, delimiter=",")
# 
# clf4 = OneClassSVM(kernel = 'rbf', gamma = 'auto').fit(X)
# outliers_RobustScaler = clf4.predict(X_RobustScaler)
# np.savetxt("OneClassSVM_rbf_RobustScaler.csv", outliers_RobustScaler, delimiter=",")
# 
# clf5 = OneClassSVM(kernel = 'rbf', gamma = 'auto').fit(X)
# outliers_StandardScaler = clf5.predict(X_StandardScaler)
# np.savetxt("OneClassSVM_rbf_StandardScaler.csv", outliers_StandardScaler, delimiter=",")
# 
# 
# ###########rbf###########
# clf = OneClassSVM(kernel = 'sigmoid', gamma = 'auto').fit(X)
# outliers_MaxAbsScaler = clf.predict(X_MaxAbsScaler)
# np.savetxt("OneClassSVM_sigmoid_MaxAbsScaler.csv", outliers_MaxAbsScaler, delimiter=",")
# 
# clf2 = OneClassSVM(kernel = 'sigmoid', gamma = 'auto').fit(X)
# outliers_MinMaxScaler = clf2.predict(X_MinMaxScaler)
# np.savetxt("OneClassSVM_sigmoid_MinMaxScale.csv", outliers_MinMaxScaler, delimiter=",")
# 
# clf3 = OneClassSVM(kernel = 'sigmoid', gamma = 'auto').fit(X)
# outliers_Normalizer = clf3.predict(X_Normalizer)
# np.savetxt("OneClassSVM_sigmoid_Normalizer.csv", outliers_Normalizer, delimiter=",")
# 
# clf4 = OneClassSVM(kernel = 'sigmoid', gamma = 'auto').fit(X)
# outliers_RobustScaler = clf4.predict(X_RobustScaler)
# np.savetxt("OneClassSVM_sigmoid_RobustScaler.csv", outliers_RobustScaler, delimiter=",")
# 
# clf5 = OneClassSVM(kernel = 'sigmoid', gamma = 'auto').fit(X)
# outliers_StandardScaler = clf5.predict(X_StandardScaler)
# np.savetxt("OneClassSVM_sigmoid_StandardScaler.csv", outliers_StandardScaler, delimiter=",")



###########sigmoid###########
clf = OneClassSVM(kernel = 'poly', gamma = 'auto').fit(X)
outliers_MaxAbsScaler = clf.predict(X_MaxAbsScaler)
np.savetxt("OneClassSVM_poly_MaxAbsScaler.csv", outliers_MaxAbsScaler, delimiter=",")

clf2 = OneClassSVM(kernel = 'poly', gamma = 'auto').fit(X)
outliers_MinMaxScaler = clf2.predict(X_MinMaxScaler)
np.savetxt("OneClassSVM_poly_MinMaxScale.csv", outliers_MinMaxScaler, delimiter=",")

clf3 = OneClassSVM(kernel = 'poly', gamma = 'auto').fit(X)
outliers_Normalizer = clf3.predict(X_Normalizer)
np.savetxt("OneClassSVM_poly_Normalizer.csv", outliers_Normalizer, delimiter=",")

clf4 = OneClassSVM(kernel = 'poly', gamma = 'auto').fit(X)
outliers_RobustScaler = clf4.predict(X_RobustScaler)
np.savetxt("OneClassSVM_poly_RobustScaler.csv", outliers_RobustScaler, delimiter=",")

clf5 = OneClassSVM(kernel = 'poly', gamma = 'auto').fit(X)
outliers_StandardScaler = clf5.predict(X_StandardScaler)
np.savetxt("OneClassSVM_poly_StandardScaler.csv", outliers_StandardScaler, delimiter=",")
