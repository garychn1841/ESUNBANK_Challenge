import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

pd.set_option('display.max_columns', 20)


#read data
cctx = pd.read_csv('cctx_final_public_proba.csv')
dp = pd.read_csv('dp2_final_public_proba.csv')
remit = pd.read_csv('remit_final_public_proba.csv')
ccba = pd.read_csv('ccba2_final_public_proba.csv')
row = pd.read_csv('D:/PYTHON/Dataset/ESUNBANK_Challenge/public_x_alert_date.csv')
submit = pd.read_csv('預測的案件名單及提交檔案範例.csv')


# sc_dp = MinMaxScaler().fit(dp.drop('alert_key',axis=1))
# dp.probability = sc_dp.transform(dp.drop('alert_key',axis=1))
#
# sc_ccba = MinMaxScaler().fit(ccba.drop('alert_key',axis=1))
# ccba.probability = sc_ccba.transform(ccba.drop('alert_key',axis=1))
#
# sc_remit = MinMaxScaler().fit(remit.drop('alert_key',axis=1))
# remit.probability = sc_remit.transform(remit.drop('alert_key',axis=1))
#
# sc_cctx = MinMaxScaler().fit(cctx.drop('alert_key',axis=1))
# cctx.probability = sc_cctx.transform(cctx.drop('alert_key',axis=1))


public = pd.concat([dp])
public = public.groupby('alert_key').mean().sort_values('probability',ascending = False)

row.set_index('alert_key',inplace=True)

a = pd.concat([row,public],axis = 1).reset_index()
a.to_csv('a.csv', index = False)

bbb = np.zeros(3850)
submit.probability = bbb
print(submit)
print(a)

for i in range(len(submit)):

    flag = a[a['alert_key'] == int(submit.loc[i:i, :].alert_key)]

    if not flag.empty:
        submit.loc[i:i, 'probability'] = flag.probability.values

submit.sort_values('probability',ascending=False,inplace=True)
submit.probability.fillna(0,inplace = True)
submit.to_csv('submit2_final_mean().csv',index = False)

