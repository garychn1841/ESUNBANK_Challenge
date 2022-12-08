import pandas as pd
import numpy as np

pd.set_option('display.max_columns', 20)


#read data
cctx = pd.read_csv('cctx_public_proba.csv')
dp = pd.read_csv('dp_public_proba.csv')
remit = pd.read_csv('remit_public_proba.csv')
ccba = pd.read_csv('ccba_public_proba.csv')
row = pd.read_csv('D:/PYTHON/Dataset/ESUNBANK_Challenge/public_x_alert_date.csv')
submit = pd.read_csv('預測的案件名單及提交檔案範例.csv')



public = pd.concat([cctx,dp,remit,ccba])
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
submit.to_csv('submit.csv',index = False)