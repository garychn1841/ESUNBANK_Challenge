import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

pd.options.mode.chained_assignment = None  # default='warn'
pd.set_option('display.max_columns', 20)

remit = pd.read_csv('D:/PYTHON/Dataset/ESUNBANK_Challenge/public_train_x_remit1_full_hashed.csv')
train_x_alert = pd.read_csv('D:/PYTHON/Dataset/ESUNBANK_Challenge/train_x_alert_date.csv')
custinfo = pd.read_csv('D:/PYTHON/Dataset/ESUNBANK_Challenge/public_train_x_custinfo_full_hashed.csv')
answer = pd.read_csv('D:/PYTHON/Dataset/ESUNBANK_Challenge/train_y_answer.csv')



def remit_train_preprocessing(dataset):

    # ##OneHotEncoder
    columnTransformer = ColumnTransformer([('encoder', OneHotEncoder(sparse=False), [2])],
                                          remainder='passthrough')
    remit_arr = np.array(columnTransformer.fit_transform(remit), dtype=str)
    encoded_remit = pd.DataFrame(remit_arr)
    # encoded_remit.to_csv('encoded_remit.csv',index = False)

    #
    #
    ##reset columns
    encoded_remit.columns = ['trans_no1', 'trans_no2', 'trans_no3', 'trans_no4', 'trans_no5', 'cust_id', 'date', 'trade_amount_usd']
    encoded_remit = encoded_remit[['cust_id', 'date','trans_no1', 'trans_no2', 'trans_no3', 'trans_no4', 'trans_no5','trade_amount_usd']]
    # encoded_remit.to_csv('encoded_remit_out.csv', index=False)
#     encoded_remit.amt[encoded_remit.amt == 'nan'] = 0
#     # print(encoded_remit.isnull().any())
#     # print(encoded_remit[encoded_remit.columns[encoded_remit.isnull().any()]])
#
    ##groupby
    encoded_remit.date = pd.to_numeric(encoded_remit.date)
    encoded_remit.trans_no1 = pd.to_numeric(encoded_remit.trans_no1)
    encoded_remit.trans_no2 = pd.to_numeric(encoded_remit.trans_no2)
    encoded_remit.trans_no3 = pd.to_numeric(encoded_remit.trans_no3)
    encoded_remit.trans_no4 = pd.to_numeric(encoded_remit.trans_no4)
    encoded_remit.trans_no5 = pd.to_numeric(encoded_remit.trans_no5)
    encoded_remit.trade_amount_usd = pd.to_numeric(encoded_remit.trade_amount_usd)

    remit_group = encoded_remit.groupby(['cust_id', 'date']).sum().reset_index()
    remit_group = remit_group.sort_values(['cust_id', 'date'])
    # remit_group.to_csv('remit_out.csv',index=False)
    #
    #
    answer.drop('alert_key', inplace=True, axis=1)
    data = pd.concat([dataset, answer], axis=1)
    #
    #
    # ##concat the cust_id&cctx in data
    a = custinfo.drop('alert_key', axis=1)
    data[a.columns] = np.nan
    custinfo.occupation_code.fillna(21, inplace=True)
    b = remit_group.drop(['cust_id', 'date'], axis=1)
    data[b.columns] = np.nan
    #
    #
    #
    #
    #
    for i in range(len(data)):
        print(i)
        flag1 = custinfo[custinfo['alert_key'] == int(data.loc[i:i, :].alert_key)]
        data.loc[i:i, 'cust_id'] = flag1.cust_id.values
        data.loc[i:i, 'risk_rank'] = int(flag1.risk_rank)
        data.loc[i:i, 'occupation_code'] = int(flag1.occupation_code)
        data.loc[i:i, 'total_asset'] = int(flag1.total_asset)
        data.loc[i:i, 'AGE'] = int(flag1.AGE)

        flag2 = remit_group[remit_group['cust_id'] == data.loc[i:i, :].cust_id.values[0]]
        flag2 = flag2[flag2['date'] == int(data.loc[i:i, :].date)]
        if not flag2.empty:
            data.loc[i:i, 'trans_no1'] = int(flag2.trans_no1.values)
            data.loc[i:i, 'trans_no2'] = int(flag2.trans_no2.values)
            data.loc[i:i, 'trans_no3'] = int(flag2.trans_no3.values)
            data.loc[i:i, 'trans_no4'] = int(flag2.trans_no4.values)
            data.loc[i:i, 'trans_no5'] = int(flag2.trans_no5.values)
            data.loc[i:i, 'trade_amount_usd'] = int(flag2.trade_amount_usd.values)

    data.dropna(subset=['trade_amount_usd'], inplace=True)
    return data

#
#
data = remit_train_preprocessing(train_x_alert)
data.to_csv('remit_dataset.csv', index=False)
