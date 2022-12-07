import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
pd.options.mode.chained_assignment = None  # default='warn'
pd.set_option('display.max_columns', 20)

#for window
# dp = pd.read_csv('D:/PYTHON/Dataset/ESUNBANK_Challenge/public_train_x_dp_full_hashed.csv')
# train_x_alert = pd.read_csv('D:/PYTHON/Dataset/ESUNBANK_Challenge/train_x_alert_date.csv')
# custinfo = pd.read_csv('D:/PYTHON/Dataset/ESUNBANK_Challenge/public_train_x_custinfo_full_hashed.csv')
# answer = pd.read_csv('D:/PYTHON/Dataset/ESUNBANK_Challenge/train_y_answer.csv')
# leaderboard_cctx_dataset = pd.read_csv('D:/PYTHON/Dataset/ESUNBANK_Challenge/public_x_alert_date.csv')

#for Mac
dp = pd.read_csv('/Users/huanghui-chu/Documents/python/dataset/ESUNBANK_Challenge/public_train_x_dp_full_hashed.csv')
train_x_alert = pd.read_csv('/Users/huanghui-chu/Documents/python/dataset/ESUNBANK_Challenge/train_x_alert_date.csv')
custinfo = pd.read_csv('/Users/huanghui-chu/Documents/python/dataset/ESUNBANK_Challenge/public_train_x_custinfo_full_hashed.csv')
answer = pd.read_csv('/Users/huanghui-chu/Documents/python/dataset/ESUNBANK_Challenge/train_y_answer.csv')
leaderboard_dataset = pd.read_csv('/Users/huanghui-chu/Documents/python/dataset/ESUNBANK_Challenge/public_x_alert_date.csv')


def dp_train_preprocessing(dataset):

    dp.drop(['tx_time','exchg_rate','fiscTxId','txbranch','info_asset_code'] , axis = 1 , inplace=True)

    #Feature_labeled
    encoder = LabelEncoder()
    encoder_Y = encoder.fit_transform(dp['debit_credit'])
    dp['debit_credit'] = encoder_Y
    # dp.to_csv('dp_out.csv',index=False)
    #cctx.to_csv('out.csv',index=False)


    # ##OneHotEncoder
    columnTransformer = ColumnTransformer([('encoder',OneHotEncoder(sparse=False),[1,3,5,6])],remainder='passthrough')
    dp_arr = np.array(columnTransformer.fit_transform(dp) , dtype= str)
    encoded_dp = pd.DataFrame(dp_arr)

    #
    #
    # ##reset columns
    encoded_dp.columns = ['debit','credit','tx_type1','tx_type2','tx_type3','n_cross_bank','cross_bank','n_ATM','ATM','cust_id','date','amt']
    encoded_dp = encoded_dp[['cust_id','date','debit','credit','tx_type1','tx_type2','tx_type3','n_cross_bank','cross_bank','n_ATM','ATM','amt']]
    # encoded_dp.to_csv('encoded_dp_out.csv', index=False)
    encoded_dp.amt[encoded_dp.amt == 'nan'] = 0
    # print(encoded_dp.isnull().any())
    # print(encoded_dp[encoded_dp.columns[encoded_dp.isnull().any()]])

    ##groupby
    encoded_dp.date = pd.to_numeric(encoded_dp.date)
    encoded_dp.debit = pd.to_numeric(encoded_dp.debit)
    encoded_dp.credit = pd.to_numeric(encoded_dp.credit)
    encoded_dp.tx_type1 = pd.to_numeric(encoded_dp.tx_type1)
    encoded_dp.tx_type2 = pd.to_numeric(encoded_dp.tx_type2)
    encoded_dp.tx_type3 = pd.to_numeric(encoded_dp.tx_type3)
    encoded_dp.n_cross_bank = pd.to_numeric(encoded_dp.n_cross_bank)
    encoded_dp.cross_bank = pd.to_numeric(encoded_dp.cross_bank)
    encoded_dp.n_ATM = pd.to_numeric(encoded_dp.n_ATM)
    encoded_dp.ATM = pd.to_numeric(encoded_dp.ATM)
    encoded_dp.amt = pd.to_numeric(encoded_dp.amt)
    #
    dp_group = encoded_dp.groupby(['cust_id','date']).sum().reset_index()
    dp_group = dp_group.sort_values(['cust_id','date'])
    # dp_group.to_csv('dp_out.csv',index=False)
    #
    #
    answer.drop('alert_key',inplace=True,axis=1)
    data = pd.concat([dataset,answer],axis=1)
    #
    #
    # ##concat the cust_id&cctx in data
    a = custinfo.drop('alert_key',axis = 1)
    data[a.columns] = np.nan
    custinfo.occupation_code.fillna(21,inplace=True)
    b = dp_group.drop(['cust_id','date'], axis = 1)
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



        flag2 = dp_group[dp_group['cust_id'] == data.loc[i:i, :].cust_id.values[0]]
        flag2 = flag2[flag2['date'] == int(data.loc[i:i, :].date)]
        if not flag2.empty:
            data.loc[i:i, 'debit'] = int(flag2.debit.values)
            data.loc[i:i, 'credit'] = int(flag2.credit.values)
            data.loc[i:i, 'tx_type1'] = int(flag2.tx_type1.values)
            data.loc[i:i, 'tx_type2'] = int(flag2.tx_type2.values)
            data.loc[i:i, 'tx_type3'] = int(flag2.tx_type3.values)
            data.loc[i:i, 'n_cross_bank'] = int(flag2.n_cross_bank.values)
            data.loc[i:i, 'cross_bank'] = int(flag2.cross_bank.values)
            data.loc[i:i, 'n_ATM'] = int(flag2.n_ATM.values)
            data.loc[i:i, 'ATM'] = int(flag2.ATM.values)
            data.loc[i:i, 'amt'] = int(flag2.amt.values)


    data.dropna(subset=['amt'], inplace=True)
    return data
    # print(data)
    # data.to_csv('data_out.csv',index=False)

def dp_leaderboard_preprocessing(dataset):

    dp.drop(['tx_time','exchg_rate','fiscTxId','txbranch','info_asset_code'] , axis = 1 , inplace=True)

    #Feature_labeled
    encoder = LabelEncoder()
    encoder_Y = encoder.fit_transform(dp['debit_credit'])
    dp['debit_credit'] = encoder_Y
    # dp.to_csv('dp_out.csv',index=False)
    #cctx.to_csv('out.csv',index=False)


    # ##OneHotEncoder
    columnTransformer = ColumnTransformer([('encoder',OneHotEncoder(sparse=False),[1,3,5,6])],remainder='passthrough')
    dp_arr = np.array(columnTransformer.fit_transform(dp) , dtype= str)
    encoded_dp = pd.DataFrame(dp_arr)

    #
    #
    # ##reset columns
    encoded_dp.columns = ['debit','credit','tx_type1','tx_type2','tx_type3','n_cross_bank','cross_bank','n_ATM','ATM','cust_id','date','amt']
    encoded_dp = encoded_dp[['cust_id','date','debit','credit','tx_type1','tx_type2','tx_type3','n_cross_bank','cross_bank','n_ATM','ATM','amt']]
    # encoded_dp.to_csv('encoded_dp_out.csv', index=False)
    encoded_dp.amt[encoded_dp.amt == 'nan'] = 0
    # print(encoded_dp.isnull().any())
    # print(encoded_dp[encoded_dp.columns[encoded_dp.isnull().any()]])

    ##groupby
    encoded_dp.date = pd.to_numeric(encoded_dp.date)
    encoded_dp.debit = pd.to_numeric(encoded_dp.debit)
    encoded_dp.credit = pd.to_numeric(encoded_dp.credit)
    encoded_dp.tx_type1 = pd.to_numeric(encoded_dp.tx_type1)
    encoded_dp.tx_type2 = pd.to_numeric(encoded_dp.tx_type2)
    encoded_dp.tx_type3 = pd.to_numeric(encoded_dp.tx_type3)
    encoded_dp.n_cross_bank = pd.to_numeric(encoded_dp.n_cross_bank)
    encoded_dp.cross_bank = pd.to_numeric(encoded_dp.cross_bank)
    encoded_dp.n_ATM = pd.to_numeric(encoded_dp.n_ATM)
    encoded_dp.ATM = pd.to_numeric(encoded_dp.ATM)
    encoded_dp.amt = pd.to_numeric(encoded_dp.amt)

    dp_group = encoded_dp.groupby(['cust_id','date']).sum().reset_index()
    dp_group = dp_group.sort_values(['cust_id','date'])
    # dp_group.to_csv('dp_out.csv',index=False)


    # answer.drop('alert_key',inplace=True,axis=1)
    # data = pd.concat([dataset,answer],axis=1)


    data = dataset

    # ##concat the cust_id&cctx in data
    a = custinfo.drop('alert_key',axis = 1)
    data[a.columns] = np.nan
    custinfo.occupation_code.fillna(21,inplace=True)
    b = dp_group.drop(['cust_id','date'], axis = 1)
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



        flag2 = dp_group[dp_group['cust_id'] == data.loc[i:i, :].cust_id.values[0]]
        flag2 = flag2[flag2['date'] == int(data.loc[i:i, :].date)]
        if not flag2.empty:
            data.loc[i:i, 'debit'] = int(flag2.debit.values)
            data.loc[i:i, 'credit'] = int(flag2.credit.values)
            data.loc[i:i, 'tx_type1'] = int(flag2.tx_type1.values)
            data.loc[i:i, 'tx_type2'] = int(flag2.tx_type2.values)
            data.loc[i:i, 'tx_type3'] = int(flag2.tx_type3.values)
            data.loc[i:i, 'n_cross_bank'] = int(flag2.n_cross_bank.values)
            data.loc[i:i, 'cross_bank'] = int(flag2.cross_bank.values)
            data.loc[i:i, 'n_ATM'] = int(flag2.n_ATM.values)
            data.loc[i:i, 'ATM'] = int(flag2.ATM.values)
            data.loc[i:i, 'amt'] = int(flag2.amt.values)


    data.dropna(subset=['amt'], inplace=True)
    return data
    # print(data)
    # data.to_csv('data_out.csv',index=False)



# data = dp_train_preprocessing(train_x_alert)
data = dp_leaderboard_preprocessing(leaderboard_dataset)
data.to_csv('dp_public_dataset.csv', index = False)




#