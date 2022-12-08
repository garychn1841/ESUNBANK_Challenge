import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
pd.options.mode.chained_assignment = None  # default='warn'


#for Window
ccba = pd.read_csv('D:/PYTHON/Dataset/ESUNBANK_Challenge/public_train_x_ccba_full_hashed.csv')
train_x_alert = pd.read_csv('D:/PYTHON/Dataset/ESUNBANK_Challenge/train_x_alert_date.csv')
custinfo = pd.read_csv('D:/PYTHON/Dataset/ESUNBANK_Challenge/public_train_x_custinfo_full_hashed.csv')
answer = pd.read_csv('D:/PYTHON/Dataset/ESUNBANK_Challenge/train_y_answer.csv')
leaderboard_cctx_dataset = pd.read_csv('D:/PYTHON/Dataset/ESUNBANK_Challenge/public_x_alert_date.csv')

#for Mac
# ccba = pd.read_csv('/Users/huanghui-chu/Documents/python/dataset/ESUNBANK_Challenge/public_train_x_ccba_full_hashed.csv')
# train_x_alert = pd.read_csv('/Users/huanghui-chu/Documents/python/dataset/ESUNBANK_Challenge/train_x_alert_date.csv')
# custinfo = pd.read_csv('/Users/huanghui-chu/Documents/python/dataset/ESUNBANK_Challenge/public_train_x_custinfo_full_hashed.csv')
# answer = pd.read_csv('/Users/huanghui-chu/Documents/python/dataset/ESUNBANK_Challenge/train_y_answer.csv')
# leaderboard_dataset = pd.read_csv('/Users/huanghui-chu/Documents/python/dataset/ESUNBANK_Challenge/public_x_alert_date.csv')

def ccba_train_preprocessing(dataset):

    ccba.drop('byymm',axis = 1,inplace=True)

    ##groupby
    ccba_group = ccba.groupby('cust_id').mean().reset_index()


    answer.drop('alert_key',inplace=True,axis=1)
    data = pd.concat([dataset,answer],axis=1)


    #concat the cust_id&ccba in data
    a = custinfo.drop('alert_key',axis = 1)
    data[a.columns] = np.nan
    custinfo.occupation_code.fillna(21,inplace=True)

    b = ccba_group.drop(['cust_id'], axis = 1)
    data[b.columns] = np.nan



    for i in range(len(data)):
        print(i)
        flag1 = custinfo[custinfo['alert_key'] == int(data.loc[i:i, :].alert_key)]
        data.loc[i:i, 'cust_id'] = flag1.cust_id.values
        data.loc[i:i, 'risk_rank'] = int(flag1.risk_rank)
        data.loc[i:i, 'occupation_code'] = int(flag1.occupation_code)
        data.loc[i:i, 'total_asset'] = int(flag1.total_asset)
        data.loc[i:i, 'AGE'] = int(flag1.AGE)
    #
        flag2 = ccba_group[ccba_group['cust_id'] == data.loc[i:i, :].cust_id.values[0]]
        if not flag2.empty:
            data.loc[i:i, 'lupay'] = int(flag2.lupay.values)
            data.loc[i:i, 'cycam'] = int(flag2.cycam.values)
            data.loc[i:i, 'usgam'] = int(flag2.usgam.values)
            data.loc[i:i, 'clamt'] = int(flag2.clamt.values)
            data.loc[i:i, 'csamt'] = int(flag2.csamt.values)
            data.loc[i:i, 'inamt'] = int(flag2.inamt.values)
            data.loc[i:i, 'cucsm'] = int(flag2.cucsm.values)
            data.loc[i:i, 'cucah'] = int(flag2.cucah.values)

    #
    data.dropna(subset=['cucah'],inplace = True)
    # data.to_csv('data_out.csv',index=False)

    return data


def ccba_leaderboard_preprocessing(dataset):

    ccba.drop('byymm',axis = 1,inplace=True)

    ##groupby
    ccba_group = ccba.groupby('cust_id').mean().reset_index()


    answer.drop('alert_key',inplace=True,axis=1)
    data = dataset


    #concat the cust_id&ccba in data
    a = custinfo.drop('alert_key',axis = 1)
    data[a.columns] = np.nan
    custinfo.occupation_code.fillna(21,inplace=True)

    b = ccba_group.drop(['cust_id'], axis = 1)
    data[b.columns] = np.nan



    for i in range(len(data)):
        print(i)
        flag1 = custinfo[custinfo['alert_key'] == int(data.loc[i:i, :].alert_key)]
        data.loc[i:i, 'cust_id'] = flag1.cust_id.values
        data.loc[i:i, 'risk_rank'] = int(flag1.risk_rank)
        data.loc[i:i, 'occupation_code'] = int(flag1.occupation_code)
        data.loc[i:i, 'total_asset'] = int(flag1.total_asset)
        data.loc[i:i, 'AGE'] = int(flag1.AGE)
    #
        flag2 = ccba_group[ccba_group['cust_id'] == data.loc[i:i, :].cust_id.values[0]]
        if not flag2.empty:
            data.loc[i:i, 'lupay'] = int(flag2.lupay.values)
            data.loc[i:i, 'cycam'] = int(flag2.cycam.values)
            data.loc[i:i, 'usgam'] = int(flag2.usgam.values)
            data.loc[i:i, 'clamt'] = int(flag2.clamt.values)
            data.loc[i:i, 'csamt'] = int(flag2.csamt.values)
            data.loc[i:i, 'inamt'] = int(flag2.inamt.values)
            data.loc[i:i, 'cucsm'] = int(flag2.cucsm.values)
            data.loc[i:i, 'cucah'] = int(flag2.cucah.values)

    #
    data.dropna(subset=['cucah'],inplace = True)
    # data.to_csv('data_out.csv',index=False)

    return data



# data = cctx_train_preprocessing(train_x_alert)
data = ccba_leaderboard_preprocessing(leaderboard_cctx_dataset)
data.to_csv('ccba_public_dataset.csv',index=False)

