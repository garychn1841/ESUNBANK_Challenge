import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
pd.options.mode.chained_assignment = None  # default='warn'


#for Window
# cctx = pd.read_csv('D:/PYTHON/Dataset/ESUNBANK_Challenge/public_train_x_cdtx0001_full_hashed.csv')
# train_x_alert = pd.read_csv('D:/PYTHON/Dataset/ESUNBANK_Challenge/train_x_alert_date.csv')
# custinfo = pd.read_csv('D:/PYTHON/Dataset/ESUNBANK_Challenge/public_train_x_custinfo_full_hashed.csv')
# answer = pd.read_csv('D:/PYTHON/Dataset/ESUNBANK_Challenge/train_y_answer.csv')
# leaderboard_cctx_dataset = pd.read_csv('D:/PYTHON/Dataset/ESUNBANK_Challenge/public_x_alert_date.csv')

#for Mac
cctx = pd.read_csv('/Users/huanghui-chu/Documents/python/dataset/ESUNBANK_Challenge/public_train_x_cdtx0001_full_hashed.csv')
train_x_alert = pd.read_csv('/Users/huanghui-chu/Documents/python/dataset/ESUNBANK_Challenge/train_x_alert_date.csv')
custinfo = pd.read_csv('/Users/huanghui-chu/Documents/python/dataset/ESUNBANK_Challenge/public_train_x_custinfo_full_hashed.csv')
answer = pd.read_csv('/Users/huanghui-chu/Documents/python/dataset/ESUNBANK_Challenge/train_y_answer.csv')
leaderboard_dataset = pd.read_csv('/Users/huanghui-chu/Documents/python/dataset/ESUNBANK_Challenge/public_x_alert_date.csv')

def cctx_train_preprocessing(dataset):

    #Feature_labeled
    cctx.country[cctx['country'] == 130] = 0
    cctx.country[cctx['country'] != 0] = 1
    cctx.cur_type[cctx['cur_type'] == 47] = 111
    cctx.cur_type[cctx['cur_type'] != 111] = 222
    #cctx.to_csv('out.csv',index=False)


    ##OneHotEncoder
    columnTransformer = ColumnTransformer([('encoder',OneHotEncoder(sparse=False),[2,3])],remainder='passthrough')
    country_arr = np.array(columnTransformer.fit_transform(cctx) , dtype= str)
    encoded_cctx = pd.DataFrame(country_arr)
    # print(encoded_cctx)


    ##reset columns
    encoded_cctx.columns = ['tw','n_tw','ntd','n_ntd','cust_id','date','amt']
    encoded_cctx = encoded_cctx[['cust_id','date','tw','n_tw','ntd','n_ntd','amt']]

    ##groupby
    encoded_cctx.tw = pd.to_numeric(encoded_cctx.tw)
    encoded_cctx.n_tw = pd.to_numeric(encoded_cctx.n_tw)
    encoded_cctx.ntd = pd.to_numeric(encoded_cctx.ntd)
    encoded_cctx.n_ntd = pd.to_numeric(encoded_cctx.n_ntd)
    encoded_cctx.amt = pd.to_numeric(encoded_cctx.amt)
    encoded_cctx.date = pd.to_numeric(encoded_cctx.date)

    cctx_group = encoded_cctx.groupby(['cust_id','date']).sum().reset_index()
    cctx_group = cctx_group.sort_values(['cust_id','date'])
    # cctx_group.to_csv('out.csv',index=False)


    answer.drop('alert_key',inplace=True,axis=1)
    data = pd.concat([dataset,answer],axis=1)
    # print(data)


    ##concat the cust_id&cctx in data
    a = custinfo.drop('alert_key',axis = 1)
    data[a.columns] = np.nan
    custinfo.occupation_code.fillna(21,inplace=True)

    b = cctx_group.drop(['cust_id','date'], axis = 1)
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
        flag2 = cctx_group[cctx_group['cust_id'] == data.loc[i:i, :].cust_id.values[0]]
        flag2 = flag2[flag2['date'] == int(data.loc[i:i, :].date)]
        if not flag2.empty:
            data.loc[i:i, 'tw'] = int(flag2.tw.values)
            data.loc[i:i, 'n_tw'] = int(flag2.n_tw.values)
            data.loc[i:i, 'ntd'] = int(flag2.ntd.values)
            data.loc[i:i, 'n_ntd'] = int(flag2.n_ntd.values)
            data.loc[i:i, 'amt'] = int(flag2.amt.values)

    #
    data.dropna(subset=['amt'],inplace = True)
    # data.to_csv('data_out.csv',index=False)


    data.dropna(subset=['amt'],inplace = True)
    return data


def cctx_leaderboard_preprocessing(dataset):

    #Feature_labeled
    cctx.country[cctx['country'] == 130] = 0
    cctx.country[cctx['country'] != 0] = 1
    cctx.cur_type[cctx['cur_type'] == 47] = 111
    cctx.cur_type[cctx['cur_type'] != 111] = 222
    #cctx.to_csv('out.csv',index=False)


    ##OneHotEncoder
    columnTransformer = ColumnTransformer([('encoder',OneHotEncoder(sparse=False),[2,3])],remainder='passthrough')
    country_arr = np.array(columnTransformer.fit_transform(cctx) , dtype= str)
    encoded_cctx = pd.DataFrame(country_arr)
    # print(encoded_cctx)


    ##reset columns
    encoded_cctx.columns = ['tw','n_tw','ntd','n_ntd','cust_id','date','amt']
    encoded_cctx = encoded_cctx[['cust_id','date','tw','n_tw','ntd','n_ntd','amt']]

    ##groupby
    encoded_cctx.tw = pd.to_numeric(encoded_cctx.tw)
    encoded_cctx.n_tw = pd.to_numeric(encoded_cctx.n_tw)
    encoded_cctx.ntd = pd.to_numeric(encoded_cctx.ntd)
    encoded_cctx.n_ntd = pd.to_numeric(encoded_cctx.n_ntd)
    encoded_cctx.amt = pd.to_numeric(encoded_cctx.amt)
    encoded_cctx.date = pd.to_numeric(encoded_cctx.date)

    cctx_group = encoded_cctx.groupby(['cust_id','date']).sum().reset_index()
    cctx_group = cctx_group.sort_values(['cust_id','date'])
    # cctx_group.to_csv('out.csv',index=False)


    # answer.drop('alert_key',inplace=True,axis=1)
    # data = pd.concat([dataset,answer],axis=1)
    # print(data)

    data = dataset


    ##concat the cust_id&cctx in data
    a = custinfo.drop('alert_key',axis = 1)
    data[a.columns] = np.nan
    custinfo.occupation_code.fillna(21,inplace=True)

    b = cctx_group.drop(['cust_id','date'], axis = 1)
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
        flag2 = cctx_group[cctx_group['cust_id'] == data.loc[i:i, :].cust_id.values[0]]
        flag2 = flag2[flag2['date'] == int(data.loc[i:i, :].date)]
        if not flag2.empty:
            data.loc[i:i, 'tw'] = int(flag2.tw.values)
            data.loc[i:i, 'n_tw'] = int(flag2.n_tw.values)
            data.loc[i:i, 'ntd'] = int(flag2.ntd.values)
            data.loc[i:i, 'n_ntd'] = int(flag2.n_ntd.values)
            data.loc[i:i, 'amt'] = int(flag2.amt.values)

    #
    data.dropna(subset=['amt'],inplace = True)
    # data.to_csv('data_out.csv',index=False)


    data.dropna(subset=['amt'],inplace = True)
    return data



# data = cctx_train_preprocessing(train_x_alert)
data = cctx_leaderboard_preprocessing(leaderboard_dataset)
data.to_csv('cctx_public_dataset.csv',index=False)

