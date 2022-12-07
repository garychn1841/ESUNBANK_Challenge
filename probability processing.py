import pandas
import pandas as pd

#read data
cctx_prob = pd.read_csv('cctx_public_proba.csv')
dp_prob = pd.read_csv('dp_public_proba.csv')
remit_prob = pd.read_csv('remit_public_proba.csv')
public_alert = pd.read_csv('/Users/huanghui-chu/Documents/python/dataset/ESUNBANK_Challenge/public_x_alert_date.csv')

print(public_alert)