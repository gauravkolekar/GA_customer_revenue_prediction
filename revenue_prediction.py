#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 29 18:30:56 2018

@author: gaurav
"""
# importing important libraries
import os
import json
import pandas as pd
from pandas.io.json import json_normalize
from sklearn.ensemble import RandomForestRegressor

def load_df(csv_path='data/train.csv', nrows=None):
    JSON_COLUMNS = ['device', 'geoNetwork', 'totals', 'trafficSource']
    
    df = pd.read_csv(csv_path, 
                     converters={column: json.loads for column in JSON_COLUMNS}, 
                     dtype={'fullVisitorId': 'str'}, # Important!!
                     nrows=nrows)
    
    for column in JSON_COLUMNS:
        column_as_df = json_normalize(df[column])
        column_as_df.columns = [f"{column}.{subcolumn}" for subcolumn in column_as_df.columns]
        df = df.drop(column, axis=1).merge(column_as_df, right_index=True, left_index=True)
    print(f"Loaded {os.path.basename(csv_path)}. Shape: {df.shape}")
    return df

# raw_df = load_df(nrows=100)
# raw_df = load_df()
# os.makedirs('tmp', exist_ok=True)
#raw_df.to_feather('tmp/revenue_raw')

raw_df = pd.read_feather('tmp/revenue_raw')

def convert_to_number(df=None, col_names=[]):
    for col in col_names:
        df[col] = pd.to_numeric(df[col])
    return df

def convert_to_boolean(df=None, col_names=[]):
    for col in col_names:
        df[col] = df[col].astype(bool)
    return df

def convert_to_categorical(df=None):
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].astype('category')
            df[col] = df[col].cat.codes
    return df

# preprocessing

raw_df = convert_to_number(df=raw_df, col_names=['totals.bounces', 
                                                 'totals.hits', 
                                                 'totals.newVisits',
                                                 'totals.pageviews',
                                                 'totals.transactionRevenue',
                                                 'totals.visits',
                                                 'trafficSource.adwordsClickInfo.page'])

raw_df = convert_to_boolean(df=raw_df, col_names=['trafficSource.adwordsClickInfo.isVideoAd'])

raw_df = convert_to_categorical(raw_df)
