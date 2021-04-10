import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity

import warnings
warnings.filterwarnings(action="ignore")

import os


class ScoreCalulator():
  
    def __init__(self, path, NUM_SECTORS= 10, NUM_EXPENSES=6):
        self.NUM_SECTORS = NUM_SECTORS
        self.NUM_EXPENSES = NUM_EXPENSES
        self.data= pd.read_csv(path)
        DEFAULT_SCORE = 100
        if 'SCORE' not in self.data.columns :
            self.data['SCORE'] = [DEFAULT_SCORE]*len(self.data)
            self.data.to_csv(path)
            print("Score column added")
        try :
            self.SCORE_DATA  = pd.read_csv('ScoreData.csv')
        except:
            self.SCORE_DATA = self.CreateScoretable()


    def CreateScoretable(self):
        SCORE_DATA = pd.DataFrame()
        SCORE_DATA['CUST_ID'] = self.data['CUST_ID']
        SCORE_DATA[['SEC_WT' + str(i) for i in range(self.NUM_SECTORS)] ]  = np.random.randint(0,100, size = (len(self.data), self.NUM_SECTORS))
        SCORE_DATA[['SEC_RAT' + str(i) for i in range(self.NUM_SECTORS)] ]  = np.random.uniform(0,1, size = (len(self.data), self.NUM_SECTORS))
        SCORE_DATA[['EXP_WT' + str(i) for i in range(self.NUM_EXPENSES)] ]  = np.random.randint(0,100, size = (len(self.data), self.NUM_EXPENSES))
        SCORE_DATA[['EXP_RAT' + str(i) for i in range(self.NUM_EXPENSES)] ]  = np.random.uniform(0,1, size = (len(self.data), self.NUM_EXPENSES))
        return SCORE_DATA

    def Score(self, sec_wt,  sec_rat, exp_w, exp_rat):
        if sec_wt.size != sec_rat.size or exp_w.size != exp_rat.size:
            raise "Dimensions not matcing"
        return np.sum(sec_wt * sec_rat) + np.sum(exp_w*exp_rat)

    def ScoreCustomer(self, CUST_ID):
        row = self.SCORE_DATA[self.SCORE_DATA['CUST_ID'] == CUST_ID] 
        sec_wt = row[['SEC_WT' + str(i) for i in range(self.NUM_SECTORS)] ].values
        sec_rat = row[['SEC_RAT' + str(i) for i in range(self.NUM_SECTORS)]].values
        exp_w  = row[['EXP_WT' + str(i) for i in range(self.NUM_EXPENSES)] ].values
        exp_rat = row[['EXP_RAT' + str(i) for i in range(self.NUM_EXPENSES)]].values
        return self.Score(sec_wt,  sec_rat, exp_w, exp_rat)
    
    def CalculateAllScores(self):
        self.data['SCORE'] = self.data['CUST_ID'].map(self.ScoreCustomer)
        MAX = np.max(self.data['SCORE'])
        self.data['SCORE']  = self.data['SCORE']/MAX*100
        self.data.drop(['Unnamed: 0'], axis = 1, inplace = True)
        return self.data
        
    def WriteToFile(self):
        self.SCORE_DATA.to_csv('ScoreData.csv')
    
    def GetScoreDataFrame(self):
        return self.SCORE_DATA

    def preprocess(self, normalize=True, save=None):
        columns=['BALANCE', 'PURCHASES', 'ONEOFF_PURCHASES', 'INSTALLMENTS_PURCHASES', 'CASH_ADVANCE', 'CREDIT_LIMIT',
            'PAYMENTS', 'MINIMUM_PAYMENTS']
        for c in columns:
            Range=c+'_RANGE'
            self.data[Range]=0        
            self.data.loc[((self.data[c]>0)&(self.data[c]<=500)),Range]=1
            self.data.loc[((self.data[c]>500)&(self.data[c]<=1000)),Range]=2
            self.data.loc[((self.data[c]>1000)&(self.data[c]<=3000)),Range]=3
            self.data.loc[((self.data[c]>3000)&(self.data[c]<=5000)),Range]=4
            self.data.loc[((self.data[c]>5000)&(self.data[c]<=10000)),Range]=5
            self.data.loc[((self.data[c]>10000)),Range]=6
        columns=['BALANCE_FREQUENCY', 'PURCHASES_FREQUENCY', 'ONEOFF_PURCHASES_FREQUENCY', 'PURCHASES_INSTALLMENTS_FREQUENCY', 
            'CASH_ADVANCE_FREQUENCY', 'PRC_FULL_PAYMENT']
        for c in columns:  
            Range=c+'_RANGE'
            self.data[Range]=0
            self.data.loc[((self.data[c]>0)&(self.data[c]<=0.1)),Range]=1
            self.data.loc[((self.data[c]>0.1)&(self.data[c]<=0.2)),Range]=2
            self.data.loc[((self.data[c]>0.2)&(self.data[c]<=0.3)),Range]=3
            self.data.loc[((self.data[c]>0.3)&(self.data[c]<=0.4)),Range]=4
            self.data.loc[((self.data[c]>0.4)&(self.data[c]<=0.5)),Range]=5
            self.data.loc[((self.data[c]>0.5)&(self.data[c]<=0.6)),Range]=6
            self.data.loc[((self.data[c]>0.6)&(self.data[c]<=0.7)),Range]=7
            self.data.loc[((self.data[c]>0.7)&(self.data[c]<=0.8)),Range]=8
            self.data.loc[((self.data[c]>0.8)&(self.data[c]<=0.9)),Range]=9
            self.data.loc[((self.data[c]>0.9)&(self.data[c]<=1.0)),Range]=10
        columns=['PURCHASES_TRX', 'CASH_ADVANCE_TRX']  
        for c in columns:  
            Range=c+'_RANGE'
            self.data[Range]=0
            self.data.loc[((self.data[c]>0)&(self.data[c]<=5)),Range]=1
            self.data.loc[((self.data[c]>5)&(self.data[c]<=10)),Range]=2
            self.data.loc[((self.data[c]>10)&(self.data[c]<=15)),Range]=3
            self.data.loc[((self.data[c]>15)&(self.data[c]<=20)),Range]=4
            self.data.loc[((self.data[c]>20)&(self.data[c]<=30)),Range]=5
            self.data.loc[((self.data[c]>30)&(self.data[c]<=50)),Range]=6
            self.data.loc[((self.data[c]>50)&(self.data[c]<=100)),Range]=7
            self.data.loc[((self.data[c]>100)),Range]=8
        columns=['SCORE']  
        for c in columns:
            Range=c+'_RANGE'
            self.data[Range]=0
            self.data.loc[((self.data[c]>0)&(self.data[c]<=10)),Range]=1
            self.data.loc[((self.data[c]>10)&(self.data[c]<=20)),Range]=2
            self.data.loc[((self.data[c]>20)&(self.data[c]<=30)),Range]=3
            self.data.loc[((self.data[c]>30)&(self.data[c]<=40)),Range]=4
            self.data.loc[((self.data[c]>40)&(self.data[c]<=50)),Range]=5
            self.data.loc[((self.data[c]>50)&(self.data[c]<=60)),Range]=6
            self.data.loc[((self.data[c]>60)&(self.data[c]<=70)),Range]=7
            self.data.loc[((self.data[c]>70)&(self.data[c]<=80)),Range]=8
            self.data.loc[((self.data[c]>80)&(self.data[c]<=90)),Range]=9
            self.data.loc[((self.data[c]>90)&(self.data[c]<=100)),Range]=10
        self.data.drop(['BALANCE', 'BALANCE_FREQUENCY', 'PURCHASES',
          'ONEOFF_PURCHASES', 'INSTALLMENTS_PURCHASES', 'CASH_ADVANCE',
          'PURCHASES_FREQUENCY',  'ONEOFF_PURCHASES_FREQUENCY',
          'PURCHASES_INSTALLMENTS_FREQUENCY', 'CASH_ADVANCE_FREQUENCY',
          'CASH_ADVANCE_TRX', 'PURCHASES_TRX', 'CREDIT_LIMIT', 'PAYMENTS',
          'MINIMUM_PAYMENTS', 'PRC_FULL_PAYMENT', 'SCORE' ], axis=1, inplace=True)
        if save is not None:
          self.data.to_csv(save+'processed_data_with_score.csv', index=False)
        self.data.drop(['CUST_ID'], axis=1, inplace=True)
        self.X= np.asarray(self.data)
        if normalize==True:
          scale = StandardScaler()
          self.X = scale.fit_transform(self.X)