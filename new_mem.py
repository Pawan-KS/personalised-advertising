import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import NearestNeighbors

import warnings
warnings.filterwarnings(action="ignore")

import os



class member_updater():

  def access_proc_data(self, path):
    self.procdf = pd.read_csv(path)
    return self.procdf.head()
  
  def access_form_data(self, path):
    self.formdf = pd.read_csv(path)
    return self.formdf.head()

  def train(self):
    self.data = self.formdf.copy()
    self.data.workclass = pd.factorize(self.data.workclass)[0]
    self.data.education = pd.factorize(self.data.education)[0]
    self.data['marital-status'] = pd.factorize(self.data['marital-status'])[0]
    self.data.occupation = pd.factorize(self.data.occupation)[0]
    self.data.gender = pd.factorize(self.data.gender)[0]
    self.data['native-country'] = pd.factorize(self.data['native-country'])[0]
    self.data.city = pd.factorize(self.data.city)[0]
    
    X = np.array(self.data)[:,1:]
    self.nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(X)

  def predict(self, test):
    test.workclass = pd.factorize(test.workclass)[0]
    test.education = pd.factorize(test.education)[0]
    test['marital-status'] = pd.factorize(test['marital-status'])[0]
    test.occupation = pd.factorize(test.occupation)[0]
    test.gender = pd.factorize(test.gender)[0]
    test['native-country'] = pd.factorize(test['native-country'])[0]
    test.city = pd.factorize(test.city)[0]
    distances, indices = self.nbrs.kneighbors(test.iloc[:,1:])
    self.dff = pd.DataFrame()
    for x in indices:
      self.dff = pd.concat([self.dff, self.procdf.iloc[x,1:]], axis=0)
    self.dff = pd.concat([pd.DataFrame(test.iloc[:,0]),self.dff.reset_index().drop(['index'],axis=1)], axis=1)
    return self.dff

  def update(self, save_filepath):
    newdf = pd.concat([self.procdf, self.dff], axis=0)
    newdf.to_csv(save_filepath, index=None)
 
