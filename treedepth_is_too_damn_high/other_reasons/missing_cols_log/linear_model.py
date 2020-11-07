import pandas as pd
import numpy as np
import random

import xgboost as xgb

import plotly.graph_objects as go
import statsmodels.api as sm

from itertools import combinations

import gen

def display(preds):
 theList=[]
 for i in range(len(preds)):
  theList.append(preds[i])
  if i%4==3:
   print(theList)
   theList=[]


def linear_model(df, printem=False):
 df = df.copy()
 df["x0"]=1
 
 explanatories = [c for c in df.columns if c[0]=='x']
 
 linearModel = sm.GLM(df["y"], df[explanatories])
 linearModel = linearModel.fit()
 
 preds = np.array(linearModel.predict(df[explanatories]))
 acts = np.array(df["y"])
 
 if printem:
  display(preds)
  print("")
  display(acts)
  print("")
 
 errs = preds-acts
 
 MAE=sum(abs(errs))/len(errs)
 RMSE=np.sqrt(sum(errs*errs)/len(errs))
 
 return MAE,RMSE

def interact(df, explanatories, max_depth):
 for d in range(max_depth):
  combs = combinations(explanatories, d+1)
  for comb in combs:
   colName="".join(list(comb))
   col=df[list(comb)].product(axis=1)
   df[colName]=col
 return df

def linear_model_with_interxns(trainDf, testDf, max_depth=2, printem=False):
 
 trainDf = trainDf.copy()
 testDf = testDf.copy()
 
 explanatories = [c for c in trainDf.columns if c[0]=='x']
 
 trainDf = interact(trainDf, explanatories, max_depth)
 testDf = interact(testDf, explanatories, max_depth)
 
 trainDf["x0"]=1
 testDf["x0"]=1
 
 explanatories = [c for c in trainDf.columns if c[0]=='x']
 
 linearModel = sm.GLM(trainDf["y"], trainDf[explanatories], family=sm.families.Poisson())
 linearModel = linearModel.fit()
 
 preds = np.array(linearModel.predict(testDf[explanatories]))
 acts = np.array(testDf["y"])
 
 if printem:
  display(preds)
  print("")
  display(acts)
  print("")
 
 errs = preds-acts
 
 MAE=sum(abs(errs))/len(errs)
 RMSE=np.sqrt(sum(errs*errs)/len(errs))
 
 return MAE,RMSE

if __name__ == '__main__':
 df=gen.generate()
 print("MD1")
 print(linear_model(df))
 print("MD2")
 print(linear_model_with_interxns(df,df,2))
 print("MD3")
 print(linear_model_with_interxns(df,df,3))
 #print("MD4")
 #print(linear_model_with_interxns(df,4))
