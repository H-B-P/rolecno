import pandas as pd
import numpy as np
import random

import xgboost as xgb

import plotly.graph_objects as go

import gen

def display(preds):
 theList=[]
 for i in range(len(preds)):
  theList.append(preds[i])
  if i%4==3:
   print(theList)
   theList=[]

def model(df, params={'max_depth':1}, rounds=10, printem=False):
 params["base_score"] = sum(df["y"])/len(df["y"])
 
 explanatories = [c for c in df.columns if c[0]=='x']
 
 dtrain = xgb.DMatrix(df[explanatories], label=df["y"], weight=df["w"])
 
 bst = xgb.train(params, dtrain, rounds)
 
 preds = np.array(bst.predict(dtrain))
 #acts = np.array(df["y"])
 acts = np.array(dtrain.get_label())
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
 print("TD1")
 print(model(df,{'max_depth':1,'learning_rate':0.3}, rounds=100))
 print("TD2")
 print(model(df,{'max_depth':2,'learning_rate':0.3}, rounds=100))
 print("TD3")
 print(model(df,{'max_depth':3,'learning_rate':0.3}, rounds=100))
 print("TD4")
 print(model(df,{'max_depth':4,'learning_rate':0.3}, rounds=100))
 print("TD4, 2x rounds")
 print(model(df,{'max_depth':4,'learning_rate':0.3}, rounds=200))
 print("TD5")
 print(model(df,{'max_depth':5,'learning_rate':0.3}, rounds=100))
