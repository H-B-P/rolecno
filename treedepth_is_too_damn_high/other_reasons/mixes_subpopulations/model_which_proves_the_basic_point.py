import pandas as pd
import numpy as np
import random

import xgboost as xgb

import plotly.graph_objects as go

import gen

def model(trainDf, testDf, params={'max_depth':1}, rounds=10):
 params["base_score"] = sum(trainDf["y"])/len(trainDf["y"])
 
 dtrain = xgb.DMatrix(trainDf[["x1", "x2", "x3", "x4"]], label=trainDf["y"])
 dtest = xgb.DMatrix(testDf[["x1", "x2", "x3", "x4"]], label=testDf["y"])
 
 bst = xgb.train(params, dtrain, rounds)
 
 preds = np.array(bst.predict(dtest))
 acts = np.array(testDf["y"])
 errs = preds-acts
 
 MAE=sum(abs(errs))/len(errs)
 RMSE=np.sqrt(sum(errs*errs)/len(errs))
 
 return MAE,RMSE

if __name__ == '__main__':
 df1=gen.generate(100, 4000, False)
 df2=gen.generate(100, 4000, False)
 print("Mislinked TD1")
 print(model(df1,df2,{'max_depth':1,'learning_rate':0.3}, rounds=100))
 print("TD1")
 print(model(df1,df2,{'max_depth':1,'learning_rate':0.3, 'objective':'count:poisson'}, rounds=100))
 print("TD1 (but 2x rounds)")
 print(model(df1,df2,{'max_depth':1,'learning_rate':0.3, 'objective':'count:poisson'}, rounds=200))
 print("TD2")
 print(model(df1,df2,{'max_depth':2,'learning_rate':0.3, 'objective':'count:poisson'}, rounds=100))
 print("TD3")
 print(model(df1,df2,{'max_depth':3,'learning_rate':0.3, 'objective':'count:poisson'}, rounds=100))
