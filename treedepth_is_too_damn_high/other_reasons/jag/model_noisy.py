import pandas as pd
import numpy as np
import random

import xgboost as xgb

import plotly.graph_objects as go

import gen

def model(trainDf, testDf, params={'max_depth':1}, rounds=10):
 params["base_score"] = sum(trainDf["y"])/len(trainDf["y"])
 
 explanatories = ["x"]+[c for c in trainDf.columns if c[0]=='d']
 
 dtrain = xgb.DMatrix(trainDf[explanatories], label=trainDf["y"])
 dtest = xgb.DMatrix(testDf[explanatories], label=testDf["y"])
 
 bst = xgb.train(params, dtrain, rounds)
 
 preds = np.array(bst.predict(dtest))
 acts = np.array(testDf["y"])
 errs = preds-acts
 
 MAE=sum(abs(errs))/len(errs)
 RMSE=np.sqrt(sum(errs*errs)/len(errs))
 
 return MAE,RMSE

if __name__ == '__main__':
 df1=gen.generateWithDecoys(0,3,3,10, 1)
 df2=gen.generateWithDecoys(0,3,3,10, 1)
 print(model(df1,df2,{'max_depth':1,'learning_rate':0.5}, rounds=1000))
 print(model(df1,df2,{'max_depth':1,'learning_rate':0.5}, rounds=2000))
 print(model(df1,df2,{'max_depth':2,'learning_rate':0.5}, rounds=1000))
