import pandas as pd
import numpy as np
import random

import xgboost as xgb

import plotly.graph_objects as go

import canonical_mixes_subpopulations


import gen_efficient_unbalanced as gen

def model(trainDf, testDf, params={'max_depth':1}, rounds=10):
 params["base_score"] = sum(trainDf["y"])/len(trainDf["y"])
 
 explanatories = [c for c in trainDf.columns if c[0]=='x']
 
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
 df1=gen.generateIII(1, 100000, False, 2.0)
 df2=gen.generateIII(1, 100000, False, 2.0)
 print("Mislinked TD1")
 print(model(df1,df2,{'max_depth':1,'learning_rate':0.3}, rounds=1000))
 print("TD1")
 print(model(df1,df2,{'max_depth':1,'learning_rate':0.3, 'objective':'count:poisson'}, rounds=1000))
 print("TD1 (but 2x rounds)")
 print(model(df1,df2,{'max_depth':1,'learning_rate':0.3, 'objective':'count:poisson'}, rounds=2000))
 print("TD2")
 print(model(df1,df2,{'max_depth':2,'learning_rate':0.3, 'objective':'count:poisson'}, rounds=1000))
 print("TD3")
 print(model(df1,df2,{'max_depth':3,'learning_rate':0.3, 'objective':'count:poisson'}, rounds=1000))
 print("TD4")
 print(model(df1,df2,{'max_depth':4,'learning_rate':0.3, 'objective':'count:poisson'}, rounds=1000))
 print("canon 1")
 print(canonical_mixes_subpopulations.canonically_model(df1,df2, 1000, 0.1, 0))
 print("canon 2")
 print(canonical_mixes_subpopulations.canonically_model(df1,df2, 2000, 0.1, 0))
