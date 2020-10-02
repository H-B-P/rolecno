import pandas as pd
import numpy as np
import random

import xgboost as xgb

import plotly.graph_objects as go

import gen

def model(trainDf, testDf, params={'max_depth':1}, rounds=10, roundsAtATime=1, modelDefRes=0.01, show=False):
 params["base_score"] = sum(trainDf["y"])/len(trainDf["y"])
 
 dtrain = xgb.DMatrix(trainDf[["x"]], label=trainDf["y"])
 dtest = xgb.DMatrix(testDf[["x"]], label=testDf["y"])
 
 xCol = list(np.arange(min(trainDf["x"]), max(trainDf["x"]), modelDefRes))
 modelDefDf = pd.DataFrame({"x":xCol})
 ddef = xgb.DMatrix(modelDefDf)
 
 bst = xgb.train(params, dtrain, min(roundsAtATime, rounds))
 roundsToGo=rounds-roundsAtATime
 while roundsToGo>0:
  bst = xgb.train(params, dtrain, min(roundsAtATime, roundsToGo), xgb_model=bst)
  defPreds = bst.predict(ddef)
  if show:
   fig = go.Figure()
   fig.add_trace(go.Scatter(x=trainDf["x"], y=trainDf["y"],
                     mode='markers',
                     name='data'))
   fig.add_trace(go.Scatter(x=modelDefDf["x"], y=defPreds,
                     mode='lines',
                     name='model'))
   fig.show()
  roundsToGo=roundsToGo-roundsAtATime
  
 
 preds = np.array(bst.predict(dtest))
 acts = np.array(testDf["y"])
 errs = preds-acts
 
 MAE=sum(abs(errs))/len(errs)
 RMSE=np.sqrt(sum(errs*errs)/len(errs))
 
 return MAE,RMSE

if __name__ == '__main__':
 df1=gen.generate(2,20,2)
 df2=gen.generate(2,20,2)
 print(model(df1,df2,{'max_depth':1,'learning_rate':0.3}, rounds=50, roundsAtATime=10, show=True))
