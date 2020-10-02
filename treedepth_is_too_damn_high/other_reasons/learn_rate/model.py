import pandas as pd
import numpy as np
import random

import xgboost as xgb

import gen

def model_and_get_stats(trainDf, testDf, params={'max_depth':2, "n_rounds":5}):
 explanatoryCols = list(trainDf.columns)
 explanatoryCols.remove("y")
 dtrain = xgb.DMatrix(trainDf[explanatoryCols], label=trainDf["y"])
 dtest = xgb.DMatrix(testDf[explanatoryCols], label=testDf["y"])
 
 bst = xgb.train(params, dtrain, params["n_rounds"])
 
 preds = np.array(bst.predict(dtest))
 acts = np.array(testDf["y"])
 errs = preds-acts
 
 MAE=sum(abs(errs))/len(errs)
 RMSE=np.sqrt(sum(errs*errs)/len(errs))
 
 return MAE,RMSE

if __name__ == '__main__':
 random.seed(0)
 df1=gen.generate(0,10,10)#00)
 df2=gen.generate(0,10,10)#00)
 
 print(df1)
 print(df2)
 
 mae, rmse = model_and_get_stats(df1,df2)
 
 print(mae, rmse)
