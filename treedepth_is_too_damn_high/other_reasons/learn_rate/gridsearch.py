import pandas as pd
import numpy as np
import random

import xgboost as xgb

import gen
import model

treeDepths=[1,2,3,4,5]
learnRates=[0.1, 0.2, 0.5, 1]
nRoundList=[10, 20, 50, 100, 200, 500, 1000]

#random.seed(0)
train=gen.generate(1,10,1000, 2)
test=gen.generate(1,10,1000, 2)

for treeDepth in treeDepths:
 print("TREEDEPTH: "+str(treeDepth))
 mins=[]
 for learnRate in learnRates:
  opList=[]
  for nRound in nRoundList:
   mae, rmse = model.model_and_get_stats(train, test, {'max_depth':treeDepth, 'learning_rate':learnRate, "n_rounds":nRound})
   opList.append(round(mae, 4))
  print(opList)
  mins.append(min(opList))
 print("MIN: " + str(min(mins)))
