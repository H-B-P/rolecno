import pandas as pd
import numpy as np
import random

import xgboost as xgb

import gen
import sprinkle_model

sprinkleCounts=[0,63]
learnRates=list(np.arange(0.01, 0.5, 0.01))
nRoundList=[1000]

#random.seed(0)
train=gen.generate(2,2,5)
test=gen.generate(0,2,5)

scmins={}

for sprinkleCount in sprinkleCounts:
 print("SC: "+str(sprinkleCount))
 mins=[]
 for learnRate in learnRates:
  opList=[]
  for nRound in nRoundList:
   mae, rmse = sprinkle_model.model(train, test, {'max_depth':1, 'learning_rate':learnRate}, nRound, nRound, 0.01, sprinkleCount, False)
   opList.append(round(rmse, 4))
  print(opList)
  mins.append(min(opList))
 print("MIN: " + str(min(mins)))
 scmins[sprinkleCount]=min(mins)

print(scmins)
