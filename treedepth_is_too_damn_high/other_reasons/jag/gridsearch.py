import pandas as pd
import numpy as np
import random

import xgboost as xgb

import gen
import model

treeDepths=[1,2]
#learnRates=[0.3]#list(np.arange(0.01, 1, 0.01))+list(np.arange(0.001, 0.1, 0.001))
learnRates={1:list(np.arange(0.01, 1, 0.01)), 2:list(np.arange(0.001, 0.1, 0.001))}
#learnRates={1:[0.01], 2:[0.01]}
nRoundList=[500]

#random.seed(0)
train=gen.generate(2,20,2)
test=gen.generate(0,20,2)

tdmins={}

for treeDepth in treeDepths:
 print("TREEDEPTH: "+str(treeDepth))
 mins=[]
 for learnRate in learnRates[treeDepth]:#learnRates:
  opList=[]
  for nRound in nRoundList:
   mae, rmse = model.model(train, test, {'max_depth':treeDepth, 'learning_rate':learnRate}, nRound, nRound)
   opList.append(round(rmse, 4))
  print(opList)
  mins.append(min(opList))
 print("MIN: " + str(min(mins)))
 tdmins[treeDepth]=min(mins)

print(tdmins)
