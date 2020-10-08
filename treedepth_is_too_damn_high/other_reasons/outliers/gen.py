import pandas as pd
import numpy as np
import random
import math

def generate(N, explanatories, error, outlierRarity, outlierMagnitude, evenlySpreadOutliers=True, continuous=False, continuousStdev=1):
    dictForDf = {}
    
    dictForDf["y"]=[]
    for j in range(explanatories):
     dictForDf["x"+str(j)]=[]

    df=pd.DataFrame(dictForDf)

    for i in range(N):
     newY=0
     dictToAppend = {}
     for j in range(explanatories):
      if continuous:
       newX=np.random.normal(0, continuousStdev)
      else:
       newX=random.choice([-1,1])
      dictToAppend["x"+str(j)]=newX
      newY+=newX
     newY = np.random.normal(newY, error)
     if ((evenlySpreadOutliers and i%outlierRarity==0) or ((not evenlySpreadOutliers) and random.random()<1/outlierRarity)):
      newY=outlierMagnitude
     dictToAppend["y"]=newY
     df = df.append(dictToAppend, ignore_index=True)
    return df



if __name__ == '__main__':
 theDf=generate(100, 5, 1, 10, 50)
 print(theDf)
