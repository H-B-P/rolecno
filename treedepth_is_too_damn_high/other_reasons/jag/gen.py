import pandas as pd
import numpy as np
import random


import plotly.graph_objects as go

def generate(sigma=1, density=1, iterations=2):
    dictForDf = {}

    dictForDf["x"]=[]
    dictForDf["y"]=[]

    df=pd.DataFrame(dictForDf)

    for i in range(iterations*20*density+1):
     newX = float(i)/float(density)
     newY = np.random.normal(convert_to_W(newX), sigma)
     dictToAppend = {"x":newX, "y":newY}
     df = df.append(dictToAppend, ignore_index=True)
    return df

def generateWithDecoys(sigma=0, density=1, iterations=2, decoys=5, decoysigma=1):
    dictForDf = {}

    dictForDf["x"]=[]
    dictForDf["y"]=[]

    for j in range(decoys):
     dictForDf["d"+str(j)]=[]

    df=pd.DataFrame(dictForDf)

    for i in range(iterations*20*density+1):
     newX = float(i)/float(density)
     newY = np.random.normal(convert_to_W(newX), sigma)
     dictToAppend = {"x":newX, "y":newY}
     for j in range(decoys):
      dictToAppend["d"+str(j)]=np.random.normal(0, decoysigma)
     df = df.append(dictToAppend, ignore_index=True)
    return df


def convert_to_W(x):
 remainder = x%20
 basis = x-x%20
 if remainder<10:
  return 10-remainder
 else:
  return remainder-10

if __name__ == '__main__':
 theDf=generateWithDecoys(0, 2,3, 5, 1)
 print(theDf)
 #fig = go.Figure()
 #fig.add_trace(go.Scatter(x=theDf["x"], y=theDf["y"],
 #                   mode='markers',
 #                   name='data'))
 #fig.show()
