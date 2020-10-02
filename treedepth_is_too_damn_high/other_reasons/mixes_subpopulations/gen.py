import pandas as pd
import numpy as np
import random
import math

def generate(base, N, solo=False):
    dictForDf = {}

    dictForDf["x1"]=[]
    dictForDf["x2"]=[]
    dictForDf["x3"]=[]
    dictForDf["x4"]=[]
    dictForDf["t"]=[]
    dictForDf["y"]=[]

    df=pd.DataFrame(dictForDf)

    for i in range(N):
     newX1 = random.choice([0,1])
     newX2 = random.choice([0,1])
     newX3 = random.choice([0,1])
     newX4 = random.choice([0,1])
     if solo:
      newT=1
     else:
      newT = random.choice([-1,1])
     newYbase = base* math.pow(2, newX1 + newX2*newT - newX3*newT - newX4)
     newY = np.random.poisson(newYbase)
     dictToAppend = {"x1":newX1,"x2":newX2,"x3":newX3,"x4":newX4, "t": newT, "y":newY}
     df = df.append(dictToAppend, ignore_index=True)
    return df

def generateII(base, N, solo=False):
    dictForDf = {}

    dictForDf["x1"]=[]
    dictForDf["x2"]=[]
    dictForDf["x3"]=[]
    dictForDf["x4"]=[]
    dictForDf["x5"]=[]
    dictForDf["x6"]=[]
    dictForDf["x7"]=[]
    dictForDf["x8"]=[]
    dictForDf["t"]=[]
    dictForDf["y"]=[]

    df=pd.DataFrame(dictForDf)

    for i in range(N):
     newX1 = random.choice([0,1])
     newX2 = random.choice([0,1])
     newX3 = random.choice([0,1])
     newX4 = random.choice([0,1])
     newX5 = random.choice([0,1])
     newX6 = random.choice([0,1])
     newX7 = random.choice([0,1])
     newX8 = random.choice([0,1])
     if solo:
      newT=1
     else:
      newT = random.choice([-1,1])
     newYbase = base* math.pow(2, newX1 + newX2 + newX3*newT + newX4*newT - newX5*newT - newX6*newT - newX7 - newX8)
     newY = np.random.poisson(newYbase)
     dictToAppend = {"x1":newX1,"x2":newX2,"x3":newX3,"x4":newX4,"x5":newX5,"x6":newX6,"x7":newX7,"x8":newX8, "t": newT, "y":newY}
     df = df.append(dictToAppend, ignore_index=True)
    return df

def generateIII(base, N, solo=False):
    dictForDf = {}

    dictForDf["x1"]=[]
    dictForDf["x2"]=[]
    dictForDf["x3"]=[]
    dictForDf["x4"]=[]
    dictForDf["x5"]=[]
    dictForDf["x6"]=[]
    dictForDf["x7"]=[]
    dictForDf["x8"]=[]
    dictForDf["x9"]=[]
    dictForDf["x10"]=[]
    dictForDf["x11"]=[]
    dictForDf["x12"]=[]
    dictForDf["t"]=[]
    dictForDf["y"]=[]

    df=pd.DataFrame(dictForDf)

    for i in range(N):
     newX1 = random.choice([0,1])
     newX2 = random.choice([0,1])
     newX3 = random.choice([0,1])
     newX4 = random.choice([0,1])
     newX5 = random.choice([0,1])
     newX6 = random.choice([0,1])
     newX7 = random.choice([0,1])
     newX8 = random.choice([0,1])
     newX9 = random.choice([0,1])
     newX10 = random.choice([0,1])
     newX11 = random.choice([0,1])
     newX12 = random.choice([0,1])
     if solo:
      newT=1
     else:
      newT = random.choice([-1,1])
     newYbase = base* math.pow(2, (newX1 + newX2 + newX3) + (newX4 + newX5 + newX6)*newT - (newX7+newX8+newX9)*newT - (newX10 + newX11+newX12))
     newY = np.random.poisson(newYbase)
     dictToAppend = {"x1":newX1,"x2":newX2,"x3":newX3,"x4":newX4,"x5":newX5,"x6":newX6,"x7":newX7,"x8":newX8,"x9":newX9,"x10":newX10,"x11":newX11,"x12":newX12, "t": newT, "y":newY}
     df = df.append(dictToAppend, ignore_index=True)
    return df

if __name__ == '__main__':
 theDf=generate(5, 10, False)
 print(theDf)
 theDf=generateII(5, 10, False)
 print(theDf)
 theDf=generateIII(5, 10, False)
 print(theDf)
