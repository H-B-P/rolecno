import pandas as pd
import numpy as np
import random
import math

def generate(baseA, baseB, N):
    dictForDf = {}

    dictForDf["x1"]=[]
    dictForDf["x2"]=[]
    dictForDf["x3"]=[]
    dictForDf["x4"]=[]
    dictForDf["y"]=[]

    df=pd.DataFrame(dictForDf)

    for i in range(N):
     newX1 = random.choice([-1,1])
     newX2 = random.choice([-1,1])
     newX3 = random.choice([-1,1])
     newX4 = random.choice([-1,1])
     newYbase = baseA* math.pow(2, newX1 + newX2 - newX3 - newX4) + baseB* math.pow(2, newX1 - newX2 + newX3 - newX4)
     #newYbase = baseA* math.pow(2, newX1 + newX2 + newX3 + newX4) + baseB* math.pow(2, - newX1 - newX2 - newX3 - newX4)
     newY = np.random.poisson(newYbase)
     dictToAppend = {"x1":newX1,"x2":newX2,"x3":newX3,"x4":newX4, "y":newY}
     df = df.append(dictToAppend, ignore_index=True)
    return df

def generate_trio(baseA, baseB, baseC, N):
    dictForDf = {}

    dictForDf["x1"]=[]
    dictForDf["x2"]=[]
    dictForDf["x3"]=[]
    dictForDf["x4"]=[]
    dictForDf["y"]=[]

    df=pd.DataFrame(dictForDf)

    for i in range(N):
     newX1 = random.choice([-1,1])
     newX2 = random.choice([-1,1])
     newX3 = random.choice([-1,1])
     newX4 = random.choice([-1,1])
     newYbase = baseA* math.pow(2, newX1 + newX2 - newX3 - newX4) + baseB* math.pow(2, newX1 - newX2 + newX3 - newX4) + baseC* math.pow(2, newX1 - newX2 - newX3 + newX4)
     newY = np.random.poisson(newYbase)
     dictToAppend = {"x1":newX1,"x2":newX2,"x3":newX3,"x4":newX4, "y":newY}
     df = df.append(dictToAppend, ignore_index=True)
    return df

def generate_trio_x2(baseA, baseB, baseC, N):
    dictForDf = {}

    dictForDf["x1"]=[]
    dictForDf["x2"]=[]
    dictForDf["x3"]=[]
    dictForDf["x4"]=[]
    dictForDf["x5"]=[]
    dictForDf["x6"]=[]
    dictForDf["x7"]=[]
    dictForDf["x8"]=[]
    dictForDf["y"]=[]

    df=pd.DataFrame(dictForDf)

    for i in range(N):
     newX1 = random.choice([-1,1])
     newX2 = random.choice([-1,1])
     newX3 = random.choice([-1,1])
     newX4 = random.choice([-1,1])
     newX5 = random.choice([-1,1])
     newX6 = random.choice([-1,1])
     newX7 = random.choice([-1,1])
     newX8 = random.choice([-1,1])
     newYbase = baseA* math.pow(2, (newX1+newX2) + (newX3+newX4) - (newX5+newX6) - (newX7+newX8)) + baseB* math.pow(2, (newX1+newX2) - (newX3+newX4) + (newX5+newX6) - (newX7+newX8)) + baseC* math.pow(2, (newX1+newX2) - (newX3+newX4) - (newX5+newX6) + (newX7+newX8))
     newY = np.random.poisson(newYbase)
     dictToAppend = {"x1":newX1,"x2":newX2,"x3":newX3,"x4":newX4,"x5":newX5,"x6":newX6,"x7":newX7,"x8":newX8, "y":newY}
     df = df.append(dictToAppend, ignore_index=True)
    return df

if __name__ == '__main__':
 theDf=generate(1, 1, 100)
 print(theDf)
