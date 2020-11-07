import pandas as pd
import numpy as np
import random
import math

defaultOps=[[[[11,13],[17,19]],
             [[23,29],[31,37]]],
            [[[41,43],[47,53]],
             [[59,61],[67,71]]]]

defaultWeights=[[[[1,1],[1,1]],
                 [[1,1],[1,1]]],
                [[[1,1],[1,1]],
                 [[1,1],[1,1]]]]

def generate(ops=defaultOps, weights=defaultWeights):
    dictForDf = {}

    dictForDf["x1"]=[]
    dictForDf["x2"]=[]
    dictForDf["x3"]=[]
    dictForDf["x4"]=[]
    dictForDf["y"]=[]
    dictForDf["w"]=[]

    df=pd.DataFrame(dictForDf)

    for i in range(16):
     newX1 = (i/8)%2
     newX2 = (i/4)%2
     newX3 = (i/2)%2
     newX4 = (i/1)%2
     newY = ops[newX1][newX2][newX3][newX4]
     newW = weights[newX1][newX2][newX3][newX4]
     dictToAppend = {"x1":newX1,"x2":newX2,"x3":newX3,"x4":newX4, "y":newY, "w":newW}
     df = df.append(dictToAppend, ignore_index=True)
    return df


if __name__ == '__main__':
 theDf=generate(defaultOps,defaultWeights)
 print(theDf)
