import pandas as pd
import numpy as np
import random

def generate(k=1,  D=10,  d=3,  N=1000):
    dictForDf = {}

    for i in range(D):
     dictForDf["x"+str(i)]=[]

    df=pd.DataFrame(dictForDf)

    for i in range(N):
     dictToAppend={}
     for j in range(D):
      dictToAppend["x"+str(j)]=random.choice([0,1])
     df = df.append(dictToAppend, ignore_index=True)

    columnsThatMakeSpecial=["x"+str(i) for i in range(d)]

    print(columnsThatMakeSpecial)

    df["Special"]=1

    for c in columnsThatMakeSpecial:
     df["Special"]=df["Special"]*df[c]

    df["y"]=np.random.normal(k*df["Special"], 1)

    return df
