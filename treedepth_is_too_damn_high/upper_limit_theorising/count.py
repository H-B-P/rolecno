import pandas as pd
import numpy as np
import itertools

import gen

k=0.2
D=10
d=2
N=1600

df = gen.generate(k,D,d,N)

print(df)

xes = ["x"+str(i) for i in range(D)]

dcombos = list(itertools.combinations(xes, d))

binaries = [[]]

for i in range(d):
 newBinaries=[]
 for b in binaries:
  newBinaries.append(b+[0])
  newBinaries.append(b+[1])
 binaries=newBinaries



#startingGuess=sum(df["y"])/len(df["y"])

pullFromTarget = sum(df[df["Special"]==1]["y"])

print(pullFromTarget)

print(len(dcombos), len(binaries), len(dcombos)*len(binaries))

winners=[]

for dcombo in dcombos:
  for binary in binaries:
    matchCol = df[list(dcombo)]==binary
    #print(matchCol)
    matchedDf = df[matchCol.prod(axis=1)==True]
    #print(matchedDf)
    pull = sum(matchedDf["y"])
    #print(matchedDf)
    #print(pull)
    if pull>pullFromTarget:
     winners.append((dcombo, binary, pull))

print(winners)

print(len(winners))


#relevantBinary = binaries[0]

#print(relevantBinary)

#print(relevantCols)

#truthyDf = df[relevantXCols]==relevantBinary

#print(truthyDf.prod(axis=1))


