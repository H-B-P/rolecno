import pandas as pd
import numpy as np
import random

def generate(linkage="log", explan=3, probshare=0.9):
 bigList=[{"xOrigin":1, "p":0.5}, {"xOrigin":0, "p":0.5}]
 
 for i in range(explan):
  newList=[]
  for d in bigList:
   dtrue = d.copy()
   dfalse = d.copy()
   dtrue["x"+str(i)]=1
   dfalse["x"+str(i)]=0
   if d["xOrigin"]==1:
    dtrue["p"] = d["p"]*probshare
    dfalse["p"] = d["p"]*(1-probshare)
   else:
    dtrue["p"] = d["p"]*(1-probshare)
    dfalse["p"] = d["p"]*probshare
   newList.append(dtrue)
   newList.append(dfalse)
  bigList = newList
 
 df = pd.DataFrame(bigList)
 
 df["y"]=0
 
 for i in range(explan):
  df["y"]=df["y"]+df["x"+str(i)]
 
 if linkage=="log":
  df["y"]=np.power(2, df["y"])
 
 df["av"]=df["y"]*df["p"]
 
 return df

if __name__ == '__main__':
 theDf=generate()
 print(theDf)
 print(sum(theDf["p"]))
