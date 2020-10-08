import pandas as pd
import numpy as np
import random
import gen_analytic

def check_missings(linkage,present, absent, prob):
 df = gen_analytic.generate(linkage, present+absent, prob)
 presentFriends = ["x"+str(i) for i in range(present)]
 df=df[presentFriends+["p", "av"]]
 df = df.groupby(presentFriends).sum()
 df["defactoY"]=df["av"]/df["p"]
 if linkage=="log":
  df["defactoY"]=np.log(df["defactoY"])
 return df[["defactoY"]]

if __name__ == '__main__':
 df = check_missings("log",2,6, 0.9)
 print(df)
