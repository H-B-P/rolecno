import pandas as pd
import numpy as np
import random

import xgboost as xgb

import plotly.graph_objects as go

import gen

import model
import linear_model
import canonical_mixes_risks

df1=gen.generate_trio_x2(1, 1,1, 10000)
df2=gen.generate_trio_x2(1, 1,1, 10000)
print("Mislinked TD1")
print(model.model(df1,df2,{'max_depth':1,'learning_rate':0.3}, rounds=100))
print("TD1")
print(model.model(df1,df2,{'max_depth':1,'learning_rate':0.3, 'objective':'count:poisson'}, rounds=500))
print("TD1 (but 2x rounds)")
print(model.model(df1,df2,{'max_depth':1,'learning_rate':0.3, 'objective':'count:poisson'}, rounds=1000))
print("TD2")
print(model.model(df1,df2,{'max_depth':2,'learning_rate':0.3, 'objective':'count:poisson'}, rounds=1000))
print("TD3")
print(model.model(df1,df2,{'max_depth':3,'learning_rate':0.3, 'objective':'count:poisson'}, rounds=1000))
print("TD4")
print(model.model(df1,df2,{'max_depth':4,'learning_rate':0.3, 'objective':'count:poisson'}, rounds=1000))
print("TD5")
print(model.model(df1,df2,{'max_depth':5,'learning_rate':0.3, 'objective':'count:poisson'}, rounds=1000))
print("TD6")
print(model.model(df1,df2,{'max_depth':6,'learning_rate':0.3, 'objective':'count:poisson'}, rounds=1000))
print("TD7")
print(model.model(df1,df2,{'max_depth':7,'learning_rate':0.3, 'objective':'count:poisson'}, rounds=1000))
print("TD8")
print(model.model(df1,df2,{'max_depth':8,'learning_rate':0.3, 'objective':'count:poisson'}, rounds=1000))
print ("LD1")
print(linear_model.linear_model_with_interxns(df1,df2,1))
print ("LD2")
print(linear_model.linear_model_with_interxns(df1,df2,2))
print ("LD3")
print(linear_model.linear_model_with_interxns(df1,df2,3))
print ("LD4")
print(linear_model.linear_model_with_interxns(df1,df2,4))
print ("LD5")
print(linear_model.linear_model_with_interxns(df1,df2,5))
print ("LD6")
print(linear_model.linear_model_with_interxns(df1,df2,6))
print ("LD7")
print(linear_model.linear_model_with_interxns(df1,df2,7))
print ("LD8")
print(linear_model.linear_model_with_interxns(df1,df2,8))
print("canonical 1")
print(canonical_mixes_risks.canonically_model(df1,df2, 2000, 1, 0.02))
print("canonical 1")
print(canonical_mixes_risks.canonically_model(df1,df2, 2000, 1, 0.05))
print("canonical 1")
print(canonical_mixes_risks.canonically_model(df1,df2, 10000, 1, 0.02))
print("canonical 2")
print(canonical_mixes_risks.canonically_model(df1,df2, 2000, 2, 0.02))
print("canonical 3")
print(canonical_mixes_risks.canonically_model(df1,df2, 2000, 3, 0.02))
