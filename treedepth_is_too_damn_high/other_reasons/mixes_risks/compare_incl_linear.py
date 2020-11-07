import pandas as pd
import numpy as np
import random

import xgboost as xgb

import plotly.graph_objects as go

import gen

import model
import canonical_mixes_risks

import linear_model

df1=gen.generate_trio(1, 1,1, 3000)
df2=gen.generate_trio(1, 1,1, 3000)
print("Mislinked TD1")
print(model.model(df1,df2,{'max_depth':1,'learning_rate':0.3}, rounds=100))
print("TD1")
print(model.model(df1,df2,{'max_depth':1,'learning_rate':0.3, 'objective':'count:poisson'}, rounds=100))
print("TD1 (but 2x rounds)")
print(model.model(df1,df2,{'max_depth':1,'learning_rate':0.3, 'objective':'count:poisson'}, rounds=200))
print("TD2")
print(model.model(df1,df2,{'max_depth':2,'learning_rate':0.3, 'objective':'count:poisson'}, rounds=100))
print("TD3")
print(model.model(df1,df2,{'max_depth':3,'learning_rate':0.3, 'objective':'count:poisson'}, rounds=100))
print("LD1")
print(linear_model.linear_model_with_interxns(df1,df2,1))
print("LD2")
print(linear_model.linear_model_with_interxns(df1,df2,2))
print("LD3")
print(linear_model.linear_model_with_interxns(df1,df2,3))
print("canonical 1")
print(canonical_mixes_risks.canonically_model(df1,df2, 2000, 1, 0.1))
print("canonical 2")
print(canonical_mixes_risks.canonically_model(df1,df2, 2000, 2, 0.1))
print("canonical 3")
print(canonical_mixes_risks.canonically_model(df1,df2, 2000, 3, 0.1))
