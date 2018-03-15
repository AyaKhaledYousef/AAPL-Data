# -*- coding: utf-8 -*-
"""
Created on Mon Mar 12 14:41:11 2018

"""

import numpy as np 
import pandas as pd 

C1 = pd.Series(np.random.randn(1000000)) # generat random float numbers
C2 = pd.Series(np.random.randn(1000000))
C3 = pd.Series(np.random.randn(1000000))

df = pd.DataFrame({'C1':C1, 'C2':C2 ,'C3':C3 }) #creat table  with 3 coiumns

df['C4'] = np.random.choice(['A','B','C','D'], len(df)) #creat column random char  standered to len df 

df=df.pivot_table(values=('C1','C2','C3'),index='C4',aggfunc=np.sum) #calculate sum of each group


df['Sum'] = df.sum(axis=1) #sum rows

print(df) 
