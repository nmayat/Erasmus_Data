# -*- coding: utf-8 -*-
"""
Created on Sat Mar 11 15:58:40 2023

@author: nils
"""

import pandas as pd
import os 


data_files = [f for f in os.listdir('Erasmus_Data') if (f.endswith('.csv') and 'mobilities eligible finalised started in')]
data = pd.DataFrame()

for file in data_files:
    year_data = pd.read_csv('Erasmus_Data/'+ file, delimiter = ';')
    data = pd.concat([data, year_data], axis=0, ignore_index=True)

data.to_pickle('Erasmus_Data/data_complete.pkl')

