# -*- coding: utf-8 -*-
"""
Created on Tue Mar 14 22:15:08 2023

@author: nils
"""
import pandas as pd 
import pycountry
import numpy as np

def clean_age(x):
    try:
        x = int(x)
    except:
        x = np.nan

    if x < 5 or x > 90:
        x = np.nan
    return x


    
data = pd.read_pickle('Erasmus_Data/data_complete.pkl')
print(f"The shape of the data to clean is: {data.shape}")
print(f"The columns in the original dataset are: {data.columns}")
data = data.drop(['Project Reference', 'Academic Year', 'Participant Profile',
                  'GroupLeader', 'Sending Organisation Erasmus Code',
                  'Receiving Organisation Erasmus Code'], axis=1)

data.isna().sum()
data.dropna(inplace=True)
print(f"There are {data[data['Participant Nationality'] == '-'].shape[0]} without Nationality")
data = data.drop(data[data['Participant Nationality'] == '-'].index, axis=0)
data = data.rename(columns={'Sending Country Code': 'Sending Country', 'Receiving Country Code': 'Receiving Country',
                            'Activity (mob)': 'Activity'})
country_dict = {c.alpha_2: c.name for c in pycountry.countries}
country_dict.update({'XK': 'Kosovo', 'EL': 'Greece', 'UK': 'United Kingdom', "TP": "East Timor", "AN": "Netherlands Antilles",
                     "AB": "Albania", "CP": "Clipperton Island"})
data[['Participant Nationality',
      'Sending Country',
      'Receiving Country']] = data[['Participant Nationality',
                                    'Sending Country',
                                    'Receiving Country']].applymap(lambda x: country_dict.get(x, x))
data = data.loc[data.index.repeat(data['Participants'])].reset_index(drop=True)

data['Activity'] = data['Activity'].replace({"Advance Planning Visit – EVS": "Advance Planning Visit - EVS",
                                             "Training/teaching assignments abroad": "Teaching/training assignments abroad"})
data['Field of Education'] = data['Field of Education'].replace(
    "? Unknown ?", "Unknown")

data['Education Level'] = data['Education Level']\
    .replace({"??? - ? Unknown ?": "Unknown",
              "ISCED-2 - Lower secondary education": "Lower Secondary Education",
              "ISCED-3 - Upper secondary education": "Upper Secondary Education",
              "ISCED-4 - Post-secondary non-tertiary education": "Post-secondary non-Tertiary Education",
              "ISCED-5 - Short-cycle within the first cycle / Short-cycle tertiary education (EQF-5)": "Short-cycle within the first cycle / Short-cycle tertiary education",
              "ISCED-6 - First cycle / Bachelor’s or equivalent level (EQF-6)": "First cycle / Bachelor’s or equivalent level",
              "ISCED-7 - Second cycle / Master’s or equivalent level (EQF-7)": "Second cycle / Master’s or equivalent level",
              "ISCED-8 - Third cycle / Doctoral or equivalent level (EQF-8)": "Third cycle / Doctoral or equivalent level",
              "ISCED-9 - Not elsewhere classified": "Not elsewhere classified",
              })

print(f"These are the unique values in the age column before cleaning: {data['Participant Age'].unique()}")
data['Participant Age'] = data['Participant Age'].apply(lambda x: clean_age(x))

data['Mobility Start Month'] = pd.to_datetime(data['Mobility Start Month'])
data['Mobility End Month'] = pd.to_datetime(data['Mobility End Month'])

data['Special Needs'] = data['Special Needs'].replace({'No':0, 'Yes':1})
data['Fewer Opportunities'] = data['Fewer Opportunities'].replace({'No':0, 'Yes':1})

data.to_pickle('Erasmus_Data/data_clean.pkl')

data = data.drop(['Mobility Start Month', 'Mobility End Month', 'Participant Gender',
                  'Special Needs', 'Fewer Opportunities', 'Participant Nationality',
                  'Sending City', 'Sending Organization', 'Receiving City', 'Receiving Organization','Participants'], axis=1)

mean_age = data['Participant Age'].mean()
data['Participant Age'].fillna(mean_age, inplace=True)
data.to_pickle('Erasmus_Data/model_data.pkl')