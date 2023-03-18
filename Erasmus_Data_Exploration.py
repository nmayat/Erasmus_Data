# -*- coding: utf-8 -*-
"""
Created on Sat Mar 11 17:24:58 2023

@author: nils
"""

import plotly.io as pio
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import pycountry
import plotly.subplots as sp
from geopy.geocoders import Nominatim
import chart_studio
import chart_studio.plotly as py
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

username='Mahuvej'
api_key='PVqDbavnEGBPLqw1oiN0'

chart_studio.tools.set_credentials_file(username=username,
                                        api_key=api_key)

mapbox_access_token = 'pk.eyJ1IjoibWFodXZlaiIsImEiOiJjbGZlMnVhZWkxNHRrM3huMXFldmxsMDNoIn0.vH_J9CTHTVoKyaOykLXNnQ'
#pio.renderers.default = 'browser'
layout = dict(
    height = 800,
    # top, bottom, left and right margins
    margin = dict(t = 0, b = 0, l = 0, r = 0),
    font = dict(color = '#FFFFFF', size = 11),
    paper_bgcolor = '#000000',
    mapbox = dict(
        # here you need the token from Mapbox
        accesstoken = mapbox_access_token,
        bearing = 0,
        # we want the map to be "parallel" to our screen, with no angle
        pitch = 0,
        # default level of zoom
        zoom = 3,

    )
)
def plot_world_dist(data,column):
    counts = data.groupby(column).size().reset_index(name='Count')
    all_countries = pd.DataFrame(
        {'Country': [c.name for c in pycountry.countries]})
    merged = all_countries.merge(
        counts, left_on='Country', right_on=column, how='left')
    merged['Count'].fillna(0, inplace=True)
    fig = px.choropleth(merged, locations='Country', locationmode='country names', color='Count',
                        color_continuous_scale='Cividis',  hover_data=['Count'],
                        center={'lat': 50, 'lon': 10})
    fig.update_layout(title=column)
    
    py.plot(fig, filename="Erasmus_"+column, auto_open = True)

def plot_hist(data, column):
    data = data.groupby(column).size().reset_index(name='Count')
    fig = px.bar(data_frame=data, x=column,y = 'Count', title='Histogram of '+column)
    fig = fig.update_layout(xaxis_categoryorder='total descending')
    fig.update_xaxes(tickangle=45)

    fig.show()
    py.plot(fig, filename="Erasmus_"+column, auto_open = True)
    


data = pd.read_pickle('Erasmus_Data/data_clean.pkl')

plot_world_dist(data, 'Participant Nationality')
plot_world_dist(data, 'Sending Country')
plot_world_dist(data, 'Receiving Country')

plot_hist(data, 'Activity')
plot_hist(data, 'Education Level')
plot_hist(data, 'Mobility Duration')

fig = px.bar(data_frame=data.groupby('Activity')['Mobility Duration'].mean().reset_index(),
             x='Activity', y='Mobility Duration',
             title='Mean Mobility Duration by Activity',
             labels={'Activity': 'Activity', 'Mobility Duration': 'Mean Mobility Duration'})

fig = fig.update_layout(xaxis_categoryorder='total descending')
fig.show()
py.plot(fig, filename="Erasmus_"+'Mobility_Duration_by_Activity', auto_open = True)

plot_hist(data, 'Participant Age')

fig = px.bar(data_frame=data.groupby('Activity')['Participant Age'].mean().reset_index(),
             x='Activity', y='Participant Age',
             title='Mean Participant Age by Activity',
             labels={'Activity': 'Activity', 'Participant Age': 'Mean Participant Age'})

fig = fig.update_layout(xaxis_categoryorder='total descending')
fig.show()
py.plot(fig, filename="Erasmus_"+'Participant_Age_by_Activity', auto_open = True)

plot_hist(data, 'Participant Gender')

plot_hist(data, 'Field of Education')

plot_hist(data,'Mobility Start Month')
counts = data['Mobility Start Month'].value_counts()



print(data['Special Needs'].sum())
print(data['Special Needs'].sum() / data.shape[0])

print(data['Fewer Opportunities'].sum())
print(data['Fewer Opportunities'].sum() / data.shape[0])



# Filter dataframe to have only the columns you need and focus on the sending country
sending_country = 'Germany'  # Replace with the country you want to focus on
filtered_df = data[data['Sending Country'] == sending_country][['Sending Country', 'Receiving Country']]

geolocator = Nominatim(user_agent="erasmus_app", timeout=10) 

def get_latitude_longitude(country):
    manual_mapping = {
        'Georgia': (42.3154, 43.3569),
        'Palestine': (31.9466, 35.2731)
    }

    if country in manual_mapping:
        return manual_mapping[country]

    location = geolocator.geocode(country)

    if location is not None:
        return location.latitude, location.longitude
    else:
        print(f"Failed to find coordinates for {country}")
        return None, None


sending_lat, sending_lon = get_latitude_longitude(sending_country)

# Group by receiving country and count the number of students
country_counts = filtered_df.groupby('Receiving Country').size().reset_index(name='Number of Students')
country_counts['Receiving Latitude'], country_counts['Receiving Longitude'] = zip(*country_counts['Receiving Country'].apply(get_latitude_longitude))
country_counts.dropna(subset=['Receiving Latitude', 'Receiving Longitude'], inplace=True)

# Create a subplot with a world map in the background
fig = sp.make_subplots(rows=1, cols=1, specs=[[{"type": "scattergeo"}]])

# Add a choropleth map for the receiving countries with custom hovertemplate
fig.add_trace(go.Choropleth(
    locations=country_counts['Receiving Country'],
    z=country_counts['Number of Students'],
    text=country_counts['Receiving Country'],
    hovertemplate='%{text}<br>Number of Students: %{z}',
    colorscale='Viridis',
    autocolorscale=False,
    reversescale=False,
    marker_line_color='darkgray',
    marker_line_width=0.5,
    colorbar_tickprefix='',
    colorbar_title='Number of Students',
    locationmode="country names",
))

# Add arrow annotations with a fixed width
arrow_width = 1.5
base_opacity = 0.8

for index, row in country_counts.iterrows():
    fig.add_trace(
        go.Scattergeo(
            lon=[sending_lon, row['Receiving Longitude']],
            lat=[sending_lat, row['Receiving Latitude']],
            mode='lines',
            line=dict(width=arrow_width, color='red'),
            opacity=base_opacity,
            showlegend=False,
            hoverinfo='none',  # Suppress the display of coordinates when hovering over the arrows
        )
    )

# Update the layout for the world map
fig.update_geos(
    showland=True,
    landcolor='rgb(243, 243, 243)',
    countrycolor='rgb(204, 204, 204)',
    showcountries=True,
    projection_type="equirectangular",
)

fig.update_layout(title_text=f'Movement of Erasmus Students from {sending_country}')

fig.show()
py.plot(fig, filename="Erasmus_"+ sending_country+ 'Movement', auto_open = True)
