#!/usr/bin/env python
# coding: utf-8

# # Exploring Criminalistic Colombian Data with plotly
# 
# In this notebook we are going to explore some graphing functions of plotly. For this we will use data from homicides registered in Colombia by the national police [national police](https://www.policia.gov.co/grupo-informacion-criminalidad/estadistica-delictiva). 
# 
# 
# First we will need to clean the data so that it matches the info from the .json polygon data. After this, plotly will be used to graph the map with the different frequencies of homicides by region in Colombia.
# 
# 

# In[1]:


# Data Structure
import json
import pandas as pd
from unidecode import unidecode
from urllib.request import urlopen

# Graph Modules Plotlty 
import plotly.graph_objects as go


# ## Modifying the data
# 
# The only thing we have to do in this case is to replace some names of regions of Colombia so that they match the data labels of the geometry file .json, so we will only use pandas and unidecode to remove accents from some words

# In[2]:


df = pd.read_excel('homicidios2021.xlsx')
df.columns = [i.lower().strip() for i in df.columns]
df.head(5)
# import os
# os.listdir()


# In[3]:


departamentos = []
for i in range(len(df)):
  if df.loc[i]['departamento'] == 'CUNDINAMARCA' and df.loc[i]['municipio'] == 'BOGOTÁ D.C. (CT)':
    departamentos.append('SANTAFE DE BOGOTA D.C')
  else:
    departamentos.append(df.loc[i]['departamento'])

df['departamento'] = departamentos
df['departamento'].replace({'GUAJIRA':'LA GUAJIRA','VALLE':'VALLE DEL CAUCA',
                            'SAN ANDRES':'ARCHIPIELAGO DE SAN ANDRES PROVIDENCIA Y SANTA CATALINA'},inplace=True)
df['departamento'].replace({old:unidecode(old) for old in df['departamento'].unique()},inplace=True)
df['departamento'].replace({'NARINO':'NARIÑO'},inplace=True)

contador = df['departamento'].value_counts()
contador


# ## Time To Use Plotly
# 
# The homicide frequency data by region is ready, we simply implement plotly using polygon data to draw the map of Colombia (data extracted from the Github profile of [john-guerra](https://gist.github.com/john-guerra/))

# In[4]:


with urlopen('https://gist.githubusercontent.com/john-guerra/43c7656821069d00dcbc/raw/be6a6e239cd5b5b803c6e7c2ec405b793a9064dd/Colombia.geo.json') as response:
    counties = json.load(response)

locs = contador.index

for loc in counties['features']:
    loc['id'] = loc['properties']['NOMBRE_DPT']

fig = go.Figure(go.Choroplethmapbox(
                     geojson=counties,
                      locations=locs,
                      z=contador.values,
                      colorscale='deep',
                      colorbar_title="Number of homicides"),)
fig.update_layout(mapbox_style="carto-positron",
                        mapbox_zoom=5,
                        mapbox_center = {"lat": 4.570868, "lon": -74.2973328},
                        width=800, height=1000,
                        margin=dict(l=0,r=0,b=100,t=40,pad=4
    ))
fig.show()


# ## An Interesting Ad
# 
# This same type of statistics about homicides in the Colombian case is present for other types of crimes. I invite you to visit my Colombian crime statistics portal that I built with Streamlit and python

# In[ ]:





# In[ ]:




