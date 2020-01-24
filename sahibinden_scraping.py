#!/usr/bin/env python
# coding: utf-8

# In[1]:


#importing dependencies
import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup as bs


# In[2]:


headers = {'User-Agent':'Mozilla/5.0'}


# In[8]:


#empty lists to organize data based on class
model_info = []
ad_title = []
year_km_color = [] #year, km, and color are listed under same class
price = []
ad_date = []
location = []


# In[4]:


page_offset = list(np.arange(0,1000,50)) #parse through page 1 to 20


# In[5]:


min_km = [0, 50000, 85000, 119000, 153000, 190000, 230000] #filter cars based on km
max_km = [50000, 85000, 119000, 153000, 190000, 230000, 500000] #filter cars based on km


# In[9]:


for i, j in zip(min_km, max_km):
    for page in page_offset:
        r = requests.get(f'https://www.sahibinden.com/volkswagen-polo?pagingOffset={page}&pagingSize=50&a4_max={j}&sorting=date_asc&a4_min={i}', headers=headers)
        soup = bs(r.content,'lxml')
        model_info += soup.find_all("td",{"class":"searchResultsTagAttributeValue"})
        ad_title += soup.find_all("td",{"class":"searchResultsTitleValue"})
        year_km_color += soup.find_all("td",{"class":"searchResultsAttributeValue"})
        price += soup.find_all("td",{"class":"searchResultsPriceValue"})
        ad_date += soup.find_all("td",{"class":"searchResultsDateValue"})
        location += soup.find_all("td",{"class":"searchResultsLocationValue"})


# In[10]:


len(model_info) 


# In[11]:


#just to get the text which is the information needed
model_info_text = []
for i in range(0,6731):
    model_info_text.append(model_info[i].text)


# In[12]:


len(ad_title)


# In[13]:


ad_title_text = []
for i in range(0,6731):
    ad_title_text.append(ad_title[i].text)


# In[14]:


len(price)


# In[15]:


price_text = []
for i in range(0,6731):
    price_text.append(price[i].text)


# In[16]:


len(ad_date)


# In[17]:


ad_date_text = []
for i in range(0,6731):
    ad_date_text.append(ad_date[i].text)


# In[18]:


len(location)


# In[19]:


location_text = []
for i in range(0,6731):
    location_text.append(location[i].text)


# In[20]:


len(year_km_color)


# In[21]:


year_km_color_text = []
for i in range(0,20193):
    year_km_color_text.append(year_km_color[i].text)


# In[22]:


year_km_color_text[:5]


# In[23]:


year_text = year_km_color_text[::3] #every third element is year


# In[24]:


year_km_color_text.pop(0) #remove the first element. Now every third element is km


# In[25]:


km_text = year_km_color_text[::3]


# In[26]:


year_km_color_text.pop(0)


# In[27]:


color_text = year_km_color_text[::3]


# In[28]:


print(len(year_text))
print(len(km_text))
print(len(color_text))


# In[29]:


df = pd.DataFrame({"model":model_info_text, "ad_title":ad_title_text,
                   "year":year_text, "km":km_text, "color":color_text,
                   "price":price_text, "ad_date":ad_date_text, "location":location_text})


# In[30]:


print(df.shape)
print(df['ad_title'].nunique())


# In[31]:


#save dataframe to csv file
df.to_csv(r'C:\Users\soner\Desktop\Data_Science\Projects\predicting_used_cars_prices\polo_data.csv', index=None, header=True)


# In[ ]:




