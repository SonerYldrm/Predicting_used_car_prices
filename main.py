#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[557]:


df = pd.read_csv('polo_data.csv')


# # Data Cleaning

# In[558]:


print(df.shape)
df.head()


# New line indicators (\n) needs to removed. I will use pandas remove() function with regex parameter set True. Similarly TL representing Turkish currency in price cell should be removed to be able make numerical analysis.

# In[559]:


df = df.replace('\n','',regex=True) #remove \n in cells


# In[560]:


df.price = df.price.replace('TL','',regex=True) #remove TL in cells of price column


# In[561]:


df.head()


# In[562]:


df.dtypes


# In[563]:


df['ad_title'].nunique()


# ### ad_date

# The data type of date is object. To be able to use the dates properly, I need to convert data dype to datetime. Before using astpye() function, I need to change the name of months to English.

# In[564]:


months = {"Ocak":"January", "Şubat":"February", "Mart":"March", "Nisan":"April",
          "Mayıs":"May", "Haziran":"June","Temmuz":"July","Ağustos":"August",
          "Eylül":"September", "Ekim":"October", "Kasım":"November", "Aralık":"December"}


# In[565]:


df.ad_date = df.ad_date.replace(months, regex=True)


# In[566]:


df.head()


# In[567]:


df.ad_date = pd.to_datetime(df.ad_date)


# In[568]:


df.head()


# ### km and price

# km column is truncated while reading the csv file. It is because of 'dot' used in thousands. For example, 25.000 which is twenty five thousands detected as 25.0. To fix this issue, I will simply multiply 'km' column with 1000.

# In[569]:


df.km = df.km * 1000


# In[570]:


df.head()


# Due to 'dot' used in prices, I could not change the data type to int or float so I will remove the 'dot' first.

# In[571]:


df.iloc[:,5] = df.iloc[:,5].str.replace(r'.','')


# In[572]:


df.iloc[:,5] = df.iloc[:,5].str.replace(r',','') # in some cells ',' used


# In[573]:


df.head()


# In[574]:


df.price = df.price.astype('float64')


# In[575]:


df.dtypes


# ### Missing values

# In[576]:


df.isna().any()


# There is no missing value in the dataframe.

# ### Location

# In Turkey, location might be a factor in determining the price of a used car. Location data in our dataframe inclused city and district. I don't think a cars price change in different districts of the same city. Therefore, I will modify location data to include only the name of the city.

# In[577]:


df.location.nunique()


# In[578]:


df.ad_title.nunique()


# In[451]:


import re


# In[579]:


s = df['location']


# In[580]:


len(s)


# In[581]:


city_district = []
for i in range(0,6731):
    city_district.append(re.sub( r"([A-Z, 'Ç', 'İ', 'Ö', 'Ş', 'Ü'])", r" \1", s[i]).split()) #add letters specific to Turkish alphabet


# In[582]:


city_district[:5]


# In[583]:


len(city_district)


# In[584]:


city = []
for i in range(0,6731):
    city.append(city_district[i][0])


# In[585]:


city[:5]


# In[586]:


len(city)


# In[587]:


df['city'] = city


# In[588]:


df.head()


# In[589]:


df.city.nunique()


# Successful! There are 81 cities in Turkey.

# In[464]:


df.to_csv(r'C:\Users\soner\Desktop\Data_Science\Projects\predicting_used_cars_prices\polo_data_cleaned.csv', index=None, header=True)


# # Exploratory Data Analysis

# ### Price

# In[590]:


df.price.mean()


# In[591]:


df.price.median()


# There is a big difference between mean and median which indicates outliers. So first need to check if there is ubnormal values.

# In[592]:


print(df.price.max())
print(df.price.min())


# In[593]:


df.price.sort_values(ascending=False)


# Highest value and the lowest four values seem ubnormal so I will remove these.

# In[594]:


df['price'][1904]


# In[595]:


df.drop(1904, inplace=True)


# In[596]:


df.price.max()


# In[597]:


df.drop([6497,5603,4186,5203], inplace=True)


# In[598]:


df.price.sort_values(ascending=False)


# In[599]:


print(df.price.mean())
print(df.price.median())


# In[600]:


df.price.mode()


# In[601]:


sns.set(style='darkgrid')


# In[485]:


plt.figure(figsize=(8,5))
sns.boxplot(y='price', data=df, width=0.5)


# In[482]:


x = df.price
plt.figure(figsize=(10,6))
sns.distplot(x).set_title('Frequency Distribution Plot of Prices')


# The graphs above show the frequency distribution of prices. It is slightly right skewed which means there are some outliers with very high prices.

# ## Date

# I don't think date by itself has an effect on the price but waiting periof of the ad on website might have some effect. Longer waiting time might motivate owner to reduce the price. So I will add a columnt indicating the number of days ad has been on the website. Data was scraped on 18.01.2020.

# In[602]:


df['ad_duration'] = pd.to_datetime('2020-01-18') - df['ad_date']


# In[603]:


df.head()


# In[604]:


df['ad_duration'].isna().any()


# In[605]:


df.ad_duration = df.ad_duration.astype('str')


# In[606]:


df.ad_duration.dtype


# In[607]:


df.ad_duration = df.ad_duration.replace('days','',regex=True)


# In[608]:


df.head()


# In[609]:


df.ad_duration[0]


# In[610]:


df.ad_duration = df.ad_duration.str.split(' ', expand=True)


# In[611]:


df.head()


# In[612]:


df.ad_duration = df.ad_duration.astype('int64')


# In[613]:


print(df.ad_duration.mean())
print(df.ad_duration.median())


# In[614]:


d = df.ad_duration
plt.figure(figsize=(10,6))
sns.distplot(d).set_title('Frequency Distribution Plot of Ad Duration')


# In[615]:


e = d[d<50]


# In[616]:


plt.figure(figsize=(10,6))
sns.distplot(e).set_title('Frequency Distribution Plot of Ad Duration (<50)')


# In[617]:


plt.figure(figsize=(8,5))
sns.boxplot(y='ad_duration', data=df, width=0.5)


# ## Location

# In[618]:


df[['price','city']].groupby(['city']).mean().sort_values(by='price', ascending=False).head()


# In[619]:


a = df.city.value_counts()[:10]


# In[620]:


df.city.value_counts()[:10].sum()


# In[621]:


df.city.value_counts().sum()


# In[622]:


df_location = pd.DataFrame({"count": a , "share": a/6726})


# In[623]:


df_location


# In[624]:


df_location.share.sum()


# 62% percent of all ads are in top 10 cities.

# ## Color

# In[625]:


df.color.value_counts()


# In[626]:


df.color.value_counts().sum()


# In[627]:


c = df.color.value_counts()[:10]


# In[628]:


df_color = pd.DataFrame({"count": c , "share": c/6726})
df_color


# In[629]:


df_color.share[:3].sum()


# It seems like the optimal choice of color is white for wolkswagen polo. More than half of the cars are white followed by red and black. Top 3 colors cover 72% of all cars.

# ## Year

# In[630]:


df.year.value_counts()[:10]


# The age of the car definitely effects the prices. However, instead of the model year of the car, it makes more sense to use is as age. So I will substiture 'year' column from current year.

# In[631]:


df['age'] = 2020 - df['year']


# In[632]:


df.head()


# In[633]:


a = df.age
plt.figure(figsize=(10,6))
sns.distplot(a).set_title('Frequency Distribution Plot of Age of the Cars')


# ## km

# In[523]:


print(df.km.mean())
print(df.km.median())


# In[524]:


k = df.km
plt.figure(figsize=(10,6))
sns.distplot(k).set_title('Frequency Distribution Plot of Km')


# ## ad_title

# In[279]:


conda install -c https://conda.anaconda.org/conda-forge wordcloud


# In[280]:


from wordcloud import WordCloud, STOPWORDS 


# In[525]:


len(df.ad_title)


# In[526]:


df.ad_title[6726]


# In[527]:


text_list = list(df.ad_title)


# In[528]:


text = '-'.join(text_list)


# In[531]:


wordcloud = WordCloud(background_color='white').generate(text)


# In[532]:


plt.figure(figsize=(10,6))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()


# In[536]:


stopwords = ['VW', 'VOLKSWAGEN', 'POLO', 'MODEL', 'KM']


# In[537]:


wordcloud = WordCloud(stopwords=stopwords).generate(text)


# In[538]:


plt.figure(figsize=(10,6))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()


# ## model

# In[634]:


df.model.value_counts()


# Model column includes three different kinds of information: engine size, fuel type and variant. After checking the values, I found out that only engine size information is complete for all cells. Fuel type and variant are missing for most of the cells so I created a separate column for engine size.

# In[635]:


df.model[0]


# In[636]:


#remove spaces
df.model = df.model.replace(' ','',regex=True)


# In[637]:


engine = [x[:3] for x in df.model]


# In[638]:


engine[:5]


# In[639]:


df['engine'] = engine


# In[640]:


df.head()


# In[641]:


df.engine.value_counts()


# In[642]:


df.engine = df.engine.astype('float64')


# In[643]:


df[['engine','price']].groupby(['engine']).mean().sort_values(by='price', ascending=False)


# In[644]:


df[['engine','age','km']].groupby(['engine']).mean().sort_values(by='age', ascending=False)


# # Regression Model

# ### Feature selection

# In[645]:


df.head()


# In[646]:


plt.figure(figsize=(10,6))
sns.regplot(x='km', y='price', data=df).set_title('Km vs Price')


# In[647]:


df.shape


# In[648]:


df = df[df.km < 400000]


# In[649]:


df.shape


# In[650]:


plt.figure(figsize=(10,6))
sns.regplot(x='km', y='price', data=df).set_title('Km vs Price')


# In[651]:


plt.figure(figsize=(10,6))
sns.regplot(x='age', y='price', data=df).set_title('Age vs Price')


# In[652]:


df = df[df.age < 30]


# In[653]:


plt.figure(figsize=(10,6))
sns.regplot(x='age', y='price', data=df).set_title('Age vs Price')


# In[654]:


plt.figure(figsize=(10,6))
sns.regplot(x='ad_duration', y='price', data=df).set_title('Ad Duration vs Price')


# In[655]:


plt.figure(figsize=(10,6))
sns.regplot(x='engine', y='price', data=df).set_title('Engine vs Price')


# ### Correlation matrix

# In[656]:


print(df.corr())


# In[657]:


corr = df.corr()


# In[659]:


plt.figure(figsize=(10,6))
sns.heatmap(corr, vmax=1, square=True)


# ### Linear regression model

# In[384]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
from sklearn.metrics import r2_score


# In[682]:


X = df[['age','km','engine','ad_duration']]
y = df['price']


# In[683]:


X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)


# In[697]:


linreg = LinearRegression()


# In[698]:


linreg.fit(X_train, y_train)


# In[665]:


scores = cross_val_score(linreg, X_train, y_train, cv=5)
print(np.mean(scores))


# In[675]:


scores_test = cross_val_score(linreg, X_test, y_test, cv=5)
print(np.mean(scores_test))


# In[699]:


linreg.score(X_train, y_train)


# In[700]:


linreg.score(X_test, y_test)


# In[703]:


plt.figure(figsize=(10,6))
sns.residplot(x=y_pred, y=y_test)


# ### random forest regressor

# In[691]:


from sklearn.ensemble import RandomForestRegressor


# In[754]:


regr = RandomForestRegressor(max_depth=10, random_state=0, n_estimators=10)


# In[765]:


regr.fit(X_train, y_train)


# In[766]:


print('R-squared score (training): {:.3f}'
     .format(regr.score(X_train, y_train)))


# In[767]:


print('R-squared score (training): {:.3f}'
     .format(regr.score(X_test, y_test)))


# In[779]:


regr.predict([[4,75000,1.2,1]])


# ### withoud ad_duration

# In[759]:


X1 = df[['age','km','engine']]
y1 = df['price']


# In[769]:


regr1 = RandomForestRegressor(max_depth=10, random_state=0, n_estimators=10)


# In[770]:


X_train1, X_test1, y_train1, y_test1 = train_test_split(X1, y1, random_state=42)


# In[771]:


regr1.fit(X_train1, y_train1)


# In[773]:


print('R-squared score (training): {:.3f}'
     .format(regr1.score(X_train1, y_train1)))


# In[774]:


print('R-squared score (training): {:.3f}'
     .format(regr1.score(X_test1, y_test1)))


# In[ ]:




