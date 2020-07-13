#!/usr/bin/env python
# coding: utf-8

# #  Predicting real estate prices.

# ### Data description and task statement
# The task is to build a model that determines the probability of" default " of the borrower. Two tables are provided for training the model. In ACCOUNTS - the customer's Id and its history (payment history, legal status of the person, loan term, loan purpose, etc.), in CUSTOMERS-the account Id and whether or not the default occurred. Also, a breakdown is made for which clients to train the model and for which they will be checked.
# 

# ### Importing libraries

# In[3]:


import pandas as pd
import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as sts
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
import statistics
import warnings
warnings.filterwarnings('ignore')

#import machine learning
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler 

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris 

from sklearn.tree import DecisionTreeClassifier

from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from sklearn.externals import joblib #save model

from xgboost import XGBClassifier
import xgboost

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import precision_score, f1_score, recall_score
from sklearn.metrics import roc_auc_score, roc_curve


# ### Getting the values and viewing the tables.

# 

# In[4]:


#SampleAccounts  = pd.read_csv("C:/Users/User/Desktop/python/Обучение/dataset/nedv_test_t.csv", delimiter=';');
#SampleCustomers  = pd.read_csv("C:/Users/User/Desktop/python/Обучение/dataset/nedv_train_t.csv", delimiter=';');
SampleAccounts  = pd.read_excel("C:/Users/User/Desktop/python/Обучение/dataset/nedv_test_t.xlsx");
SampleCustomers  = pd.read_excel("C:/Users/User/Desktop/python/Обучение/dataset/nedv_train_t.xlsx");

# соединим две таблицы.
#data = pd.concat([y_2018,y_2019],sort=False)
#data


# 

# In[5]:


data = SampleCustomers.copy()
#data


# ###  The analyzed table.

# In[6]:


# sns.heatmap(data.corr(),annot=True,cmap='RdYlGn',linewidths=0.2)
# plt.title('Title', y=1.75, size=15)
# fig=plt.gcf()
# fig.set_size_inches(25,25)
# plt.show()


# In[7]:


#data.corr()
correlations_data = data.corr()['цена_продажа'].sort_values()
#correlations_data


# In[8]:


# процент пропущенный значений
for col in data.columns:
        pct_missing = np.mean(data[col].isnull())
        if pct_missing  != 0:
            print('{} - {}%'.format(col, round(pct_missing*100)))            


# In[146]:


# может удалить в цикле с пропусками более 50 код в Счастье.


# Immediately delete those whose correlation is close to zero. BUT remember that there will be categorical data and so on, so we will delete it wisely.

# Deleting extra columns

# In[9]:


# на всякий случай сделаем копию датафрейма 

df_main = data
drop_elements = ['вместимость дет.садов', 'кол-во школьников', 'вместимость школ', 
                 'число школ в рейтинге ТОП 20 школ москвы','кол-во дошкольников']
df_main = df_main.drop(drop_elements, axis=1)

# И еще удалим  признаки, у которых: 
# - очень слабые "связи", и они могут испортить наши модели, внеся шум.

drop_elements2 = ['id','timestamp','кол-во больничных коек на райоене']
df_main = df_main.drop(drop_elements2, axis=1)
data=df_main


# In[10]:


data


# We can rename columns.

# In[11]:


#! pip install transliterate
import transliterate
#numerics = numerics.rename(columns=lambda x: x+'x') # Добавит х к названиям стоблцов
numerics = data.copy()

result = [] # созадем серию чтобы к ней прибалять result.append(col)
for col in numerics.columns:
    col = col.replace(' ','_')[:15]
    col = col.replace('-','_')
    col = col.replace('ь','')
    col = col.replace('ъ','')
    col=transliterate.translit(col, 'ru', reversed=True)
    result.append(col) 
numerics.columns = result
numerics.columns 


# In[12]:


# для боевых задач быстрее так
#data.columns = ['A' + str(i) for i in range(1, 16)] + ['class']


# ### Remove the emissions

# In[13]:


data.columns


# In[14]:


#data[data["жилая площадь"].isnull()]
#data.describe().T
#column_list = ["score","gdp","social","healthy","freedom","generosity","corruption"]
numerics=data
for col in numerics.columns:
    if(numerics[col].dtype == np.float64 or numerics[col].dtype == np.int64):
        sns.boxplot(x = numerics[col])
        plt.xlabel(col)
        plt.show()


# In[15]:


data.drop(data[((data['площадь'] < 20) | (data['площадь'] > 170)) & (data['площадь'] !=0)].index, inplace = True) 
data.drop(data[((data['жилая площадь'] < 17) | (data['жилая площадь'] > 150)) & (data['жилая площадь']!=0)].index, inplace = True) 
data.drop(data[((data['этаж'] < 1) | (data['этаж'] > 25)) & (data['этаж']!=0)].index, inplace = True) 
data.drop(data[((data['год постройки'] < 1940) | (data['год постройки'] > 2020)) & (data['год постройки']!=0)].index, inplace = True) 
data.drop(data[(data['макс этажи'] > 25) & (data['макс этажи']!=0)].index, inplace = True)  
data.drop(data[((data['площадь кухни'] < 5) | (data['площадь кухни'] > 20)) & (data['площадь кухни']!=0)].index, inplace = True)
data.drop(data[((data['число комнат'] < 1) | (data['число комнат'] > 5)) & (data['число комнат']!=0)].index, inplace = True)

# В каждом случае можем посмотредь диаграмму
data['жилая площадь']


# In[16]:


#data[data["жилая площадь"].isnull()]
#data.describe().T
#column_list = ["score","gdp","social","healthy","freedom","generosity","corruption"]
numerics=data
for col in numerics.columns:
    if(numerics[col].dtype == np.float64 or numerics[col].dtype == np.int64):
        sns.boxplot(x = numerics[col])
        plt.xlabel(col)
        plt.show()


# In[ ]:





# ### Clearing data

# In[17]:


# процент пропущенный значений
for col in data.columns:
        pct_missing = np.mean(data[col].isnull())
        if pct_missing  != 0:
            print('{} - {}%'.format(col, round(pct_missing*100)))


# #### Where there is a dependency of omitted with other signs, we calculate. Where there isn't just put 0 or the median

# In[18]:


# жил площадь связана с площадью. надйем коэфициент
koef=data['площадь'].mean(axis=0)/data['жилая площадь'].mean(axis=0) # коэфициент зависимости
print (data['площадь'].mean(axis=0),data['жилая площадь'].mean(axis=0),koef )
data["жилая площадь"]= data["жилая площадь"].fillna(data["площадь"]/koef)


# Fill in the missing values for categorical features with the most common ones, and for numeric ones with zeros or averages.
# Next, we will experiment whether 0 or the median is better.

# #### First fill in the zeros, then the median. after training the model we look at what ekxit

# In[31]:


#df.final_pmt_date.fillna(0, inplace=True)
#numerical_columns = numerical_columns.fillna(numerical_columns.median(axis=0), axis=0)
#df.fillna(0, inplace=True) 
# категориельные наиболее частыми
data_describe = data.describe(include=[object])
for c in data_describe:
    data[c] = data[c].fillna(data_describe[c]['top'])
    
# числовые нулями   
data.fillna(0, inplace=True) 


# Проверяем

# In[32]:


# процент пропущенный значений
for col in data.columns:
        pct_missing = np.mean(data[col].isnull())
        if pct_missing  != 0:
            print('{} - {}%'.format(col, round(pct_missing*100)))


# ### we divide it into categories

# In[33]:


categorical_columns = [c for c in data.columns if data[c].dtype.name == 'object']
numerical_columns   = [c for c in data.columns if data[c].dtype.name != 'object']
print ("Cat_")
print (categorical_columns)
print ("Num_")
print (numerical_columns)
data[categorical_columns].describe()


# In[34]:


for c in categorical_columns:
    print (data[c].unique())


# In[35]:


data['есть ли культурный объект, входящий ТОП 25 достопримечательностей Москвы'].value_counts(dropna=False) 


# Разобьем на бинарные и прочие >2

# In[36]:


binary_columns    = [c for c in categorical_columns if data_describe[c]['unique'] == 2]
nonbinary_columns = [c for c in categorical_columns if data_describe[c]['unique'] > 2]
print (binary_columns, nonbinary_columns)


# In[37]:


data['есть ли культурный объект, входящий ТОП 25 достопримечательностей Москвы'].value_counts(dropna=False) 


# In[38]:


#data = SampleCustomers.copy()


# быстрый но некоретный способ

# In[39]:


for c in binary_columns:
    top = data_describe[c]['top']
    top_items = data[c] == top
    data.loc[top_items, c] = 0
    data.loc[np.logical_not(top_items), c] = 1
data[binary_columns]


# Более правльный вариант разбивки. На рабочей моделе четко указаны поля и вариации.

# In[41]:



df = data

# половой признак переведем в числа

# df['тип жилья'] = df['тип жилья'].map({'Investment': 0, 
#                                  'OwnerOccupier': 1}).astype(int)

# С помощью цилка для колонок с 2-мя категориями: 'yes' или 'no'

# list_yes_no = ['есть ли культурный объект, входящий ТОП 25 достопримечательностей Москвы']
# for column in list_yes_no:
#     df[column] = df[column].map({'no': 0, 
#                                  'yes': 1}).astype(int)
    
#df['timestamp']=df['timestamp'].dt.year если нужно в дату

#там где много признаков/ разбивает в ряд. не добавляет новые. делать если до 5 кат.   

le = preprocessing.LabelEncoder()
le.fit(df['район'])
df['район']=le.transform(df['район'])
#list(le.inverse_transform(df['район'])) # обратное декодирование
#https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelEncoder.html
le.fit(df['год постройки'])
df['год постройки']=le.transform(df['год постройки'])

# from sklearn import preprocessing 
# label_encoder = preprocessing.LabelEncoder() 
# label_encoder.fit(df['район']) 
# label_encoder.transform(df['район']) 

# районов очень много. добаляем стоблцы.
data_nonbinary = pd.get_dummies(df['район'])
print (data_nonbinary.columns)

# надо учесть что стоблец район надо выкинут.
#  соединяем все в один 

data = pd.concat((data_numerical, data[binary_columns], data_nonbinary), axis=1)
data = pd.DataFrame(data, dtype=float)

data


# In[ ]:





# ### Normalize

# In[42]:


dataframe=data.copy()
#dataframe

#Standardization
# scale_features_std = StandardScaler() 
# array = dataframe['цена_продажа'].values.reshape(-1, 1)
# features_train = scale_features_std.fit_transform(array) 
# features_test = scale_features_std.transform(array) 
# dataframe['цена_продажа']=features_train

# Feature scaling with MinMaxScaler 
# from sklearn.preprocessing import MinMaxScaler 
# scale_features_mm = MinMaxScaler() 
# features_train = scale_features_mm.fit_transform(features_train) 
# features_test = scale_features_mm.transform(features_test) 

dataframe['цена_продажа']


# In[43]:



array = dataframe['цена_продажа'].values.reshape(-1, 1)
#data_scaler = StandardScaler().fit(array)
#data_rescaled = data_scaler.transform(array)
#data_rescaled2 = MinMaxScaler().fit_transform(array)

min_max_scaler = preprocessing.MinMaxScaler(array)
dataframe['цена_продажа'] = MinMaxScaler().fit_transform(array)
# dataframe['цена_продажа']
#dataframe['цена_продажа']=round(dataframe['цена_продажа'],1)+2
#dataframe['цена_продажа']=dataframe['площадь индустриальной зоны']
dataframe['цена_продажа']


# In[44]:


data_numerical = data[numerical_columns]
dataframe = (data_numerical - data_numerical.mean()) / data_numerical.std()
#dataframe['площадь']=dataframe['площадь'].astype("int")
dataframe['площадь']


# In[45]:


# data = pd.concat((data_numerical, data[binary_columns]), axis=1)
# data = pd.DataFrame(data, dtype=float)
# data


# In[ ]:





# In[168]:


df=dataframe

X = df.drop(['цена_продажа'], axis=1)
y = df['цена_продажа']

# зададим параметры для всех обучающих признаков 
TEST_SIZE = 0.3 
RAND_STATE = 42
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = TEST_SIZE, random_state=RAND_STATE)
#df.info()


# In[169]:


tree = DecisionTreeClassifier(criterion='entropy',max_depth=4,random_state=40)
tree.fit(X_train, y_train)
predicted_y = tree.predict(X_test)
print('Accuracy: {:.2f}'.format(tree.score(X_test, y_test)))
y_test_preds_tree = tree.predict(X) # здесь будет храниться весь массив предсказанных ответов


# In[ ]:


# SVC
SVC_model = SVC()  
SVC_model.fit(X_train, y_train)
predicted_y = SVC_model.predict(X_test)
print('Accuracy: {:.2f}'.format(SVC_model.score(X_test, y_test)))     
# здесь будет храниться весь массив предсказанных ответов
y_test_preds_SVC_model = SVC_model.predict(X) # здесь будет храниться весь массив предсказанных ответов


# In[ ]:


# GaussianNB
clf = GaussianNB()
clf.fit(X_train, y_train)
predicted_y = clf.predict(X_test)
y_test_preds_clf = clf.predict(X) # здесь будет храниться весь массив предсказанных ответов
print('Accuracy: {:.2f}'.format(clf.score(X_test, y_test)))


# In[ ]:


tree = DecisionTreeClassifier(criterion='entropy',max_depth=4,random_state=40)
tree.fit(X_train, y_train)
predicted_y = tree.predict(X_test)
print('Accuracy: {:.2f}'.format(tree.score(X_test, y_test)))
y_test_preds_tree = tree.predict(X) # здесь будет храниться весь массив предсказанных ответов


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[12]:


# или так
data['тип жилья'].value_counts(dropna=False) 


# In[ ]:


# удалим нулевой столбец, он не нужен
df = data
#df = df.drop(columns='id')

# половой признак переведем в числа
df['тип жилья'] = df['тип жилья'].map({'Investment': 0, 
                                 'OwnerOccupier': 1}).astype(int)

# С помощью цилка для колонок с 2-мя категориями: 'yes' или 'no'
list_yes_no = ['есть ли культурный объект, входящий ТОП 25 достопримечательностей Москвы']
for column in list_yes_no:
    df[column] = df[column].map({'no': 0, 
                                 'yes': 1}).astype(int)
    
#df['timestamp']=df['timestamp'].dt.year если нужно в дату

#там где много признаков    
le = preprocessing.LabelEncoder()
le.fit(df['район'])
df['район']=le.transform(df['район'])
#list(le.inverse_transform(df['район'])) # обратное декодирование
#https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelEncoder.html


# In[14]:


binary_columns    = [c for c in categorical_columns if data_describe[c]['unique'] == 2]
nonbinary_columns = [c for c in categorical_columns if data_describe[c]['unique'] > 2]
print (binary_columns, nonbinary_columns)


# In[ ]:





# In[ ]:





# In[ ]:





# In[121]:


# процент пропущенный значений
data=SampleCustomers
for col in data.columns:
        pct_missing = np.mean(data[col].isnull())
        if pct_missing  != 0:
            print('{} - {}%'.format(col, round(pct_missing*100)))


# Из жизни мы знаем что: 
# 
# -вместимость дет.садов - 21.0%
# -вместимость школ - 21.0%
# -кол-во больничных коек на райоене - 47.0%
# 
# все это второстепенные показатели. Просто заполним нулями. Хотя сопоставив районы с непропущенными данными можно былобы найти данные довольно точно.
# 
# Важные показатели, пропуски будем запонять осмысленно.
# 
# -жилая площадь - 20.0%
# -этаж - 1.0%
# -макс этажи - 38.0%
# -из чего изготовлен - 38.0%
# -год постройки - 50.0%
# -число комнат - 38.0%
# -площадь кухни - 38.0% 
# 
# -жилая площадь - сильна привязана к площади, запоним на основании ее.
# 
# Найдем закономерность, насколько жалая площадь зависит от общей. Для этого найдем срденюю жилую и среднюю общую площадь и просто вычислим коффэфициент. 
# 
# НО перед этим проверим данные на выбросы. Чтобы построить диаграмму посмотрим на типы данных.

# In[122]:


data.info()


# In[123]:


#! pip install transliterate
import transliterate
#numerics = numerics.rename(columns=lambda x: x+'x') # Добавит х к названиям стоблцов
numerics = data.copy()

result = [] # созадем серию чтобы к ней прибалять result.append(col)
for col in numerics.columns:
    col = col.replace(' ','_')[:15]
    col = col.replace('-','_')
    col = col.replace('ь','')
    col = col.replace('ъ','')
    col=transliterate.translit(col, 'ru', reversed=True)
    result.append(col) 
numerics.columns = result
numerics.columns 


# In[124]:


#data[data["жилая площадь"].isnull()]
#data.describe().T
#column_list = ["score","gdp","social","healthy","freedom","generosity","corruption"]
numerics=data
for col in numerics.columns:
    if(numerics[col].dtype == np.float64 or numerics[col].dtype == np.int64):
        sns.boxplot(x = numerics[col])
        plt.xlabel(col)
        plt.show()
    


# In[ ]:





# Довольно печальная картинка, выбросов много и они большие. Это и площадь в 5000 кв/м и 100 этажные здания, что конечно бывает, но нам мешает.

# In[125]:


diap = data.drop(data[(data['площадь'] < 20) | (data['площадь'] > 120)].index)  # удаление в дипазоне

plt.hist(diap['площадь'].dropna(), bins = 10, edgecolor = 'k');
plt.xlabel('площадь'); plt.ylabel('площадь');
plt.title('Title');
data=diap


# In[126]:


diap = data.drop(data[(data['жилая площадь'] < 20) | (data['жилая площадь'] > 150)].index)  # удаление в дипазоне

plt.hist(diap['жилая площадь'].dropna(), bins = 10, edgecolor = 'k');
plt.xlabel('жилая площадь'); plt.ylabel('жилая площадь');
plt.title('Title');

data=diap


# In[127]:


diap = data.drop(data[(data['этаж'] < 1) | (data['этаж'] > 25)].index)  # удаление в дипазоне

plt.hist(diap['этаж'].dropna(), bins = 10, edgecolor = 'k');
plt.xlabel('этаж'); plt.ylabel('count');
plt.title('Title');
data=diap


# In[128]:


diap =data.drop(data[(data['год постройки'] < 1940) | (data['год постройки'] > 2020)].index)  # удаление в дипазоне
plt.hist(diap['год постройки'].dropna(), bins = 10, edgecolor = 'k');
plt.xlabel('год постройки'); plt.ylabel('count');
plt.title('Title');
data=diap


# In[129]:


data = data.drop(data[(data['макс этажи'] < 1) | (data['макс этажи'] > 25)].index)  # удаление в дипазоне
data = data.drop(data[(data['площадь кухни'] < 5) | (data['площадь кухни'] > 20)].index)
data = data.drop(data[(data['число комнат'] < 1) | (data['число комнат'] > 5)].index)


# final_pmt_date - 7.0%  не так много. Можно обыло бы удалить, но в наборе есть еще FACT_CLOSE_DATE -финальная дата платежа. Сделаем так. Если значения final_pmt_date пропущено а FACT_CLOSE_DATE есть то прировняем значения, а если нет то удалим. 

# In[130]:


numerics=data
for col in numerics.columns:
    if(numerics[col].dtype == np.float64 or numerics[col].dtype == np.int64):
        sns.boxplot(x = numerics[col])
        plt.xlabel(col)
        plt.show()


# In[131]:


koef=data['площадь'].mean(axis=0)/data['жилая площадь'].mean(axis=0) # коэфициент зависимости
print (data['площадь'].mean(axis=0),data['жилая площадь'].mean(axis=0),koef )
data["жилая площадь"]= data["жилая площадь"].fillna(data["площадь"]/koef)


# Было 7% стало 6% не многого добились. Удлим эти 6%

# Посмотрим корреляцию на данном этапе.

# In[132]:


import seaborn as sns
sns.heatmap(data.corr(),annot=True,cmap='RdYlGn',linewidths=0.2)
plt.title('Title', y=1.75, size=15)
fig=plt.gcf()
fig.set_size_inches(25,25)
plt.show()


# Видим что все что связано со школой сильно коррелирует между собой. Оставим один признак, тот что наиболее связан с ценой -это количествео школ. Остальные удалим.

# In[133]:


data.columns


# Если данные содержат пропущенные значения, то имеется две простые альтернативы:
# 
# удалить столбцы с такими значениями (data = data.dropna(axis=1)),
# удалить строки с такими значениями (data = data.dropna(axis=0))

# In[134]:


# во избежание избыточности признаков удалим некоторые признаки:
# на всякий случай сделаем копию датафрейма 

df_main = data.copy()
drop_elements = ['вместимость дет.садов', 'кол-во школьников', 'вместимость школ', 
                 'число школ в рейтинге ТОП 20 школ москвы','кол-во дошкольников']
df_main = df_main.drop(drop_elements, axis=1)

# И еще удалим  признаки, у которых: 
# - очень слабые "связи", и они могут испортить наши модели, внеся шум.

drop_elements2 = ['id','timestamp','кол-во больничных коек на райоене']
df_main = df_main.drop(drop_elements2, axis=1)
data=df_main


# In[135]:


import seaborn as sns
sns.heatmap(data.corr(),annot=True,cmap='RdYlGn',linewidths=0.2)
plt.title('Title', y=1.75, size=15)
fig=plt.gcf()
fig.set_size_inches(25,15)
plt.show()


# Переведем категориальный признаки в числовые. Просмотрим данные типа objekt и datetime64

# In[136]:


categorical_columns = [c for c in data.columns if data[c].dtype.name == 'object']
numerical_columns   = [c for c in data.columns if data[c].dtype.name != 'object']
print ("Cat",categorical_columns)
print ("----")
print ("Num",numerical_columns)
data[categorical_columns].describe()


# In[137]:


for c in categorical_columns:
    print (data[c].unique())


# In[138]:


data['тип жилья'].value_counts(dropna=False) 


# In[139]:


data['есть ли культурный объект, входящий ТОП 25 достопримечательностей Москвы'].value_counts(dropna=False) 


# По остальным объектам (дата , районы) понятно что вариаций будем много.

# In[140]:


# удалим нулевой столбец, он не нужен
df = data
#df = df.drop(columns='id')

# половой признак переведем в числа
df['тип жилья'] = df['тип жилья'].map({'Investment': 0, 
                                 'OwnerOccupier': 1}).astype(int)

# С помощью цилка для колонок с 2-мя категориями: 'yes' или 'no'
list_yes_no = ['есть ли культурный объект, входящий ТОП 25 достопримечательностей Москвы']
for column in list_yes_no:
    df[column] = df[column].map({'no': 0, 
                                 'yes': 1}).astype(int)
    
#df['timestamp']=df['timestamp'].dt.year если нужно в дату

#там где много признаков    
le = preprocessing.LabelEncoder()
le.fit(df['район'])
df['район']=le.transform(df['район'])
#list(le.inverse_transform(df['район'])) # обратное декодирование
#https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelEncoder.html


# Однако у нас есть еще два категориальный признака - tenure и MonthlyCharges, количество месяцев пользования услугами и месячный размер оплаты.
# 
# Без анализа видно что месяцев слишком много чтобы разбивать их на категории, тоже и со вторым признаком. Поэтому разобьем их.
# 
# Эксперементальным путем из графика видно, что на большее количество разбивать смысла нет -4 оптимально. (Меняем bins=4 для изучения)
# 

# In[141]:


correlations_data = data.corr()['цена_продажа'].sort_values()
correlations_data


# In[142]:


#from pandas.plotting import scatter_matrix
#scatter_matrix(data, alpha=0.05, figsize=(10, 10));


# In[ ]:





# In[143]:


#df.final_pmt_date.fillna(0, inplace=True)

df = df.fillna(data.median(axis=0), axis=0)
#df.fillna(0, inplace=True) 
df


# In[144]:



# standardize the data attributes
#df['цена_продажа'] = preprocessing.scale(df['цена_продажа'])
df


# In[180]:


# dataframe = df['цена_продажа']
# array = df['цена_продажа'].values.reshape(-1, 1).astype(np.float64)
# data_scaler = StandardScaler().fit(array)
# data_rescaled = data_scaler.transform(array)
# data_rescaled2 = MinMaxScaler().fit_transform(array)

min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0,100))
df['цена_продажа'] = min_max_scaler.fit_transform(df[['цена_продажа']])
df['цена_продажа']


# In[ ]:





# In[172]:


#min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0,1))

np_scaled = min_max_scaler.fit_transform(df[['цена_продажа', 'площадь']])
df_norm = pd.DataFrame(np_scaled)
#print(df_norm[:5])
volume = df_norm[0]
bags = df_norm[1]
plt.plot(volume, bags, 'r.')
plt.show()

from scipy.stats import pearsonr
corr, p_value = pearsonr(df['цена_продажа'], df['площадь'])
print(corr)


# Поверхностный анализ показывает что признаки с пропущенными даннымми и некоторые столбцы не существенно влияют на конечный результат. В боевой задаче мы бы изучили и разбили их на группы но сейчас просто заполним нулями или удалим для упрощения модели.

# ### Приступем к обучению модели
# 

# In[146]:


X = df.drop(['цена_продажа'], axis=1)
y = df['цена_продажа']
# зададим параметры для всех обучающих признаков 
TEST_SIZE = 0.3 
RAND_STATE = 42
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = TEST_SIZE, random_state=RAND_STATE)


# In[619]:


# GradientBoostingClassifier 
gbc = GradientBoostingClassifier()
gbc.fit(X_train, y_train)
predicted_y = gbc.predict(X_test)# здесь будет храниться весь массив предсказанных ответов
print('Accuracy: {:.2f}'.format(gbc.score(X_test, y_test)))

# здесь будет храниться весь массив предсказанных ответов
y_test_preds_gbc = gbc.predict(X) 


# In[620]:


tree = DecisionTreeClassifier(criterion='entropy',max_depth=4,random_state=40)
tree.fit(X_train, y_train)
predicted_y = tree.predict(X_test)
print('Accuracy: {:.2f}'.format(tree.score(X_test, y_test)))
y_test_preds_tree = tree.predict(X) # здесь будет храниться весь массив предсказанных ответов


# In[181]:


# GaussianNB
clf = GaussianNB()
clf.fit(X_train, y_train)
predicted_y = clf.predict(X_test)
y_test_preds_clf = clf.predict(X) # здесь будет храниться весь массив предсказанных ответов
print('Accuracy: {:.2f}'.format(clf.score(X_test, y_test)))


# In[147]:


# LogisticRegression
classifier = LogisticRegression(solver='lbfgs',random_state=40)
classifier.fit(X_train, y_train)
predicted_y = classifier.predict(X_test)
print('Accuracy: {:.2f}'.format(classifier.score(X_test, y_test)))
y_test_preds_classifier = gbc.predict(X) # здесь будет храниться весь массив предсказанных ответов


# In[ ]:





# In[ ]:





# In[ ]:




