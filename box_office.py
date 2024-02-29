from __future__ import unicode_literals
import numpy as np
import re
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from collections import Counter
from math import *
from sklearn.model_selection import cross_validate
from sklearn.model_selection import cross_val_score
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTEENN
from sklearn.linear_model import LinearRegression,Ridge,Lasso,RidgeCV
from sklearn.ensemble import RandomForestRegressor,BaggingRegressor,GradientBoostingRegressor,AdaBoostRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestCentroid
from sklearn.metrics import classification_report, f1_score,accuracy_score ,confusion_matrix
from sklearn.linear_model import LinearRegression
from sklearn.metrics import confusion_matrix
from collections import Counter
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import IsolationForest
import matplotlib.gridspec as gridspec
from sklearn.impute import KNNImputer
import seaborn as sns
import warnings
from warnings import simplefilter
import ast
simplefilter(action='ignore', category=FutureWarning)
pd.options.mode.chained_assignment = None  # default='warn'
from warnings import simplefilter
# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)
pd.set_option('display.max_columns', None)
import warnings
with warnings.catch_warnings():
    warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)




dict_columns = ['belongs_to_collection', 'genres', 'production_companies',
                'production_countries', 'spoken_languages', 'Keywords', 'cast', 'crew']
def text_to_dict(df):
    for column in dict_columns:
        df[column] = df[column].apply(lambda x: {} if pd.isna(x) else ast.literal_eval(x) )
    return df

train_df = text_to_dict(pd.read_csv('./files/train.csv'))
test_df = text_to_dict(pd.read_csv('./files/test.csv'))

train_df.head()
train_df.info()
train_df.isna().sum()
train_df.shape[0]
train_df[["belongs_to_collection","title"]].sample(10)


for index,row in train_df.iterrows():
    if row["belongs_to_collection"] != {}:
        for i in row["belongs_to_collection"]:
            train_df["belongs_to_collection"][index] = i["name"]
    else:
        title = row["title"]
        if (title.split()[0].isdigit()) and (len(title.split()) >= 1 ) :
            train_df["belongs_to_collection"][index] = (title.strip())+" Collection"
        else:
            regex = r"([a-zA-Z \d \, \[ \] \' \. \& \( \) \_ \! \? \x00-\x7F]*)"
            train_df["belongs_to_collection"][index] = ((re.findall(regex,title))[0]) + " Collection"

train_df.replace(0,train_df["budget"].mean(),inplace=True)
#####
test_df.replace(0,test_df["budget"].mean(),inplace=True)

train_df[train_df["budget"] == 0]


train_df["genres"] = train_df["genres"].apply(lambda x: train_df["genres"].mode()[0] if x=={} else x)
test_df["genres"] = test_df["genres"].apply(lambda x: test_df["genres"].mode()[0] if x=={} else x)

list_of_genres = list(train_df['genres'].apply(lambda x: [i['name'] for i in x] if x != {} else []).values)
train_df["genre_count"] = train_df["genres"].apply(lambda x: len(x))
train_df['all_genres'] = train_df['genres'].apply(lambda x: ' '.join(sorted([i['name'] for i in x])) if x != {} else '')
top_genres = [m[0] for m in Counter([i for j in list_of_genres for i in j]).most_common(15)]
for g in top_genres:
    train_df['genre_' + g] = train_df['all_genres'].apply(lambda x: 1 if g in x else 0)
test_df["genre_count"] = test_df["genres"].apply(lambda x: len(x))
test_df['all_genres'] = test_df['genres'].apply(lambda x: ' '.join(sorted([i['name'] for i in x])) if x != {} else '')
for g in top_genres:
    test_df['genre_' + g] = test_df['all_genres'].apply(lambda x: 1 if g in x else 0)


train_df[train_df["genres"] == {}]
train_df[["genres","genre_count"]].sample(5)



filler = train_df["production_companies"].value_counts().index.to_list()[1]
train_df["production_companies"] = train_df["production_companies"].apply(lambda x: filler if x=={} else x)
fillert = test_df["production_companies"].value_counts().index.to_list()[1]
test_df["production_companies"] = test_df["production_companies"].apply(lambda x: fillert if x=={} else x)

list_of_companies = list(train_df['production_companies'].apply(lambda x: [i['name'] for i in x] if x != {} else []).values)
train_df["companies_count"] = train_df["production_companies"].apply(lambda x: len(x))
train_df['all_production_companies'] = train_df['production_companies'].apply(lambda x: ' '.join(sorted([i['name'] for i in x])) if x != {} else '')
top_companies = [m[0] for m in Counter([i for j in list_of_companies for i in j]).most_common(30)]
for g in top_companies:
    train_df['production_company_' + g] = train_df['all_production_companies'].apply(lambda x: 1 if g in x else 0)
test_df["companies_count"] = test_df["production_companies"].apply(lambda x: len(x))
test_df['all_production_companies'] = test_df['production_companies'].apply(lambda x: ' '.join(sorted([i['name'] for i in x])) if x != {} else '')
for g in top_companies:
    test_df['production_company_' + g] = test_df['all_production_companies'].apply(lambda x: 1 if g in x else 0)

train_df[train_df["production_companies"] == {}]
train_df["production_countries"] = train_df["production_countries"].apply(lambda x: train_df["production_countries"].mode()[0] if x=={} else x)
test_df["production_countries"] = test_df["production_countries"].apply(lambda x: test_df["production_countries"].mode()[0] if x=={} else x)

list_of_countries = list(train_df['production_countries'].apply(lambda x: [i['name'] for i in x] if x != {} else []).values)
train_df["countres_count"] = train_df["production_countries"].apply(lambda x: len(x)) 
train_df['all_countries'] = train_df['production_countries'].apply(lambda x: ' '.join(sorted([i['name'] for i in x])) if x != {} else '')
top_countries = [m[0] for m in Counter([i for j in list_of_countries for i in j]).most_common(25)]
for g in top_countries:
    train_df['production_country_' + g] = train_df['all_countries'].apply(lambda x: 1 if g in x else 0)
test_df["countres_count"] = test_df["production_countries"].apply(lambda x: len(x)) 
test_df['all_countries'] = test_df['production_countries'].apply(lambda x: ' '.join(sorted([i['name'] for i in x])) if x != {} else '')
for g in top_countries:
    test_df['production_country_' + g] = test_df['all_countries'].apply(lambda x: 1 if g in x else 0)

train_df[["production_countries","countres_count"]]
test_df.isna().sum()
test_df["release_date"].iloc[828] = test_df["release_date"].mode()[0]


train_df["release_date"] = pd.to_datetime(train_df["release_date"])
train_df["release_year"] = train_df["release_date"].dt.year
train_df["release_month"] = train_df["release_date"].dt.month
train_df["release_day"] = train_df["release_date"].dt.day
train_df["release_day_name"] = train_df["release_date"].dt.day_name()
################
test_df["release_date"] = pd.to_datetime(test_df["release_date"])
test_df["release_year"] = test_df["release_date"].dt.year
test_df["release_month"] = test_df["release_date"].dt.month
test_df["release_day"] = test_df["release_date"].dt.day
test_df["release_day_name"] = test_df["release_date"].dt.day_name()

train_df["runtime"] = train_df["runtime"].fillna(train_df["runtime"].mean())
#####
test_df["runtime"] = test_df["runtime"].fillna(test_df["runtime"].mean())


train_df["spoken_languages"] = train_df["spoken_languages"].apply(lambda x: train_df["spoken_languages"].mode()[0] if x=={} else x)
test_df["spoken_languages"] = test_df["spoken_languages"].apply(lambda x: test_df["spoken_languages"].mode()[0] if x=={} else x)

list_of_languages = list(train_df['spoken_languages'].apply(lambda x: [i['name'] for i in x] if x != {} else []).values)
train_df["languages_count"] = train_df["spoken_languages"].apply(lambda x: len(x))
train_df['all_languages'] = train_df['spoken_languages'].apply(lambda x: ' '.join(sorted([i['name'] for i in x])) if x != {} else '')
top_languages = [m[0] for m in Counter([i for j in list_of_languages for i in j]).most_common(30)]
for g in top_languages:
    train_df['language_' + g] = train_df['all_languages'].apply(lambda x: 1 if g in x else 0)
test_df["languages_count"] = test_df["spoken_languages"].apply(lambda x: len(x))
test_df['all_languages'] = test_df['spoken_languages'].apply(lambda x: ' '.join(sorted([i['name'] for i in x])) if x != {} else '')
for g in top_languages:
    test_df['language_' + g] = test_df['all_languages'].apply(lambda x: 1 if g in x else 0)

fi = train_df["Keywords"].value_counts().index.to_list()[1]
train_df["Keywords"] = train_df["Keywords"].apply(lambda x: fi if x=={} else x)
#####
fi3 = test_df["Keywords"].value_counts().index.to_list()[1]
test_df["Keywords"] = test_df["Keywords"].apply(lambda x: fi3 if x=={} else x)


train_df["Keywords_count"] = train_df["Keywords"].apply(lambda x: len(x))
#######3
test_df["Keywords_count"] = test_df["Keywords"].apply(lambda x: len(x))

train_df[["Keywords","Keywords_count"]]
