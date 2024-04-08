from Ingest_data import df
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
import pandas as pd


# separating the target variable and the features
x = df.drop(['Electric Range','Base MSRP','DOL Vehicle ID','Legislative District'],axis=1) #try using fillna by mean of it(Base MSRP)
y = df['Electric Range']

# splitting the data into training and testing sets
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)


# specifying column to impute missing values
impute_col_most_freq = ['County','City','Postal Code','Vehicle Location','Electric Utility','2020 Census Tract']#try removing legislative district(so many null values, very skewed)

# specifying columns for one hot encoding
categorical_col_encoding = ['VIN (1-10)','County','City','State','Postal Code','Make','Model','Electric Vehicle Type','Clean Alternative Fuel Vehicle (CAFV) Eligibility','Vehicle Location','Electric Utility','2020 Census Tract']# try ,'Model Year' later


# Imputing missing values using most frequent strategy
trf1 = ColumnTransformer([('imputing_most_freq',SimpleImputer(strategy='most_frequent'),[1, 2, 4, 10, 11, 12])],remainder='passthrough')

# doing categorical encoding
trf2 = ColumnTransformer([
    ('ohe',OneHotEncoder(sparse_output=False,handle_unknown='ignore'),[0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 12])],remainder='passthrough')


#try StandardScaler over numerical columns

# df['Electric Range'] = StandardScaler().fit_transform(df['Electric Range'].values.reshape(-1,1))


