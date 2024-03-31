import pandas as pd
from sklearn.model_selection import train_test_split


dfl = pd.read_csv('Electric_Vehicle_Population_Data.csv')
df = pd.DataFrame(dfl.loc[:50000,:])
#
x = df.drop(['Electric Range','Base MSRP','DOL Vehicle ID','Legislative District'],axis=1) #try using fillna by mean of it(Base MSRP)
y = df['Electric Range']


x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)

impute_col_most_freq = ['County','City','Postal Code','Vehicle Location','Electric Utility','2020 Census Tract']
impute_col_median = ['Legislative District']#try removing legislative district(so many null values, very skewed)

# specifying columns for one hot encoding
categorical_col_encoding = ['VIN (1-10)','County','City','State','Postal Code','Make','Model','Electric Vehicle Type','Clean Alternative Fuel Vehicle (CAFV) Eligibility','Vehicle Location','Electric Utility','2020 Census Tract']# try ,'Model Year' later

col_index = [x_train.columns.get_loc(col) for col in categorical_col_encoding]
print(col_index)

# print(df.dtypes)


# sns.boxplot(
#     x = df.index,
#     y = y,
#     showmeans=True,
#     data=df
# )
# plt.xlabel("Data Point")
# plt.ylabel("Electric Range")
# plt.title("Distribution of Electric Range with Outliers")
# plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
# plt.tight_layout()
#
# plt.show()