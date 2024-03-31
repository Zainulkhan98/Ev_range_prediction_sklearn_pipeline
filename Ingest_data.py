import pandas as pd


# Ingesting data from csv file -- Electric_Vehicle_Population_Data.csv
dfl = pd.read_csv('Data/Electric_Vehicle_Population_Data.csv')
df = pd.DataFrame(dfl.loc[:10000,:])  # limiting the data to 10000 rows for faster processing

