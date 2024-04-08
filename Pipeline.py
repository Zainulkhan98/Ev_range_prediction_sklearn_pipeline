from sklearn.pipeline import Pipeline
from Feature_engineering import trf1,trf2
from Modeling import model
from Feature_engineering import x_train,y_train
from sklearn import set_config
import pandas as pd


# Defining the pipeline
pipe = Pipeline(steps=[('imputing_most_freq',trf1),
                ('ohe',trf2),
                ('model',model)
                ])

# Fitting the pipeline
pipe.fit(x_train,y_train)

# For visualizing the pipeline
set_config(display='diagram')
