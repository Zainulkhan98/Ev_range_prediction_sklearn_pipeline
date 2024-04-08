from Pipeline import pipe
from Feature_engineering import x_test,y_test,x_train
import numpy as np
from sklearn.metrics import r2_score,mean_squared_error,mean_absolute_error
import pandas as pd


# Evaluating the model using mse,rmse,mae,r2_score
pred = pipe.predict(x_test)
r2_score = r2_score(y_test,pred)
mse = mean_squared_error(y_test,pred)
rmse = np.sqrt(mean_squared_error(y_test,pred))
mae = mean_absolute_error(y_test,pred)


print('r2_score',r2_score)
print('mse',mse)
print('rmse',rmse)
print('mae',mae)

# For converting the x_train data to csv file
# x_train_data = pd.DataFrame(x_train)
# x_train_data.to_csv('x_train_data.csv', index=False)

