import cleaning_dataset as cl

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import pickle 

# # TRAIN TEST SPLIT
df_final = cl.cleaning_dataframe(cl.df, df_giraffe= True)

y = df_final['price']
X = df_final.drop(['price', 'id'], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Cleaning X_train with the imputations 
stats_from_X_train = cl.stats(X_train)
X_train_clean = cl.transform_cleaning_traintestsplit(X_train, stats_from_X_train)

# Cleaning X_test with the imputations from X_train
X_test_clean  = cl.transform_cleaning_traintestsplit(X_test, stats_from_X_train)

# Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_clean)
X_test_scaled = scaler.transform(X_test_clean)


# # XGBOOST MODEL
xgb_model = xgb.XGBRegressor(n_estimators=3000, random_state=43, learning_rate=0.05, subsample= 0.8)
xgb_model.fit(X_train_scaled, y_train)
y_pred = xgb_model.predict(X_test_scaled)

r2 = r2_score(y_test, y_pred)
print("R2 Score:", r2)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print("Root Mean Squared Error (RMSE):", rmse, "€")

rae = mean_absolute_error(y_test, y_pred)
print("Mean Absolute Error (RAE):", rae, "€")

mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)


# # Save the model and the scaler
with open("model_and_scaler/scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

with open("model_and_scaler/model.pkl", "wb") as f:
    pickle.dump(xgb_model, f)


