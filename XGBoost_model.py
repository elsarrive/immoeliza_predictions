import cleaning_dataset as cl
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import xgboost as xgb
import pickle 


# # XGBOOST MODEL
xgb_model = xgb.XGBRegressor(n_estimators=3000, random_state=43, learning_rate=0.05, subsample= 0.8)
xgb_model.fit(cl.X_train_scaled, cl.y_train)
y_pred = xgb_model.predict(cl.X_test_scaled)

r2 = r2_score(cl.y_test, y_pred)
print("R2 Score:", r2)

rmse = np.sqrt(mean_squared_error(cl.y_test, y_pred))
print("Root Mean Squared Error (RMSE):", rmse, "€")

rae = mean_absolute_error(cl.y_test, y_pred)
print("Mean Absolute Error (RAE):", rae, "€")

mse = mean_squared_error(cl.y_test, y_pred)
print("Mean Squared Error:", mse)


# # Save the model and the scaler
with open("model_and_scaler/scaler.pkl", "wb") as f:
    pickle.dump(cl.scaler, f)

with open("model_and_scaler/model.pkl", "wb") as f:
    pickle.dump(xgb_model, f)
