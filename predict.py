"""XGBoost on every numeric column, trained on log hours."""

import numpy as np
from xgboost import XGBRegressor
import pandas as pd

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
columns = [c for c in test.columns if c != "project_id" and pd.api.types.is_numeric_dtype(test[c])]
model = XGBRegressor(n_estimators=400, max_depth=6, learning_rate=0.05, subsample=0.8)
model.fit(train[columns], np.log1p(train["hours_spent"]))
guess = np.expm1(model.predict(test[columns])).clip(0)
pd.DataFrame({"project_id": test["project_id"], "hours_spent": guess}).to_csv("predictions.csv", index=False)
