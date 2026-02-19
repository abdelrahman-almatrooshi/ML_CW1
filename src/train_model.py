import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor

np.random.seed(42)

train = pd.read_csv('data/CW1_train.csv')
test  = pd.read_csv('data/CW1_test.csv')

y = train['outcome']
X = train.drop('outcome', axis=1)

categorical_cols = ['cut', 'color', 'clarity']
numeric_cols = [c for c in X.columns if c not in categorical_cols]

preprocessor = ColumnTransformer([
    ('cat', OrdinalEncoder(), categorical_cols),
    ('num', 'passthrough', numeric_cols),
])

ensemble = {
    'hgb1': Pipeline([('pre', preprocessor), ('m', HistGradientBoostingRegressor(max_iter=1000, max_depth=3, learning_rate=0.03, min_samples_leaf=20, random_state=42))]),
    'hgb2': Pipeline([('pre', preprocessor), ('m', HistGradientBoostingRegressor(max_iter=1500, max_depth=3, learning_rate=0.02, min_samples_leaf=20, random_state=67))]),
    'hgb3': Pipeline([('pre', preprocessor), ('m', HistGradientBoostingRegressor(max_iter=800, max_depth=4, learning_rate=0.03, min_samples_leaf=25, random_state=7))]),
    'rf1':  Pipeline([('pre', preprocessor), ('m', RandomForestRegressor(n_estimators=300, max_features=0.5, min_samples_leaf=1, random_state=42, n_jobs=-1))]),
    'rf2':  Pipeline([('pre', preprocessor), ('m', RandomForestRegressor(n_estimators=300, max_features=0.4, min_samples_leaf=1, random_state=3, n_jobs=-1))]),
}

preds = {}
for k, pipe in ensemble.items():
    pipe.fit(X, y)
    preds[k] = pipe.predict(test)
    print(f'{k} done')

yhat = np.mean(list(preds.values()), axis=0)
pd.DataFrame({'yhat': yhat}).to_csv('outputs/CW1_submission_k21190308.csv', index=False)
print(f'saved {len(yhat)} predictions')