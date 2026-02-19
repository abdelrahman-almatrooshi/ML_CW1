import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Set seed
np.random.seed(123)

# Import training data
trn = pd.read_csv('CW1_train.csv')
X_tst = pd.read_csv('CW1_test.csv') # This does not include true outcomes (obviously)

# Identify categorical columns
categorical_cols = ['cut', 'color', 'clarity']

# One-hot encode categorical variables (fit on train only to avoid data leak)
trn = pd.get_dummies(trn, columns=categorical_cols, drop_first=True)
X_trn = trn.drop(columns=['outcome'])
y_trn = trn['outcome']
train_cols = list(X_trn.columns)

# Encode test using same columns as train (missing cols -> 0, extra cols dropped)
X_tst = pd.get_dummies(X_tst, columns=categorical_cols, drop_first=True)
X_tst = X_tst.reindex(columns=train_cols, fill_value=0)

# Train your model (using a simple LM here as an example)
model = LinearRegression()
model.fit(X_trn, y_trn)

# Test set predictions
yhat_lm = model.predict(X_tst)

# Format submission:
# This is a single-column CSV with nothing but your predictions
out = pd.DataFrame({'yhat': yhat_lm})
out.to_csv('CW1_submission_KNUMBER.csv', index=False) # Please use your k-number here

################################################################################

# At test time, we will use the true outcomes
tst = pd.read_csv('CW1_test_with_true_outcome.csv') # You do not have access to this

# This is the R^2 function
def r2_fn(yhat):
    eps = y_tst - yhat
    rss = np.sum(eps ** 2)
    tss = np.sum((y_tst - y_tst.mean()) ** 2)
    r2 = 1 - (rss / tss)
    return r2

# How does the linear model do?
print(r2_fn(yhat_lm))




