import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import statsmodels.tools.eval_measures as ste

# Set seed for reproducible results
np.random.seed(414)

# Gen toy data
X = np.linspace(0, 15, 1000)
y = 3 * np.sin(X) + np.random.normal(1 + X, .2, 1000)

train_X, train_y = X[:700], y[:700]
test_X, test_y = X[300:], y[300:]

train_df = pd.DataFrame({'X': train_X, 'y': train_y})
test_df = pd.DataFrame({'X': test_X, 'y': test_y})

# Linear Fit
poly_1 = smf.ols(formula='y ~ 1 + X', data=train_df).fit()
print(ste.mse(poly_1.predict(test_df), test_y, axis=0))

# Quadratic Fit
poly_2 = smf.ols(formula='y ~ 1 + X + I(X**2)', data=train_df).fit()
print(ste.mse(poly_2.predict(test_df), test_y, axis=0))

