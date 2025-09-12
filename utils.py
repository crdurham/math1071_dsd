import numpy as np
import pandas as pd
import statsmodels.api as sm # type: ignore


def lin_reg(X, y):
    '''Fit a weighted logistic regression model with feature data X and label data y. Returns the results of
    fitting the model.'''
    if not isinstance(y, pd.Series):
        y = pd.Series(y, index=X.index)
    else:
        y = y.reindex(X.index)

    

    model = sm.OLS(
        y,
        sm.add_constant(X),
        family=sm.families.Binomial(),
    )

    results = model.fit()
    return model, results

