# %%
# address the issue of imbalanced classes
from sklearn.metrics import log_loss
from sklearn.utils import class_weight
import numpy as np
# added import for XGB
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RandomizedSearchCV
from xgboost import XGBClassifier

def balanced_log_loss(y_true, y_pred):
    class_weights = class_weight.compute_class_weight('balanced', classes = np.unique(y_true), y = y_true)
    weights = class_weights[y_true.astype(int)]
    loss = log_loss(y_true, y_pred, sample_weight = weights)
    return loss

# %%
def adjusted_prob(y_hat):
    i = 0
    for y_hat_classA, y_hat_classB in y_hat:

        y_hat_classA = np.max([np.min([y_hat_classA, 1 - 10 ** -15]), 10 ** -15])
        y_hat_classB = np.max([np.min([y_hat_classB, 1 - 10 ** -15]), 10 ** -15])

        y_hat[i, 0] = y_hat_classA
        y_hat[i, 1] = y_hat_classB
        i += 1
    
    return y_hat

# %%
# # # # # # # # # # # # # # # # # # # #
# # # # K-Fold Cross Validation # # # #
# # # # # # # # # # # # # # # # # # # #
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

n_splits = 10
cv_score_LR = 0
train_score_LR = 0

skf = StratifiedKFold(n_splits = n_splits, shuffle = True, random_state = 42)

for i, (train_index, test_index) in enumerate(skf.split(X, y)):
    
    X_train, X_test = X.iloc[train_index, :], X.iloc[test_index, :]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    log_model = LogisticRegression(C = 0.1, class_weight='balanced')
    log_model.fit(X_train, y_train)

    y_hat_test_LR = log_model.predict_proba(X_test)
    y_hat_test_LR = adjusted_prob(y_hat_test_LR)
    cv_score_LR += balanced_log_loss(y_test, y_hat_test_LR)

    y_hat_train_LR = log_model.predict_proba(X_train)
    y_hat_train_LR = adjusted_prob(y_hat_train_LR)
    train_score_LR += balanced_log_loss(y_train, y_hat_train_LR)


print('Train :', train_score_LR / 10)
print('CV score: ', cv_score_LR / 10)


