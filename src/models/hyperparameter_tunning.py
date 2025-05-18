from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier

def tunning_hyperparam(X_train_balanced, y_train_balanced):
    #  Define hyperparameter grids for each model
    param_grid_xgb = {
        'n_estimators': [100, 200],
        'max_depth': [3, 6, 10],
        'learning_rate': [0.01, 0.1]
    }

    param_grid_cat = {
        'iterations': [100, 200],
        'depth': [3, 6, 10],
        'learning_rate': [0.01, 0.1]
    }

    param_grid_lgb = {
        'n_estimators': [100, 200],
        'num_leaves': [20, 40, 60],
        'learning_rate': [0.01, 0.1]
    }

    #  Run GridSearchCV separately for each model
    grid_xgb = GridSearchCV(XGBClassifier( eval_metric='mlogloss'), param_grid_xgb, scoring='f1_macro', cv=3)
    grid_xgb.fit(X_train_balanced, y_train_balanced)
    print("\n\nBest XGB Params:", grid_xgb.best_params_)

    grid_cat = GridSearchCV(CatBoostClassifier(verbose=0), param_grid_cat, scoring='f1_macro', cv=3)
    grid_cat.fit(X_train_balanced, y_train_balanced)
    print("\n\nBest CatBoost Params:", grid_cat.best_params_)

    grid_lgb = GridSearchCV(LGBMClassifier(verbose=-1), param_grid_lgb, scoring='f1_macro', cv=3)
    grid_lgb.fit(X_train_balanced, y_train_balanced)
    print("\n\nBest LightGBM Params:", grid_lgb.best_params_)
    tun_hyperparam = ("yes")
    return tun_hyperparam