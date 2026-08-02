"""
model_training.py
------------------
Model definitions, hyperparameter search spaces, and training utilities.
"""

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.pipeline import Pipeline

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False


def get_candidate_models():
    """Return a dict of candidate estimators to compare (baseline hyperparameters)."""
    models = {
        "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42),
        "DecisionTree": DecisionTreeClassifier(random_state=42),
        "RandomForest": RandomForestClassifier(random_state=42, n_jobs=-1),
        "GradientBoosting": GradientBoostingClassifier(random_state=42),
    }
    if XGBOOST_AVAILABLE:
        models["XGBoost"] = XGBClassifier(
            random_state=42, eval_metric="logloss", n_jobs=-1
        )
    return models


def get_param_grids():
    """Hyperparameter search spaces for the tunable candidate models."""
    grids = {
        "LogisticRegression": {
            "model__C": [0.01, 0.1, 1, 10],
            "model__penalty": ["l2"],
            "model__solver": ["lbfgs"],
        },
        "DecisionTree": {
            "model__max_depth": [3, 5, 7, 10, None],
            "model__min_samples_leaf": [1, 5, 10, 20],
        },
        "RandomForest": {
            "model__n_estimators": [200, 400],
            "model__max_depth": [5, 10, 15, None],
            "model__min_samples_leaf": [1, 5, 10],
            "model__max_features": ["sqrt", "log2"],
        },
        "GradientBoosting": {
            "model__n_estimators": [100, 200],
            "model__learning_rate": [0.03, 0.1, 0.2],
            "model__max_depth": [2, 3, 4],
        },
    }
    if XGBOOST_AVAILABLE:
        grids["XGBoost"] = {
            "model__n_estimators": [200, 400],
            "model__max_depth": [3, 5, 7],
            "model__learning_rate": [0.03, 0.1, 0.2],
            "model__subsample": [0.8, 1.0],
            "model__colsample_bytree": [0.8, 1.0],
        }
    return grids


def build_pipeline(preprocessor, model):
    """Wrap a preprocessor + estimator into a single sklearn Pipeline."""
    return Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])


def tune_model(pipeline, param_grid, X_train, y_train, cv=5, search_type="grid",
                n_iter=20, scoring="roc_auc", random_state=42):
    """
    Run hyperparameter tuning with cross-validation.
    Uses GridSearchCV for small grids, RandomizedSearchCV for large ones.
    """
    if search_type == "grid":
        search = GridSearchCV(
            pipeline, param_grid, cv=cv, scoring=scoring, n_jobs=-1, verbose=0
        )
    else:
        search = RandomizedSearchCV(
            pipeline, param_grid, cv=cv, scoring=scoring, n_jobs=-1,
            n_iter=n_iter, random_state=random_state, verbose=0,
        )
    search.fit(X_train, y_train)
    return search
