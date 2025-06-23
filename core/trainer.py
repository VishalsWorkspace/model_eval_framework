# trainer.py
from sklearn.base import BaseEstimator

def train_model(model: BaseEstimator, X_train, y_train):
    model.fit(X_train, y_train)
    return model
