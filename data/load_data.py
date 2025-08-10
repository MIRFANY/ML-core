import pandas as pd 
from sklearn.datasets import load_iris

def get_iris_data():
    iris = load_iris():
    X = pd.DataFrame(iris.data, columns=iris.feature_names)
    y = pd.Series(iris.target, name="target")
    return X, y
