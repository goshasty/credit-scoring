import pickle

import pandas as pd
from sklearn import datasets

if __name__ == "__main__":
    iris = datasets.load_iris()
    X = iris.data[1::2, :2]
    Y = iris.target[1::2]
    logreg = pickle.load(open("model.sav", "rb"))
    res = logreg.predict(X)

    res = pd.DataFrame(columns=["preds"], data=res)
    res.to_csv("answers.csv")

    print("Score {0:.2f}".format(logreg.score(X, Y)))
