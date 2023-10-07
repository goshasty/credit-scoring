import pickle

from sklearn import datasets
from sklearn.linear_model import LogisticRegression


if __name__ == "__main__":
    iris = datasets.load_iris()
    X = iris.data[:, :2]
    Y = iris.target
    logreg = LogisticRegression(C=1e5)
    logreg.fit(X, Y)
    pickle.dump(logreg, open("model.sav", "wb"))
    logreg = pickle.load(open("model.sav", "rb"))

    print("Score {0:.2f}".format(logreg.score(X, Y)))
