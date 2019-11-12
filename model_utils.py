import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegressionCV
import getpass


class AllModels:
    def __init__(self, X, y, model_name):
        self.X = X
        self.y = y
        self.model_name = model_name

        if getpass.getuser() == 'Mitch':
            self.n_jobs = 2
        else:
            self.n_jobs = -1

        self.models = []
        self.best_scores = {}

    def fit(self):
        if self.model_name == 'SVM' or self.model_name == 'all':
            parameters = {'kernel': ('linear', 'rbf'), 'C': np.exp(np.arange(-4, 5, 4))}
            svc = SVC(gamma='scale')
            clf = GridSearchCV(svc, parameters, cv=5, n_jobs=self.n_jobs)
            clf.fit(self.X, self.y)
            self.best_scores['SVM'] = clf.best_score_
            self.models.append(clf)
        if self.model_name == 'LogReg' or self.model_name == 'all':
            clf = LogisticRegressionCV(cv=5, n_jobs=self.n_jobs)
            clf.fit(self.X, self.y)
            best_score = max(np.mean(clf.scores_, axis=1))
            self.best_scores['LogReg'] = best_score
            self.models.append(clf)
        else:
            raise Exception('not implemented yet')

if __name__ == '__main__':
    from sklearn.datasets import load_iris
    from sklearn.linear_model import LogisticRegressionCV
    X, y = load_iris(return_X_y=True)
    X = X[:100, :]
    y = y[:100]
    clf = LogisticRegressionCV(cv=5, random_state=0)
    clf.fit(X, y)
    print(clf.scores_)

