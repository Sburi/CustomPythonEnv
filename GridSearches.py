import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import GridSearchCV


class GridSearchClassifiers(Columns):
    
    def __init__(self, X_train, X_test, y_train, y_test):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        Columns.__init__(self)

    def run_tfidf_vectorizer(self):
        tfidf = TfidfVectorizer()
        self.X_train_tfidf = tfidf.fit_transform(self.X_train)
        self.X_test_tfidf = tfidf.transform(self.X_test)

    @ignore_warnings(category=ConvergenceWarning)
    def run_grid_search_on_logreg(self):
        params = {
            'max_iter': [50, 100, 500, 1000, 5000],
            'C': np.logspace(-5, 8, 15, 30),
            'solver': ['lbfgs', 'newton-cg', 'liblinear', 'sag'],
            'penalty': ['l2', 'l1', 'elasticnet']}
        model = LogisticRegression()
        model_cv = GridSearchCV(model, params, cv=5, verbose=0)
        model_cv.fit(self.X_train_tfidf, self.y_train)

        best_score = model_cv.best_score_
        best_params = model_cv.best_params_

        print('LR best score: ', best_score)
        print('LR best params: ', best_params)

    @ignore_warnings(category=ConvergenceWarning)
    def run_grid_search_on_knn(self):
        params = {
            'n_neighbors': [2, 5, 7, 10],
            'weights': ['uniform', 'distance'],
            'algorithm': ['ball_tree', 'kd_tree', 'brute'],
            'leaf_size': [2, 10, 20, 30, 50, 100]}
        model = KNeighborsClassifier()
        model_cv = GridSearchCV(model, params, cv=5, verbose=0)
        model_cv.fit(self.X_train_tfidf, self.y_train)

        best_score = model_cv.best_score_
        best_params = model_cv.best_params_

        print('KNN best score: ', best_score)
        print('KNN best params: ', best_params)

    @ignore_warnings(category=ConvergenceWarning)
    def run_grid_search_on_randomforest(self):
        params = {
            'n_estimators': [10, 50, 100, 250, 500],
            'criterion': ['gini', 'entropy', 'log_loss'],
            'max_depth': [None, 2, 10, 20, 100],
            'min_samples_split': [2, 10, 20],
            'min_samples_leaf': [1, 5, 10, 20],
            'max_features': ['sqrt', 'log2'],
            'max_leaf_nodes': [None, 2, 10, 20, 100],
            'min_impurity_decrease': [0.0, 0.001, 1, 0.0001],
            }
        model = RandomForestClassifier()
        model_cv = GridSearchCV(model, params, cv=5, verbose=0)
        model_cv.fit(self.X_train_tfidf, self.y_train)

        best_score = model_cv.best_score_
        best_params = model_cv.best_params_

        print('RF best score: ', best_score)
        print('RF best params: ', best_params)


#predict = GridSearchClassifiers(X_train, X_test, y_train, y_test)
#predict.run_tfidf_vectorizer()
#predict.run_grid_search_on_logreg()
#predict.run_grid_search_on_knn()
#predict.run_grid_search_on_randomforest()
