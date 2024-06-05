import joblib

from sklearn.datasets import fetch_openml

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import GridSearchCV

# from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import precision_score, recall_score

mnist = fetch_openml('mnist_784', as_frame=False)

X, y = mnist.data, mnist.target
X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]

kn_clf = KNeighborsClassifier()

param_grid = [{'weights': ('uniform', 'distance'), 'n_neighbors': [3, 4, 5, 6]}]

grid_search = GridSearchCV(
    estimator=kn_clf, param_grid=param_grid, cv=3, scoring='accuracy')
grid_search.fit(X_train, y_train)

final_model = grid_search.best_estimator_

joblib.dump(final_model, "model.pkl")
