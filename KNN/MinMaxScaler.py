from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

# load dataset
housing = fetch_california_housing(as_frame=True)

X = housing.data
y = housing.target

# split train and test data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, random_state=0
)

# Scaled kNN Regressor (Euclidean)
knn_euclidean = Pipeline(steps=[
    ("scaler", MinMaxScaler()),
    ("knn", KNeighborsRegressor(n_neighbors=10, metric="minkowski", p=2))  # Euclidean
])

knn_euclidean.fit(X_train, y_train)
y_pred_euclidean = knn_euclidean.predict(X_test)
mse_euclidean = mean_squared_error(y_test, y_pred_euclidean)

print(f"Mean Squared Error (k=10, Euclidean):  {mse_euclidean:.4f}")

# Manhattan
knn_manhattan = Pipeline(steps=[
    ("scaler", MinMaxScaler()),
    ("knn", KNeighborsRegressor(n_neighbors=10, metric="minkowski", p=1))  # Manhattan
])

knn_manhattan.fit(X_train, y_train)
y_pred_manhattan = knn_manhattan.predict(X_test)
mse_manhattan = mean_squared_error(y_test, y_pred_manhattan)

print(f"Mean Squared Error (k=10, Manhattan): {mse_manhattan:.4f}")
