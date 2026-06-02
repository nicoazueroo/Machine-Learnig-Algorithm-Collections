from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error

# load dataset
housing = fetch_california_housing(as_frame=True)


X = housing.data


y = housing.target

# split train and test data
X_train, X_test, y_train, y_test = train_test_split( X, y, random_state=0)

print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)

# KNN Regressor (k = 10)
regressor = KNeighborsRegressor(n_neighbors=10)

# train, predict and evalutae
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)
mse = mean_squared_error(y_test, y_pred)

print(f"\nMean Squared Error (k=10): {mse:.4f}")
