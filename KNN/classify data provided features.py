from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Load iris dataset
iris = load_iris(as_frame=True)

# Use ALL 4 features
X = iris.data  # 150 x 4
y = iris.target

# split train and test data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, random_state=0
)

print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)

# Show accuracies
for k in range(1, 21):
    clf = KNeighborsClassifier(n_neighbors=k)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred) * 100
    print(f"kNN accuracy (k={k:2d}) = {acc:.2f}%")
