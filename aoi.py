from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# 1. データの準備
iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.3)

# 2. モデルの選択と学習
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

# 3. 評価
y_pred = knn.predict(X_test)
print(f"AIの正解率: {accuracy_score(y_test, y_pred):.2f}")

print(123)