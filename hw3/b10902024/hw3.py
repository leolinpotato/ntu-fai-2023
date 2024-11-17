from collections import Counter

import numpy as np
import pandas as pd
import ipdb

# set random seed
np.random.seed(0)

"""
Tips for debugging:
- Use `print` to check the shape of your data. Shape mismatch is a common error.
- Use `ipdb` to debug your code
    - `ipdb.set_trace()` to set breakpoints and check the values of your variables in interactive mode
    - `python -m ipdb -c continue hw3.py` to run the entire script in debug mode. Once the script is paused, you can use `n` to step through the code line by line.
"""


# 1. Load datasets
def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    DO NOT MODIFY THIS FUNCTION.
    """
    # Load iris dataset
    iris = pd.read_csv(
        "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data",
        header=None,
    )
    iris.columns = [
        "sepal_length",
        "sepal_width",
        "petal_length",
        "petal_width",
        "class",
    ]

    # Load Boston housing dataset
    boston = pd.read_csv(
        "https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv"
    )

    return iris, boston


# 2. Preprocessing functions
def train_test_split(
    df: pd.DataFrame, target: str, test_size: float = 0.3
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    # Shuffle and split dataset into train and test sets
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    split_idx = int(len(df) * (1 - test_size))
    train = df.iloc[:split_idx]
    test = df.iloc[split_idx:]

    # Split target and features
    X_train = train.drop(target, axis=1).values
    y_train = train[target].values
    X_test = test.drop(target, axis=1).values
    y_test = test[target].values

    return X_train, X_test, y_train, y_test


def normalize(X: np.ndarray) -> np.ndarray:
    # Normalize features to [0, 1]
    # You can try other normalization methods, e.g., z-score, etc.
    # TODO: 1%
    
    # Min-Max normalization
    Min = np.min(X, axis=0)
    Max = np.max(X, axis=0)
    return np.array([(x-Min)/(Max-Min) for x in X])

def standardize(X: np.ndarray) -> np.ndarray:
    Mean = np.mean(X, axis=0)
    Std = np.std(X, axis=0)
    ipdb.set_trace()
    return np.array([(x-Mean)/Std for x in X])

def encode_labels(y: np.ndarray) -> np.ndarray:
    """
    Encode labels to integers.
    """
    # TODO: 1%
    labels = ["Iris-setosa", "Iris-versicolor", "Iris-virginica"]
    return np.array([labels.index(label) for label in y])


# 3. Models
class LinearModel:
    def __init__(
        self, learning_rate=0.01, iterations=1000, model_type="linear"
    ) -> None:
        self.learning_rate = learning_rate
        self.iterations = iterations
        # You can try different learning rate and iterations
        self.model_type = model_type
        self.weight = None

        assert model_type in [
            "linear",
            "logistic",
        ], "model_type must be either 'linear' or 'logistic'"

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        X = np.insert(X, 0, 1, axis=1)
        n_classes = len(np.unique(y))
        n_features = X.shape[1]

        if self.model_type == "logistic":
            self.weight = np.zeros((n_features, n_classes))
            for i in range(self.iterations):
                self.weight += self.learning_rate * self._compute_gradients(X, y)
        else:
            self.weight = np.zeros(n_features)
            for i in range(self.iterations):
                self.weight += self.learning_rate * self._compute_gradients(X, y)
        # TODO: 2%

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.insert(X, 0, 1, axis=1)
        if self.model_type == "linear":
            # TODO: 2%
            return X @ self.weight
        elif self.model_type == "logistic":
            # TODO: 2%
            return np.argmax(self._softmax(X @ self.weight), axis=1)

    def _compute_gradients(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        n_classes = len(np.unique(y))
        n_samples, n_features = X.shape

        if self.model_type == "linear":
            # TODO: 3%
            return -X.T@(X@self.weight-y) / n_samples
        elif self.model_type == "logistic":
            # TODO: 3%
            Y = np.zeros((n_samples, n_classes))
            for i in range(n_samples):
                Y[i, y[i]] = 1

            return (X.T@(Y-self._softmax(X@self.weight))) / n_samples
        
    def _softmax(self, z: np.ndarray) -> np.ndarray:
        exp = np.exp(z)
        return exp / np.sum(exp, axis=1, keepdims=True)


class DecisionTree:
    def __init__(self, max_depth: int = 5, model_type: str = "classifier"):
        self.max_depth = max_depth
        self.model_type = model_type

        assert model_type in [
            "classifier",
            "regressor",
        ], "model_type must be either 'classifier' or 'regressor'"

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self.tree = self._build_tree(X, y, 0)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.array([self._traverse_tree(x, self.tree) for x in X])

    def _build_tree(self, X: np.ndarray, y: np.ndarray, depth: int) -> dict:
        if depth >= self.max_depth or self._is_pure(y):
            return self._create_leaf(y)

        feature, threshold = self._find_best_split(X, y)
        # TODO: 4%
        mask = X[:, feature] <= threshold
        left_y, right_y = y[mask], y[~mask]
        left_X, right_X = X[mask], X[~mask]
        
        left_child = self._build_tree(left_X, left_y, depth + 1)
        right_child = self._build_tree(right_X, right_y, depth + 1)
        
        return {
            "feature": feature,
            "threshold": threshold,
            "left": left_child,
            "right": right_child,
        }

    def _is_pure(self, y: np.ndarray) -> bool:
        return len(set(y)) == 1

    def _create_leaf(self, y: np.ndarray):
        if self.model_type == "classifier":
            # TODO: 1%
            unique_elements, counts = np.unique(y, return_counts=True)
            max_count_index = np.argmax(counts)
            majority = unique_elements[max_count_index]
            return majority
        else:
            # TODO: 1%
            return np.mean(y)

    def _find_best_split(self, X: np.ndarray, y: np.ndarray) -> tuple[int, float]:
        best_gini = float("inf")
        best_mse = float("inf")
        best_feature = None
        best_threshold = None

        for feature in range(X.shape[1]):
            sorted_indices = np.argsort(X[:, feature])
            for i in range(1, len(X)):
                if X[sorted_indices[i - 1], feature] != X[sorted_indices[i], feature]:
                    threshold = (
                        X[sorted_indices[i - 1], feature]
                        + X[sorted_indices[i], feature]
                    ) / 2
                    mask = X[:, feature] <= threshold
                    left_y, right_y = y[mask], y[~mask]

                    if self.model_type == "classifier":
                        gini = self._gini_index(left_y, right_y)
                        if gini < best_gini:
                            best_gini = gini
                            best_feature = feature
                            best_threshold = threshold
                    else:
                        mse = self._mse(left_y, right_y)
                        if mse < best_mse:
                            best_mse = mse
                            best_feature = feature
                            best_threshold = threshold

        return best_feature, best_threshold

    def _gini_index(self, left_y: np.ndarray, right_y: np.ndarray) -> float:
        # TODO: 4%
        def cal_gini(y: np.ndarray):
            total = len(y)
            gini = 1
            for _class in np.unique(y):
                gini -= ((y == _class).sum() / total)**2
            return gini
        return (cal_gini(left_y)*len(left_y)+cal_gini(right_y)*len(right_y)) / (len(left_y)+len(right_y))

    def _mse(self, left_y: np.ndarray, right_y: np.ndarray) -> float:
        # TODO: 4%
        def mse(y: np.ndarray):
            return ((y - np.mean(y))**2).sum()
        return (mse(left_y)*len(left_y)+mse(right_y)*len(right_y)) / (len(left_y)+len(right_y))

    def _traverse_tree(self, x: np.ndarray, node: dict):
        if isinstance(node, dict):
            feature, threshold = node["feature"], node["threshold"]
            if x[feature] <= threshold:
                return self._traverse_tree(x, node["left"])
            else:
                return self._traverse_tree(x, node["right"])
        else:
            return node


class RandomForest:
    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 5,
        model_type: str = "classifier",
    ):
        # TODO: 1%
        self.trees = [DecisionTree(max_depth=max_depth, model_type=model_type) for i in range(n_estimators)]
        self.model_type = model_type
        self.n_estimators = n_estimators

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        n_samples = X.shape[0]
        for tree in self.trees:
            # TODO: 2%
            bootstrap_indices = np.random.choice(n_samples, int(n_samples*0.5), replace=False)
            sample_X = np.array([X[i] for i in bootstrap_indices])
            sample_y = np.array([y[i] for i in bootstrap_indices])
            tree.fit(sample_X, sample_y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        # TODO: 2%
        if self.model_type == "classifier":
            votes = []
            for tree in self.trees:
                votes.append(tree.predict(X))
            votes = np.array(votes).T
            prediction = []
            for v in votes:
                unique_elements, counts = np.unique(v, return_counts=True)
                max_count_index = np.argmax(counts)
                majority = unique_elements[max_count_index]
                prediction.append(majority)
            return np.array(prediction)
        else:
            answer = np.zeros(X.shape[0])
            for tree in self.trees:
                answer += tree.predict(X)
            return answer / self.n_estimators


# 4. Evaluation metrics
def accuracy(y_true, y_pred):
    # TODO: 1%
    return len(np.where(y_true == y_pred)[0]) / len(y_true)


def mean_squared_error(y_true, y_pred):
    # TODO: 1%
    return ((y_true - y_pred)**2).sum() / len(y_true)


# 5. Main function
def main():
    iris, boston = load_data()

    # Iris dataset - Classification
    X_train, X_test, y_train, y_test = train_test_split(iris, "class")
    X_train, X_test = normalize(X_train), normalize(X_test)
    y_train, y_test = encode_labels(y_train), encode_labels(y_test)

    logistic_regression = LinearModel(model_type="logistic")
    logistic_regression.fit(X_train, y_train)
    y_pred = logistic_regression.predict(X_test)
    print("Logistic Regression Accuracy:", accuracy(y_test, y_pred))

    decision_tree_classifier = DecisionTree(model_type="classifier")
    decision_tree_classifier.fit(X_train, y_train)
    y_pred = decision_tree_classifier.predict(X_test)
    print("Decision Tree Classifier Accuracy:", accuracy(y_test, y_pred))

    random_forest_classifier = RandomForest(model_type="classifier")
    random_forest_classifier.fit(X_train, y_train)
    y_pred = random_forest_classifier.predict(X_test)
    print("Random Forest Classifier Accuracy:", accuracy(y_test, y_pred))

    # Boston dataset - Regression
    X_train, X_test, y_train, y_test = train_test_split(boston, "medv")
    X_train, X_test = normalize(X_train), normalize(X_test)

    linear_regression = LinearModel(model_type="linear")
    linear_regression.fit(X_train, y_train)
    y_pred = linear_regression.predict(X_test)
    print("Linear Regression MSE:", mean_squared_error(y_test, y_pred))

    decision_tree_regressor = DecisionTree(model_type="regressor")
    decision_tree_regressor.fit(X_train, y_train)
    y_pred = decision_tree_regressor.predict(X_test)
    print("Decision Tree Regressor MSE:", mean_squared_error(y_test, y_pred))

    random_forest_regressor = RandomForest(model_type="regressor")
    random_forest_regressor.fit(X_train, y_train)
    y_pred = random_forest_regressor.predict(X_test)
    print("Random Forest Regressor MSE:", mean_squared_error(y_test, y_pred))


if __name__ == "__main__":
    main()
