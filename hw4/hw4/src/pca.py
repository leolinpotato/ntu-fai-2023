import numpy as np


"""
Implementation of Principal Component Analysis.
"""
class PCA:
    def __init__(self, n_components: int) -> None:
        self.n_components = n_components
        self.mean = None
        self.components = None

    def fit(self, X: np.ndarray) -> None:
        #TODO: 10%
        self.mean = np.mean(X, axis=0)

        # Compute eigenvalues and eigenvectors
        X_shifted = X - self.mean
        eigenvalues, eigenvectors = np.linalg.eigh(X_shifted.T @ X_shifted)

        # Sort eigenvalues and corresponding eigenvectors in descending order
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        # Select the top n eigenvectors
        self.components = eigenvectors[:, :self.n_components]

    def transform(self, X: np.ndarray) -> np.ndarray:
        #TODO: 2%
        return (X - self.mean) @ self.components

    def reconstruct(self, X):
        #TODO: 2%
        return self.transform(X) @ self.components.T + self.mean
