import numpy as np
from collections import Counter

class KNN: 
    def __init__(self, k=3):
        self.K=k

    def train(self, X,y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        y_prediksi = [self._prediksi(x) for x in X]
        return np.array(y_prediksi)

    def _prediksi(self, x):
        jarak_titik = [self.jarak(x,x_train) for x_train in self.X_train]
        k_terbaik = np.argsort(jarak_titik)[:self.K]
        label_k_terbaik = [self.y_train[i] for i in k_terbaik]
        hasil_voting = Counter(label_k_terbaik).most_common(1)
        return hasil_voting[0][0]

    def jarak(self, x1, x2):
        return np.sqrt(np.sum((x1-x2)**2))



