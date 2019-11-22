from sklearn.datasets import load_iris
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import LocalOutlierFactor



iris =load_iris()
input_data=iris.data
target_data=iris.target





# Generate train data
X = 0.3 * np.random.randn(100, 2)
# Generate some abnormal novel observations
X_outliers = np.random.uniform(low=-4, high=4, size=(20, 2))
X = np.r_[X + 2, X - 2, X_outliers]







clf = LocalOutlierFactor(n_neighbors=25, contamination=0.1)
y_pred = clf.fit_predict(X)


#n_errors = (y_pred != ground_truth).sum()
X_scores = clf.negative_outlier_factor_


print(y_pred)
print(X_scores)



plt.scatter(X[:,0], X[:,1],c=y_pred, cmap='Paired')
plt.title("custom DBSCAN predicted cluster outputs")

plt.tight_layout()
plt.show()







