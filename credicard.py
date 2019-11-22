import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest

data = pd.read_csv("creditcard.csv")
data = data.drop(['Time'] , axis=1)

outliers = data.loc[data['Class']==1]
normal = data.loc[data['Class']==0]

outliers = outliers.drop(['Class'] , axis=1)
normal = normal.drop(['Class'] , axis=1)

X_train = np.array(normal.iloc[0:142403,:])
X_dev = np.array(normal.iloc[142403:,:])
X_test = np.array(outliers)


clf = IsolationForest(max_samples=100)
clf.fit(X_train)

y_pred_train = clf.predict(X_train)
y_pred_dev = clf.predict(X_dev)
y_pred_test = clf.predict(X_test)

print("Accuracy dev :", list(y_pred_dev).count(1)/y_pred_dev.shape[0])
print("Accuracy test:", list(y_pred_test).count(-1)/y_pred_test.shape[0])














'''


features_train = credit_cards

print(features_train)



scaler.fit(features_train)
normalised_input_data=scaler.transform(features_train)
rng = np.random.RandomState(42)

clf = IsolationForest(contamination=0.1,n_jobs=-1,max_samples=256,random_state=rng)
clf.fit(normalised_input_data)
y_pred = clf.predict(normalised_input_data)



#print("Accuracy in Detecting Legit Cases:", list(inlier_pred_test).count(1)/inlier_pred_test.shape[0])
print("Accuracy in Detecting Fraud Cases:", list(y_pred).count(-1)/y_pred.shape[0])


print(y_pred)
#print(X_scores)

plt.scatter(normalised_input_data[:,0], normalised_input_data[:,1],c=y_pred, cmap='Paired')
plt.title("custom DBSCAN predicted cluster outputs")

plt.tight_layout()
plt.show()


'''








