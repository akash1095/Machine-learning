# First all we import basic libraries such as Pandas, numpy, matplotlib. then we will extend our import as we need.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics

# we are loading our data into our program.

data = pd.read_csv("Dataset.csv")
print(data.head())
print(data.columns)

# data visualization
data.hist(column="income", figsize=(20, 30), bins=50)
data.plot(kind="scatter", x="income", y="age", figsize=(20, 30), c="custcat", cmap=plt.get_cmap('jet'), colorbar=True)
data.plot(kind="scatter", x="income", y="age", xlim=(0, 250), figsize=(20, 30), c="custcat", cmap=plt.get_cmap('jet'), colorbar=True)
print(plt.show())


# feature  X and label y
# change Dataframe into a numpy array.
X = data[['region', 'tenure', 'age', 'marital', 'address', 'income', 'ed',
          'employ', 'retire', 'gender', 'reside']].values
y = data['custcat'].values

# Data Standardization
X = preprocessing.StandardScaler().fit(X).transform(X.astype(float))

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4)

print("the size of train set : ", X_train, y_train)
print("The size of the test set : ", X_test, y_test)

# Algorithm for K neighbour clustering
k = 4
neigh = KNeighborsClassifier(n_neighbors=k).fit(X_train, y_train)
print(neigh)

# prediction
y_hat = neigh.predict(X_test)
print(y_hat[0:10])

# accuracy Evaluation
print("The train set accuracy : ", metrics.accuracy_score(y_train, neigh.predict(X_train)))
print("The test set accuracy : ", metrics.accuracy_score(y_hat, y_test))


# finding best k value for best model.
Ks = 15
mean_acc = np.zeros(Ks-1)
std_acc = np.zeros(Ks-1)
for n in range(1, Ks):
    neigh = KNeighborsClassifier(n_neighbors=n).fit(X_train, y_train)
    y_hat = neigh.predict(X_test)
    mean_acc[n-1] = metrics.accuracy_score(y_test, y_hat)
    std_acc[n-1] = np.std(y_hat == y_test)/np.sqrt(y_hat.shape[0])

print("the Mean accuracy is ", mean_acc)
print("The std accuracy : ", std_acc)

# plot the the accuracy with std to understand pur model in better way.
plt.plot(range(1, Ks), mean_acc, 'g')
plt.fill_between(range(1, Ks), mean_acc - 1 * std_acc, mean_acc + 1 * std_acc, alpha=0.10)
plt.legend(('Accuracy ', '+/- 3xstd'))
plt.ylabel('Accuracy ')
plt.xlabel('Number of Nabors (K)')
plt.tight_layout()
plt.show()

max_accuracy = mean_acc.max()
print("We acquire maximum accurray at k=", mean_acc.argmax(), "the score is : ", max_accuracy)

















