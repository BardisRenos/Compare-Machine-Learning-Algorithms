from sklearn.datasets import load_breast_cancer
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Import some classification models
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier


# Loading the data set
data = load_breast_cancer()

# Splitting the data into images and labels
X, y = load_breast_cancer(return_X_y=True)

# The data is spited into 75% training and 25% testing data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

# prepare models
models = [('LReg', LogisticRegression()), ('LDA', LinearDiscriminantAnalysis()),
          ('KNN', KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2)),
          ('DTClass', DecisionTreeClassifier(criterion='entropy', random_state=0)), ('GAU', GaussianNB()),
          ('SVM1', SVC(kernel='rbf', random_state=0)), ('SVM2', SVC(kernel='linear', random_state=0)),
          ('RFC', RandomForestClassifier(n_estimators=5, criterion='entropy', random_state=0))]

# evaluate each model in turn
results = []
names = []
for name, classifier in models:
    classifier.fit(X_train, y_train)
    predicted_value = classifier.predict(X_test)
    acc_score = accuracy_score(y_test, predicted_value)
    results.append(acc_score)
    names.append(name)
    print("%s: %f" % (name, acc_score))

# Plotting the results
names = names
values = results
fig, axs = plt.subplots()
axs.bar(names, values)
fig.suptitle('Categorical Plotting')
plt.show()
