# Compare-Machine-Learning-Algorithms

## Introduction

Breast cancer (BC) is one of the most common cancers among women worldwide, representing the majority of new cancer cases and cancer-related deaths according to global statistics, making it a significant public health problem in today’s society. The data set could be found from this [Link](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic))

Classification and data mining methods are an effective way to classify data. Especially in medical field, where those methods are widely used in diagnosis and analysis to make decisions.

In this repo it will discover how you can create a test harness to compare multiple different machine learning algorithms in Python with scikit-learn.

## Choose and Compare the Best Machine Learning Model

How do you choose the best model for your problem?. When you work on a machine learning project, you often end up with multiple good models to choose from. Each model will have different performance and characteristics.

## Models

In order to compare ML algorithms. I choose a number of algorithms to compare in the same data set. I choose two version of support vector machine with different parameters. 

* LogisticRegression
* DecisionTreeClassifier
* KNeighborsClassifier
* LinearDiscriminantAnalysis
* GaussianNB
* SVM
* RandomForestClassifier

```python
# prepare models
models = [('LReg', LogisticRegression()), ('LDA', LinearDiscriminantAnalysis()),
          ('KNN', KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2)),
          ('DTClass', DecisionTreeClassifier(criterion='entropy', random_state=0)), ('GAU', GaussianNB()),
          ('SVM1', SVC(kernel='rbf', random_state=0)), ('SVM2', SVC(kernel='linear', random_state=0)),
          ('RFC', RandomForestClassifier(n_estimators=5, criterion='entropy', random_state=0))]
```


### Scoring

After setting the algorithms classifiers. I evaluate each algorithm and evaluate also by showing the accuracy's percentage.  


```python
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

```

```code
 LReg: 0.947368
 LDA: 0.964912
 KNN: 0.964912
 DTClass: 0.964912
 GAU: 0.894737
 SVM1: 0.947368
 SVM2: 0.947368
 RFC: 0.947368        
```

### Ploting

```python
# Plotting the results
  names = names
  values = results
  fig, axs = plt.subplots()
  axs.bar(names, values)
  fig.suptitle('Categorical Plotting')
  plt.show()
```

<p align="center"> 
<img src="https://github.com/BardisRenos/Compare-Machine-Learning-Algorithms/blob/master/myplot.png" width="450" height="350" style=centerme>
</p>
