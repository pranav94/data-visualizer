"""
Python script to vizualize data
"""
# Load libraries
import pandas
from pandas.plotting import scatter_matrix

# Configure backend to support OSX
import matplotlib as mpl
mpl.use('TkAgg')

import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

import argparse
import sys


parser = argparse.ArgumentParser(description='Helper for data visualization')

parser.add_argument('--shape', '-s', action='store_true',
                    help='returns the dimension of the dataset')
parser.add_argument('--head', '-n', action='store_true',
                    help='returns first 20 rows of the data')
parser.add_argument('--describe', '-d', action='store_true',
                    help='returns count, mean, min, max and some percentiles')
parser.add_argument('--classes', '-c', action='store_true',
                    help='returns number of instances (rows) that belong to each class for iris.data')
parser.add_argument('--box', '-b', action='store_true',
                    help='returns box and whisker plot')
parser.add_argument('--histogram', '-g', action='store_true',
                    help='returns a histogram plot')
parser.add_argument('--scatter', '-t', action='store_true',
                    help='returns a scatter plot')
parser.add_argument('--compare-models', '-m',
                    action='store_true', help='evaluate each model in turn')
parser.add_argument('--predict', '-p', action='store_true',
                    help='Predict results')
parser.add_argument('--url', '-u', type=str, required=False, help='Url or file path to read data from.\
                    Ex: https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data')

args = parser.parse_args()

if len(sys.argv) == 1:
    parser.print_help(sys.stderr)
    sys.exit(1)

# Load dataset directly from UCI ML directory
# url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"

if not args.url or "iris.data" in args.url:
    url = "iris.data"
    names = ['sepal-length', 'sepal-width',
             'petal-length', 'petal-width', 'class']
    dataset = pandas.read_csv(url, names=names)
else:
    dataset = pandas.read_csv(args.url)

if args.shape:
    print(dataset.shape)

if args.head:
    print(dataset.head(20))

if args.describe:
    print(dataset.describe())

if args.classes:
    if url != "iris.data":
        print("Classes are supported for iris.data only")
        exit
    print(dataset.groupby('class').size())

if args.box:
    dataset.plot(kind='box', subplots=True,
                 layout=(2, 2), sharex=False, sharey=True)
    plt.show()

if args.histogram:
    dataset.hist()
    plt.show()

if args.scatter:
    scatter_matrix(dataset)
    plt.show()

# Split-out validation dataset
array = dataset.values
X = array[:, 0:4]
Y = array[:, 4]
validation_size = 0.20
seed = 7
X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(
    X, Y, test_size=validation_size, random_state=seed)

# Test options and evaluation metric
seed = 7
scoring = 'accuracy'

# Spot Check Algorithms
models = []
models.append(('LR', LogisticRegression()))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC()))
# evaluate each model in turn
results = []
names = []

if args.compare_models:
    print("\nAlgorithm:\tAverage\t\t(Standard Deviation)")
    print("----------\t-------\t\t--------------------")
    for name, model in models:
        kfold = model_selection.KFold(n_splits=10, random_state=seed)
        cv_results = model_selection.cross_val_score(
            model, X_train, Y_train, cv=kfold, scoring=scoring)
        results.append(cv_results)
        names.append(name)
        msg = "%s:\t\t%f\t(%f)" % (name, cv_results.mean(), cv_results.std())
        print(msg)

    # Compare Algorithms
    fig = plt.figure()
    fig.suptitle('Algorithm Comparison')
    ax = fig.add_subplot(111)
    plt.boxplot(results)
    ax.set_xticklabels(names)
    plt.show()

if args.predict:
    # Make predictions on validation dataset
    svc = SVC()
    svc.fit(X_train, Y_train)
    predictions = svc.predict(X_validation)
    print("\n\nAccuracy of SVM: {} %\n\n".format(
        accuracy_score(Y_validation, predictions)*100))
    print("Confusing matrix\n-----------------\n {}\n\n".format(
        confusion_matrix(Y_validation, predictions)))
    print("Classification report\n-----------------\n {}\n\n".format(
        classification_report(Y_validation, predictions)))
