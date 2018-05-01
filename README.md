data-visualizer
===========

data-visualizer is built using `Python 3.0` and `scikit`

Setup
-----

pipenv is used to manage dependencies. Set-up your dependencies using `pipenv install`

Usage
-----

Run `pipenv shell` to get into the virtual environment.

`pipenv python visualizer.py --help` to returns a list of operations the script can perform.

`python visualizer.py --compare-models` will build the models and returns scoring based on accuracy for all the models. 

The algorithms used to build the models are:
* LogisticRegression
* LinearDiscriminantAnalysis
* KNeighborsClassifier
* DecisionTreeClassifier
* GaussianNB
* SVC

The code uses `iris.data` shipped with the repository as the default data source. You can override this with `--url` option.

To run predictions on validation dataset using SVM, you can run `python visualizer.py --predict`. 

