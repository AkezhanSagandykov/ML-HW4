import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import subplots, cm 
from sklearn.metrics import accuracy_score
import sklearn.model_selection as skm
from sklearn.svm import SVC 
from ISLP.svm import plot as plot_svm
random_numbers = np.random.default_rng(1)
X = random_numbers.standard_normal((60, 2)) 
y = np.array([-1]*30+[1]*30) 
X[y==1] += 1.4
X[y==-1] -= 0.5
fig, ax = subplots(figsize=(8,8)) 
ax.scatter(X[:,0],
X[:,1],
c=y, cmap=cm.coolwarm)
svm_linear_small = SVC(C=0.1, kernel='linear')
svm_linear_small.fit(X, y)
fig, ax = subplots(figsize=(8,8))
plot_svm(X,
y, svm_linear_small , ax=ax)
kfold = skm.KFold(5, random_state=0,
shuffle=True)
grid = skm.GridSearchCV(svm_linear_small,
{'C':[0.01,0.1,0.2,1,5,10,100]},
refit=True, cv=kfold, scoring='accuracy')
grid.fit(X, y)
print(grid.best_params_)
print(grid.cv_results_[('mean_test_score')])
C_parameters = [0.01, 0.1, 0.2, 1.5, 10, 100]
svc_models = {}
number_of_misclassifications = np.zeros_like(C_parameters)
number_of_misclassifications_dictionary = {}
training_error_rate = np.zeros_like(C_parameters)
training_error_rate_dictionary = {}
for i, C in enumerate(C_parameters):
    svc_linear = SVC(C=C, kernel="linear")
    svc_linear.fit(X, y)
    svc_models[C] = svc_linear
    error_rate = []
    training_error_rate[i] = 1 - accuracy_score(y, svc_linear.predict(X))
    training_error_rate_dictionary[C] = training_error_rate[i] 
    number_of_misclassifications[i] = np.sum(svc_linear.predict(X) != y)
    number_of_misclassifications_dictionary[C] = number_of_misclassifications[i]
for C, i in number_of_misclassifications_dictionary.items():
    print(str(C) + ":" + str(i))
for C, i in training_error_rate_dictionary.items():
    print(str(C) + ":" + str(i))
random_numbers = np.random.default_rng(1)
X_test = random_numbers.standard_normal((30, 2)) 
y_test = np.array([-1]*15+[1]*15) 
X_test[y_test==1] += 1.4
X_test[y_test==-1] -= 0.5
fig, ax = subplots(figsize=(8,8)) 
ax.scatter(X_test[:,0],
X_test[:,1],
c=y_test, cmap=cm.coolwarm)
test_number_of_misclassifications = np.zeros_like(C_parameters)
test_number_of_misclassifications_dictionary = {}
test_error_rate = np.zeros_like(C_parameters)
test_error_rate_dictionary = {}
for i, (C, model) in enumerate(svc_models.items()):
    test_number_of_misclassifications[i] = np.sum(model.predict(X_test) != y_test)
    test_number_of_misclassifications_dictionary[C] = test_number_of_misclassifications[i]
    test_error_rate[i] = 1 - accuracy_score(y_test, model.predict(X_test))
    test_error_rate_dictionary[C] = test_error_rate[i]
for C, i in test_number_of_misclassifications_dictionary.items():
    print(str(C) + ":" + str(i))
for C, i in test_error_rate_dictionary.items():
    print(str(C) + ":" + str(i))
