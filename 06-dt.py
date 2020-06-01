import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier

#step1:Loading data
X,y=load_breast_cancer(return_X_y=True)

#step2:Split data
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2,random_state=40,stratify=y)

#step3:Training
clf=DecisionTreeClassifier(random_state=40,class_weight='balanced')
param_grid={'criterion'        :['gini','entropy'],
            'splitter'         :['best', 'random'],
            'min_samples_leaf' :[x for x in np.arange(3,50,3)],
            'max_features'     :[x for x in np.arange(1,X.shape[1])],
            }
search = GridSearchCV(clf,param_grid,scoring='accuracy',cv=5,refit=True)
search.fit(X_train,y_train)
score_train=search.score(X_train,y_train)
score_test=search.score(X_test,y_test)
print('best hyperparameters for DecisionTreeClassifier:{}'.format(search.best_params_))
print('scores for DecisionTreeClassifier(score on training set/testing set):{:.2f}/{:.2f}'.format(score_train,score_test))
