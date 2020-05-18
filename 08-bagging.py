import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import RidgeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.model_selection import GridSearchCV

#step1:Loading data
X,y=load_breast_cancer(return_X_y=True)

#step2:Split data
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2,random_state=40,stratify=y)

#step3:Training
clf=BaggingClassifier(random_state=40)
param_grid={
    'base_estimator':[KNeighborsClassifier(n_neighbors=7,weights='uniform',leaf_size=1, metric='manhattan'),
                      RidgeClassifier(random_state=40,class_weight='balanced',alpha=0.00001)
                    ],
    'n_estimators':[x for x in np.arange(10,101,30)],
    'max_samples' :[0.3,0.7,1.0],
    'max_features' :[0.3,0.7,1.0],
    'bootstrap_features':[True,False]
    },
search = GridSearchCV(estimator=clf,param_grid=param_grid,scoring='accuracy',cv=5,refit=True,verbose=1,n_jobs=-1)
search.fit(X_train,y_train)
score=search.score(X_test,y_test)
score_train=search.score(X_train,y_train)
score_test=search.score(X_test,y_test)
print('best hyperparameters for BaggingClassifier:{}'.format(search.best_params_))
print('scores for BaggingClassifier(score on training set/testing set):{:.2f}/{:.2f}'.format(score_train,score_test))
