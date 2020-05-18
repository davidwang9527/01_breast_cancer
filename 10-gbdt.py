import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV

#step1:Loading data
X,y=load_breast_cancer(return_X_y=True)

#step2:Split data
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2,random_state=40,stratify=y)

#step3:Training
clf=GradientBoostingClassifier(random_state=40)
param_grid={
            'learning_rate':[0.001,0.003,0.01,0.03,0.1,0.3,0.6,0.9,1],
            'n_estimators':[x for x in np.arange(10,221,30)],
            'max_depth':[1,3,10,30,100]
           }
search = GridSearchCV(estimator=clf,param_grid=param_grid,scoring='accuracy',cv=5,refit=True,verbose=1,n_jobs=-1)
search.fit(X_train,y_train)
score_train=search.score(X_train,y_train)
score_test=search.score(X_test,y_test)
print('best hyperparameters for GradientBoostingClassifier:{}'.format(search.best_params_))
print('scores for GradientBoostingClassifier(score on training set/testing set):{:.2f}/{:.2f}'.format(score_train,score_test))
