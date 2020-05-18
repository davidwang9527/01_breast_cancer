import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

#step1:Loading data
X,y=load_breast_cancer(return_X_y=True)

#step2:Split data
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2,random_state=40,stratify=y)

#step3:Training
clf=RandomForestClassifier(random_state=40,class_weight='balanced')
param_grid={
            'n_estimators':[x for x in np.arange(50,301,50)],
            'max_features':['sqrt','log2'],
            'max_samples':[0.1,0.3,0.8]
           }
search = GridSearchCV(estimator=clf,param_grid=param_grid,scoring='accuracy',cv=5,refit=True,verbose=1,n_jobs=-1)
search.fit(X_train,y_train)
score=search.score(X_test,y_test)
score_train=search.score(X_train,y_train)
score_test=search.score(X_test,y_test)
print('best hyperparameters for RandomForestClassifier:{}'.format(search.best_params_))
print('scores for RandomForestClassifier(score on training set/testing set):{:.2f}/{:.2f}'.format(score_train,score_test))
