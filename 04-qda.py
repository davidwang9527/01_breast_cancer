import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

#step1:Loading data
X,y=load_breast_cancer(return_X_y=True)

#step2:Split data
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2,random_state=40,stratify=y)

#step3:Feature Engineering
pca=PCA()
standardScaler=StandardScaler()

#step4:Training
prior=np.mean(y)
clf=QuadraticDiscriminantAnalysis(priors=[1-prior,prior])
pipe=Pipeline(steps=[('pca',pca),('standardScaler',standardScaler),('clf',clf)])
param_grid={'pca__n_components':[x for x in [24,25,26,27,28,29,30]],'clf__reg_param':[0,0.01,0.03,0.1,0.3,0.6,0.9,1]},
search = GridSearchCV(pipe,param_grid,scoring='accuracy',cv=5,refit=True,n_jobs=-1)
search.fit(X_train,y_train)
print('best hyperparameters:{}'.format(search.best_params_))
score_train=search.score(X_train,y_train)
score_test=search.score(X_test,y_test)
print('score on training set/testing set:{:.2f}/{:.2f}'.format(score_train,score_test))
