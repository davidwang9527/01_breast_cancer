import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
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
clf=KNeighborsClassifier()
pipe=Pipeline(steps=[('pca',pca),('standardScaler',standardScaler),('clf',clf)])
param_grid={'pca__n_components':[x for x in np.arange(3,33,3)],
'clf__n_neighbors':[x for x in np.arange(1,31,3)],
'clf__weights'    :['uniform','distance'],
'clf__leaf_size'  :[x for x in np.arange(1,51,3)],
'clf__metric'     :['euclidean','manhattan','chebyshev']
}
search = GridSearchCV(pipe,param_grid,scoring='accuracy',cv=5,refit=True,n_jobs=-1)
search.fit(X_train,y_train)
print('best hyperparameters:{}'.format(search.best_params_))
score_train=search.score(X_train,y_train)
score_test=search.score(X_test,y_test)
print('score on training set/testing set:{:.2f}/{:.2f}'.format(score_train,score_test))
