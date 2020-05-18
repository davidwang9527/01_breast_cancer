import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV

#step1:Loading data
X,y=load_breast_cancer(return_X_y=True)

#step2:Split data
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2,random_state=40,stratify=y)

#step3:Feature Engineering
pca=PCA()
standardScaler=StandardScaler()

#step4:Training
clf=LogisticRegression(random_state=40,class_weight='balanced',max_iter=10000)
pipe=Pipeline(steps=[('pca',pca),('standardScaler',standardScaler),('clf',clf)])
param_grid=[{'pca__n_components':[x for x in np.arange(3,33,3)],'clf__solver':['liblinear'],'clf__penalty':['l1','l2'],'clf__C':[0.01,0.03,0.1,0.3,1.0,3,10,30,100,300,1000,3000,10000],'clf__intercept_scaling':[0.01,0.03,0.1,0.3,1,3,10]},
    {'pca__n_components':[x for x in np.arange(3,33,3)],'clf__solver':['newton-cg','sag','lbfgs'],'clf__penalty':['l2'],'clf__C':[0.01,0.03,0.1,0.3,1.0,3,10,30,100,300,1000,3000,10000]},
    {'pca__n_components':[x for x in np.arange(3,33,3)],'clf__solver':['saga'],'clf__penalty':['elasticnet'],'clf__l1_ratio':[0,0.1,0.3,0.6,0.9,1],'clf__C':[0.01,0.03,0.1,0.3,1.0,3,10,30,100,300,1000,3000,10000]}
]
search = RandomizedSearchCV(pipe,param_grid,scoring='accuracy',cv=5,refit=True,n_jobs=-1)
search.fit(X_train,y_train)
print('best hyperparameters:{}'.format(search.best_params_))
score_train=search.score(X_train,y_train)
score_test=search.score(X_test,y_test)
print('score on training set/testing set:{:.2f}/{:.2f}'.format(score_train,score_test))
