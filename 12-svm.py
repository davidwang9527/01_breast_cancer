from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

#step1:Loading data
X,y=load_breast_cancer(return_X_y=True)

#step2:Split data
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2,random_state=40,stratify=y)

#step3:Feature Engineering
pca=PCA()
standardScaler=StandardScaler()

#step4:training
clf=SVC(class_weight='balanced',random_state=40)
pipe=Pipeline(steps=[('pca',pca),('standardScaler',standardScaler),('clf',clf)])
param_grid=[
    {'pca__n_components':[x for x in [24,25,26,27,28,29,30]],'clf__kernel':['linear'],'clf__C':[0.003,0.01,0.03,0.1,0.3,1]},
    {'pca__n_components':[x for x in [24,25,26,27,28,29,30]],'clf__kernel':['rbf'],'clf__C':[0.01,0.03,0.1,0.3,1],'clf__gamma':[0.01,0.03,0.1,0.3,1]}
]
search = GridSearchCV(pipe,param_grid,scoring='accuracy',cv=5,refit=True,n_jobs=-1)
search.fit(X_train,y_train)
print('best hyperparameters:{}'.format(search.best_params_))
score_train=search.score(X_train,y_train)
score_test=search.score(X_test,y_test)
print('score on training set/testing set:{:.2f}/{:.2f}'.format(score_train,score_test))
