from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

#step1:Loading data
X,y=load_breast_cancer(return_X_y=True)

#step2:Split data
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2,random_state=40,stratify=y)

#step3:training
svc=SVC(class_weight='balanced',random_state=40)
param_grid=[
    {'kernel':['linear'],'C':[0.01,0.03,0.1,0.3,1]}
    ]
clf = GridSearchCV(estimator=svc,param_grid=param_grid,scoring='accuracy',cv=5,refit=True,verbose=1,n_jobs=-1)
clf.fit(X_train,y_train)
score_train=clf.score(X_train,y_train)
score_test=clf.score(X_test,y_test)
print('best hyperparameters for svc:{}'.format(clf.best_params_))
print('scores for svc(score on training set/testing set):{:.2f}/{:.2f}'.format(score_train,score_test))
