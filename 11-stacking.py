from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import RidgeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import StackingClassifier

#step1:Loading data
X,y=load_breast_cancer(return_X_y=True)

#step2:Split data
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2,random_state=40,stratify=y)

#step3:Training
clf=StackingClassifier(
  estimators=[('knn',KNeighborsClassifier(n_neighbors=7,weights='distance',leaf_size=1, metric='manhattan')),
               ('ridge',RidgeClassifier(random_state=40,class_weight='balanced',alpha=0.00001)),
  ],
  final_estimator=LogisticRegression(random_state=40,class_weight='balanced',max_iter=10000),
  cv=5,
  n_jobs=-1
)
clf.fit(X_train,y_train)
score_train=clf.score(X_train,y_train)
score_test=clf.score(X_test,y_test)
print('scores for StackingClassifier(score on training set/testing set):{:.2f}/{:.2f}'.format(score_train,score_test))
