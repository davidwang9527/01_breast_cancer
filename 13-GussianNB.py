from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import GaussianNB

#step1:Loading data
X,y=load_breast_cancer(return_X_y=True)

#step2:Split data
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2,random_state=40,stratify=y)

#step3:Training--GussianNB
clf=GaussianNB()
clf.fit(X_train,y_train)
score_train=clf.score(X_train,y_train)
score_test=clf.score(X_test,y_test)
print('scores for GaussianNB(score on training set/testing set):{:.2f}/{:.2f}'.format(score_train,score_test))
