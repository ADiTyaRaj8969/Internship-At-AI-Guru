from sklearn import datasets
from sklearn.model_selection import train_tests_split
from sklearn.svm import SVC
#Load Dataset
X,y = datasets.make_classification(n_samples = 100, n_features = 2,    n_classes=2,n_informative =2)

#split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.2)

model = SVC(kernel ='linear')
model.fit(X_train, y_train)

print("Accuracy", accuracy_score(y_test,y_pred))
