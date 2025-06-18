from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score

#Load Dataset
X = data.data
y = data.target

X_train,X_test,y_train = train_test_split(X,y,test_size = 0.2,random_state = 42)

#Train Logistics Regression model
model = LogisticRegression(max_iter =1000)
model.fit(X_train,y_train)

#Predict
y_pred = model.predict(X_test)

# Accuracy
print("Accuracy:",accuracy_score(y_test,y_pred))