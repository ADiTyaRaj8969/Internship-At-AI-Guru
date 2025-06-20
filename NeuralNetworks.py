import tensorflow as tf
from tensorflow.keras.model import Sequential
from tensorflow.keras.layers import Dense 
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder,StandardScaler
import numpy as np

iris = load_iris()
X = iris.data
y = iris.target.reshape(-1,1)

encoder = OneHotEncoder(sparse_output = False)
y_encoded = encoder.fit_transform(y)

Scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train , X_test ,y_train,y_test, = train_test_split(X_scaled,y_encoded,test_size =0.2,random_state=42)

model = Sequential([
                    Dense(10,input_shape =(4,), activation = 'relu'),
                    Dense(8,activation='relu'),
                    Dense(3,activation='softmax')
])
model.compile(
                     loss = 'categorical_crossentropy',
                 optimizer = 'adam',metrics =['accuracy']
)

model.fit(X_train,y_train,epochs = 50, batch_size = 5, verbose =1)
loss, accuracy = model.evaluate(X_test,y_test)
print(f"Test Accuracy: {accuracy*100:.2f}%")
predictions = model.predict(X_test)
predicted_classes = np.argmax(predictions,axis =1)
true_classes = np.argmax(y_test,axis =1)
print("Predicted:",predicted_classes)
print("Actual",true_classes)


