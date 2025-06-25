from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Sample Dataset
X = [[1],[2],[3],[4]]
y = [2,4,6,8]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2)

# Model
model = LinearRegression()
model.fit(X_train, y_train)

# Prediction
predictions = model.predict(X_test)
print(predictions)
