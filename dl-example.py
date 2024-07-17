import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split

X = np.array([[1], [2], [3], [4], [5]])
y = np.array([1, 3, 3, 2, 5])

# 훈련 데이터와 테스트 데이터로 분리
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 모델 정의
model = Sequential()
model.add(Dense(10, input_dim=1, activation='relu'))
model.add(Dense(1))

# 모델 컴파일
model.compile(optimizer=Adam(learning_rate=0.01), loss='mean_squared_error')

# 모델 훈련
model.fit(X_train, y_train, epochs=100, verbose=0)

# 예측 및 성능 평가
y_pred = model.predict(X_test)
mse = np.mean((y_test - y_pred.flatten())**2)

print("Mean Squared Error:", mse)
print("Predictions:", y_pred.flatten())
