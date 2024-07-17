import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split

# 가상의 예제 데이터 생성
# 실제로는 환자 데이터, 약물 데이터, 생체 지표 등으로 이루어진 데이터셋 사용
X = np.random.rand(1000, 10)  # 1000명의 환자, 10개의 특징
y = np.random.rand(1000)  # 1000명의 환자에 대한 약물 투여량

# 데이터를 훈련 데이터와 테스트 데이터로 분리
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 신경망 모델 정의
model = Sequential()
model.add(Dense(64, input_dim=X_train.shape[1], activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1))  # 약물 투여량을 예측하는 하나의 출력 뉴런

model.compile(optimizer='adam', loss='mean_squared_error')

# 모델 훈련 과정
model.fit(X_train, y_train, epochs=100, batch_size=32, verbose=1, validation_split=0.2)

# 모델을 평가
mse = model.evaluate(X_test, y_test)
print("Mean Squared Error:", mse)

# 새로운 데이터 예측
new_data = np.random.rand(1, 10)  # 가상의 새로운 환자 데이터
predicted_dosage = model.predict(new_data)
print("Predicted Dosage:", predicted_dosage)
