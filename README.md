### 코드 설명

- dl-example.py
  - Scikit-learn 라이브러리의 머신러닝을 이용해, 훈련 데이터로 선형 회귀를 학습시키고, 랜덤으로 생성한 테스트 데이터를 가지고 MSE를 측정해서 얼마나 학습이 잘 되었는지 확인하는 예제 코드
- ml-example.py
  - Tensorflow 라이브러리의 딥러닝을 이용해, 훈련 데이터로 선형 회귀를 학습시키고, 랜덤으로 생성한 테스트 데이터를 가지고 MSE를 측정해서 얼마나 학습이 잘 되었는지 확인하는 예제 코드
- optimal-dose.py
  - 다층 퍼셉트론(MLP)을 가진 인공 신경망(ANN)을 사용해 최적의 약물 투여량을 찾는 예제 코드
  - X: 환자의 다양한 특징을 나타내는 수치형 데이터 (여기서는 각각이 10개의 특징을 가진 데이터셋, 이것이 실제로는 환자의 나이, 성별, 체중, 키 등의 변수일 수 있음)
  - Y: 각 환자에 대한 최적의 약물 투여량
  - 결과적으로 얻어지는 것: '비선형'의 회귀 모델. 이 비선형의 회귀 모델을 통해 임의의 입력 데이터(= 환자 데이터)에 대해 최적의 약물 투여량을 예측할 수 있음