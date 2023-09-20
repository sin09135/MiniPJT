# -*- coding:utf-8 -*-
import pandas as pd 

# 라벨 인코딩
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split # 데이터셋 분리
from sklearn.linear_model import LogisticRegression # 알고리즘 선택
import joblib # 모델 저장 시, 사용



# 데이터 불러오기
data = pd.read_csv("data/iris.csv")

# 종속변수에 라벨인코딩 부여
le = LabelBinarizer()
# print(le.fit(data['species']))
data['species'] = le.fit_transform(data['species'])
# print(le.classes_)
# print(data['species'])

# 독립변수, 종속변수로 분리
X = data.drop(columns=['species'])
y = data['species']

# 훈련, 테스트 데이터로 검증
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

model = LogisticRegression()
model.fit(X_train, y_train) # 모델 생성 코드, 모형 학습 코드

model_file = open("models/lgr_model_iris230901.pkl", "wb")
joblib.dump(model, model_file) # Export
model_file.close() 