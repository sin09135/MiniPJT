import streamlit as st
import joblib
import numpy as np
import pandas as pd

# 모델 불러오기
model1 = joblib.load("models/gm_model.pkl")
model2 = joblib.load("models/ngm_model.pkl")

# 데이터 프레임 불러오기
df = pd.read_csv('data/전체_수정_streamlit용.csv')
df1 = pd.read_csv('data/골목상권.csv')
df2 = pd.read_csv('data/비골목상권(수정).csv')

model_cols = [
        '매출','기준_년_코드','상권_구분_코드_명','상권_코드','상권_코드_명','시간대1', '시간대2', '시간대3', '시간대4', '시간대5', 
        '분기_1', '분기_2', '분기_3','총 가구 수', '총_직장인구_수', '상권내_총_아파트_세대_수', '배후지_총_아파트_세대_수',
        '시간대_생활인구_수', '평일_생활인구_평균', '주말_생활인구_평균', '면적당_점포_수', '직장인구/상주인구', '면적당_집객시설_수'
    ]

df1 = df1[model_cols]
gol_col = df1['상권_코드_명'].tolist()


### 메인 페이지
st.title('강남구 편의점 매출 예측 서비스')
st.header('시간대별/ 분기별')

## side bar

# 상권코드명 -임시s
unique_market = df['상권_코드_명'].unique().tolist()
selected_feature1 = st.sidebar.selectbox("상권을 선택하세요", unique_market)

# 시간대 - 임시
unique_time = ['시간대1', '시간대2', '시간대3', '시간대4', '시간대5', '시간대6']
selected_feature2 = st.sidebar.selectbox("시간대를 선택하세요", unique_time)

# 분기 - 임시
unique_quarter = ['1분기', '2분기', '3분기', '4분기']
selected_feature3 = st.sidebar.selectbox("분기를 선택하세요!", unique_quarter)

## 지도 영역


## 변수 영역
feature_names_gol = df1.iloc[:, 13:].columns.tolist() 
feature_names_ngol = df2.iloc[:, 13:].columns.tolist() 

# 시간대, 분기 제외한 피쳐 slider로 입력
user_input = []


if selected_feature1 in df1['상권_코드_명'].tolist():
    for i, feature_name in enumerate(feature_names_gol):
        max_value_feature = float(df1[feature_name].max())
        min_value_feature = float(df1[feature_name].min())
        
        condition = (df['상권_코드_명'] == selected_feature1) & (df['기준_년_코드'] == 2022)
        value = df.loc[condition, feature_name]
        default_value = value.mean()

        user_input.append(st.slider(f"{feature_name}:", min_value=min_value_feature, max_value=max_value_feature, value=default_value))

else:
     for i, feature_name in enumerate(feature_names_ngol):
        max_value_feature = float(df2[feature_name].max())
        min_value_feature = float(df2[feature_name].min())

        condition = (df['상권_코드_명'] == selected_feature1) & (df['기준_년_코드'] == 2022)
        value = df.loc[condition, feature_name]
        default_value = value.mean()

        user_input.append(st.slider(f"{feature_name}:", min_value=min_value_feature, max_value=max_value_feature, value=default_value))

# ----------------------------------------------------- 시간대, 분기 값 리스트의 앞에 넣기------------------------------------------------------
time1, time2, time3, time4, time5, quarter1, quarter2, quarter3 = 0, 0, 0, 0, 0, 0, 0, 0

if selected_feature2 == unique_time[0]:
    time1 = 1
    time2 = 0
    time3 = 0
    time4 = 0
    time5 = 0
elif selected_feature2 == unique_time[1]:
    time1 = 0
    time2 = 1
    time3 = 0
    time4 = 0
    time5 = 0
elif selected_feature2 == unique_time[2]:
    time1 = 0
    time2 = 0
    time3 = 1
    time4 = 0
    time5 = 0
elif selected_feature2 == unique_time[3]:
    time1 = 0
    time2 = 0
    time3 = 0
    time4 = 1
    time5 = 0
elif selected_feature2 == unique_time[4]:
    time1 = 0
    time2 = 0
    time3 = 0
    time4 = 0
    time5 = 1
else:
    time1 = 0
    time2 = 0
    time3 = 0
    time4 = 0
    time5 = 0

if selected_feature3 == unique_quarter[0]:
    quarter1 = 1
    quarter2 = 0
    quarter3 = 0
elif selected_feature3 == unique_quarter[1]:
    quarter1 = 0
    quarter2 = 1
    quarter3 = 0
elif selected_feature3 == unique_quarter[2]:
    quarter1 = 0
    quarter2 = 0
    quarter3 = 1
else :
    quarter1 = 0
    quarter2 = 0
    quarter3 = 0

user_input[:0] = [time1, time2, time3, time4, time5, quarter1, quarter2, quarter3]


# 문자열 데이터를 숫자로 변환
numeric_user_input = []
for value in user_input:
    try:
        numeric_value = float(value)
        numeric_user_input.append(numeric_value)
    # 예외처리
    except ValueError:
        st.error(f"입력값 '{value}'은(는) 숫자로 변환할 수 없습니다.")

# 예측 버튼
if st.button("예측하기"):
    # 입력 데이터를 2D 배열로 변환 - 골목상권, 비골목상권의 피쳐 개수 만큼
    if selected_feature1 in df1['상권_코드_명'].tolist():
        if len(numeric_user_input) == len(feature_names_gol) + 8: 
            numeric_user_input = np.array(numeric_user_input).reshape(1, -1) 
            predictions = model1.predict(numeric_user_input)

            # 예측 결과 출력
            if predictions is not None:
                st.subheader('예측 결과')
                st.write(predictions)
    else: 
        if len(numeric_user_input) == len(feature_names_ngol) + 8:
            numeric_user_input = np.array(numeric_user_input).reshape(1, -1) 
            predictions = model2.predict(numeric_user_input)

            # 예측 결과 출력
            if predictions is not None:
                st.subheader('예측 결과')
                st.write(predictions)