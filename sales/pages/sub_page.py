import streamlit as st
import joblib
import numpy as np
import pandas as pd


# 모델 불러오기
model1 = joblib.load("models/gm_model.pkl")
model2 = joblib.load("models/ngm_model.pkl")

# 데이터 프레임 불러오기
df = pd.read_csv('data/전체_수정_streamlit용.csv')
df1 = pd.read_csv('data/골목_streamlit용.csv')
df2 = pd.read_csv('data/비골목_streamlit용.csv')

# main text
st.title('강남구 편의점 매출 예측 서비스')
st.header('시간대별/ 분기별')


# side bar 
with st.sidebar:
    # Select market
    unique_market = df['상권_코드_명'].unique().tolist()
    selected_feature1 = st.selectbox("상권을 선택하세요", unique_market)

    # Select time
    unique_time = ['시간대1', '시간대2', '시간대3', '시간대4', '시간대5', '시간대6']
    selected_feature2 = st.selectbox("시간대를 선택하세요", unique_time)

    # Select quarter
    unique_quarter = ['1분기', '2분기', '3분기', '4분기']
    selected_feature3 = st.selectbox("분기를 선택하세요!", unique_quarter)

## 지도 영역


## 변수 영역
feature_names_gol = df1.iloc[:, 7:].columns.tolist() 
feature_names_ngol = df2.iloc[:, 7:].columns.tolist() 
# st.write(feature_names_gol)
# st.write(feature_names_ngol)
# 시간대, 분기 제외한 피쳐 slider로 입력
user_input = []


if selected_feature1 in df1['상권_코드_명'].tolist():
    for i, feature_name in enumerate(feature_names_gol):
        max_value_feature = float(df1[feature_name].max())
        min_value_feature = float(df1[feature_name].min())
        
        # 각 피쳐당 22년 평균을 default 값으로 설정
        condition = (df1['상권_코드_명'] == selected_feature1) & (df1['기준_년_코드'] == 2022)
        value = df1.loc[condition, feature_name]
        default_value = value.mean()
        #default_value = (max_value_feature + min_value_feature) / 2

        user_input.append(st.slider(f"{feature_name}:", min_value=min_value_feature, max_value=max_value_feature, value=default_value))

else:
    for i, feature_name in enumerate(feature_names_ngol):
        max_value_feature = float(df2[feature_name].max())
        min_value_feature = float(df2[feature_name].min())

        # 각 피쳐당 22년 평균을 default 값으로 설정
        condition = (df2['상권_코드_명'] == selected_feature1) & (df2['기준_년_코드'] == 2022) 
        value = df2.loc[condition, feature_name]
        default_value = value.mean()
    
        user_input.append(st.slider(f"{feature_name}:", min_value=min_value_feature, max_value=max_value_feature, value=default_value))

# ----------------------------------------------------- 시간대, 분기 값 리스트의 앞에 넣기------------------------------------------------------
# 초기식
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

# 예측
if st.button("예측하기"):
    # 입력 데이터를 2D 배열로 변환 - 골목상권, 비골목상권의 피쳐 개수 만큼
    if selected_feature1 in df1['상권_코드_명'].tolist():
        if len(numeric_user_input) == len(feature_names_gol) + 8: 
            numeric_user_input = np.array(numeric_user_input).reshape(1, -1) 
            predictions = model1.predict(numeric_user_input)

            # 예측 결과 출력
            if predictions is not None:
                st.subheader('예측 결과')
                st.write(f"{selected_feature1}의 {selected_feature3} {selected_feature2} 예상 매출은 {predictions[0]:,.0f}원입니다.")
    else: 
        if len(numeric_user_input) == len(feature_names_ngol) + 8:
            numeric_user_input = np.array(numeric_user_input).reshape(1, -1) 
            predictions = model2.predict(numeric_user_input)

            # 예측 결과 출력
            if predictions is not None:
                st.subheader('예측 결과')
                st.write(f"{selected_feature1}의 {selected_feature3} {selected_feature2} 예상 매출은 {predictions[0]:,.0f}원입니다.")


