import streamlit as st
import joblib
import time
import numpy as np
import pandas as pd

# 모델 불러오기
model1 = joblib.load("models/gm_model.pkl")
model2 = joblib.load("models/ngm_model.pkl")

# Streamlit 앱 시작
st.title('강남구 편의점 매출 예측 서비스')
st.header('시간대별/ 분기별')

# Add a placeholder 진행 상황 바
latest_iteration = st.empty()
bar = st.progress(0)
for i in range(100):
    # Update the progress bar with each iteration.
    latest_iteration.text(f'Iteration {i+1}')
    bar.progress(i + 1)
    time.sleep(0.01)



# 사용자 입력을 받을 슬라이더 등을 추가
df1 = pd.read_csv('data/골목상권.csv')
# df2 = pd.read_csv('data/비골목상권.csv')

# 데이터의 피처 리스트를 만듭니다. 여기서는 예시로 5개의 피처를 사용합니다.
feature_names = ['시간대1', '시간대2', '시간대3', '시간대4', '시간대5']

# 각 피처에 대해 슬라이더를 생성합니다.
# 데이터의 피처 리스트를 만듭니다. 여기서는 예시로 5개의 피처를 사용합니다.

# 데이터의 피처 리스트를 만듭니다. 여기서는 예시로 5개의 피처를 사용합니다.
feature_names = [
    '시간대1', '시간대2', '시간대3', '시간대4', '시간대5',
    '분기_1', '분기_2', '분기_3',
    '총 상주인구 수', '총 가구 수', '총_직장인구_수',
    '시간대_생활인구_수', '관공서_수', '은행_수', '백화점_수',
    '숙박_시설_수', 'area', '시간대_버스_승하차승객수', '시간대_지하철_승하차승객수',
    '지하철역_수', '평일_생활인구_평균', '주말_생활인구_평균',
    '상권내_총_아파트_세대_수', '배후지_총_아파트_세대_수',
    '면적당_버스정류장_수', '면적당_점포_수', '면적당_집객시설_수',
    '직장인구/상주인구'
]

# 각 피처에 대해 슬라이더를 생성.
for feature_name in feature_names:
    max_value_feature = float(df1[feature_name].max())  # max_value를 float로 변환
    min_value_feature = float(df1[feature_name].min())  # min_value를 float로 변환
    default_value = (max_value_feature + min_value_feature) / 2.0  # 디폴트 값은 중간값으로 설정

    # 슬라이더 생성
    user_input = st.slider(f"{feature_name}:", min_value=min_value_feature, max_value=max_value_feature, value=default_value)


# #
# feature1 = st.slider("시간대1:", min_value=0, max_value=24, value=1)
# feature2 = st.slider("시간대2:", min_value=0, max_value=1000, value=0)
# feature3 = st.slider("시간대3:", min_value=0, max_value=24, value=0)
# feature4 = st.slider("시간대4:", min_value=0, max_value=24, value=0)
# feature5 = st.slider("시간대5:", min_value=0, max_value=24, value=0)
# feature6 = st.slider("분기1:", min_value=0, max_value=1000, value=1)
# feature7 = st.slider("분기2:", min_value=0, max_value=1000, value=0)
# feature8 = st.slider("분기3:", min_value=0, max_value=1000, value=0)
# feature9 = st.slider("총 상주인구 수:", min_value=0, max_value=1000, value=500)
# feature10 = st.slider("총 가구 수:", min_value=0, max_value=1000, value=500)
# feature11 = st.slider("총_직장인구_수:", min_value=0, max_value=1000, value=500)
# feature12 = st.slider("시간대_생활인구_수:", min_value=0, max_value=1000, value=500)
# feature13 = st.slider("관공서_수:", min_value=0, max_value=1000, value=500)
# feature14 = st.slider("은행_수:", min_value=0, max_value=1000, value=500)
# feature15 = st.slider("백화점_수:", min_value=0, max_value=1000, value=500)
# feature16 = st.slider("숙박_시설_수:", min_value=0, max_value=1000, value=500)
# feature17 = st.slider("area:", min_value=0, max_value=1000, value=500)
# feature18 = st.slider("시간대_버스_승하차승객수:", min_value=0, max_value=1000, value=500)
# feature19 = st.slider("시간대_지하철_승하차승객수:", min_value=0, max_value=1000, value=500)
# feature20 = st.slider("지하철역_수:", min_value=0, max_value=1000, value=500)
# feature21 = st.slider("평일_생활인구_평균:", min_value=0, max_value=1000, value=500)
# feature22 = st.slider("주말_생활인구_평균:", min_value=0, max_value=1000, value=500)
# feature23 = st.slider("상권내_총_아파트_세대_수:", min_value=0, max_value=1000, value=500)
# feature24 = st.slider("배후지_총_아파트_세대_수:", min_value=0, max_value=1000, value=500)
# feature25 = st.slider("면적당_버스정류장_수:", min_value=0, max_value=1000, value=500)
# feature26 = st.slider("면적당_점포_수:", min_value=0, max_value=1000, value=500)
# feature27 = st.slider("면적당_집객시설_수:", min_value=0, max_value=1000, value=500)
# feature28 = st.slider("직장인구/상주인구:", min_value=0, max_value=1000, value=500)



# # 사용자 입력값을 피처 형식에 맞게 데이터프레임으로 변환
# user_input = pd.DataFrame({
#     '시간대1': [feature1],
#     '시간대2': [feature2],
#     '시간대3': [feature3],
#     '시간대4': [feature4],
#     '시간대5': [feature5],
#     '분기1':[feature6],
#     '분기2':[feature7],
#     '분기3':[feature8],
#     '총 상주인구 수':[feature9],
#     '총 가구 수':[feature10],
#     '총_직장인구_수':[feature11],
#     '시간대_생활인구_수':[feature12],
#     '관공서_수':[feature13],
#     '은행_수':[feature14],
#     '백화점_수':[feature15],
#     '숙박_시설_수':[feature16],
#     'area':[feature17],
#     '시간대_버스_승하차승객수':[feature18],
#     '시간대_지하철_승하차승객수':[feature19],	
#     '지하철역_수':[feature20],
#     '평일_생활인구_평균':[feature21],	
#     '주말_생활인구_평균':[feature22],	
#     '상권내_총_아파트_세대_수':[feature23],
#     '배후지_총_아파트_세대_수':[feature24],
#     '면적당_버스정류장_수':[feature25],
#     '면적당_점포_수':[feature26],
#     '면적당_집객시설_수':[feature27],
#     '직장인구/상주인구':[feature28]})

# 예측 버튼
if st.button("예측"):
    # 예측 수행
    prediction = model1.predict(user_input)

    # 예측 결과 출력
    st.write(f"예측된 매출: {prediction[0]:.2f} 원")



