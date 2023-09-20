import streamlit as st
import joblib
import numpy as np
import pandas as pd

# 모델 불러오기
model1 = joblib.load("models/gm_model.pkl")
model2 = joblib.load("models/ngm_model.pkl")

# 데이터 프레임 불러오기
df1 = pd.read_csv('data/골목상권.csv')
df2 = pd.read_csv('data/비골목상권.csv')

df11 =  pd.read_csv('data/골목_streamlit.csv')
df22 =  pd.read_csv('data/비골목_streamlit.csv')

### 메인 페이지
st.title('강남구 편의점 매출 예측 서비스')
st.header('시간대별/ 분기별')

## side bar

# 상권코드명 -임시
unique_market =  df22['상권_코드_명'].unique().tolist()
selected_feature1 = st.sidebar.selectbox("상권코드명 선택", unique_market)

# 시간대 - 임시
unique_time = ['시간대1', '시간대2', '시간대3', '시간대4', '시간대5', '시간대6']
selected_feature2 = st.sidebar.selectbox("시간대 선택", unique_time)

# 분기 - 임시
unique_quarter = ['1분기', '2분기', '3분기', '4분기']
selected_feature3 = st.sidebar.multiselect("분기 선택", unique_quarter)

## 지도 영역


## 변수 영역
feature_ngols = [
"점포수","시간대1","시간대2","시간대3","시간대4","시간대5","분기_1","분기_2","분기_3"
,"총 상주인구 수","총 가구 수","총_직장인구_수"
,"아파트_단지_수","아파트_가격_1_억_미만_세대_수","아파트_가격_1_억_세대_수","아파트_가격_2_억_세대_수"
,"아파트_가격_3_억_세대_수","아파트_가격_4_억_세대_수","아파트_가격_5_억_세대_수","아파트_가격_6_억_이상_세대_수"
,"총_생활인구_수","시간대_생활인구_수","월요일_생활인구_수","화요일_생활인구_수","수요일_생활인구_수","목요일_생활인구_수"
,"금요일_생활인구_수","토요일_생활인구_수","일요일_생활인구_수","집객시설_수","관공서_수","은행_수","백화점_수"
,"숙박_시설_수","area","연령대_10_생활인구_수","연령대_20_생활인구_수","연령대_30_생활인구_수","연령대_40_생활인구_수"
,"연령대_50_생활인구_수","연령대_60_이상_생활인구_수","시간대_버스_승하차승객수","시간대_지하철_승하차승객수"
,"버스정류장_수","지하철역_수"]



# # 각 피처에 대해 슬라이더 또는 텍스트 입력 상자 생성.
user_input = []
for i, feature_name in enumerate(feature_ngols):  # feature_names를 feature_ngols로 변경
    max_value_feature = float(df1[feature_name].max())
    min_value_feature = float(df1[feature_name].min())

    if i == 32:
        # 15번째 피처에 대한 처리
        user_input.append(st.text_input(f"15번째 피처({feature_name}):"))
    else:
        default_value = (max_value_feature + min_value_feature) / 2.0
        user_input.append(st.slider(f"{feature_name}:", min_value=min_value_feature, max_value=max_value_feature, value=default_value))

# 문자열 데이터를 숫자로 변환
numeric_user_input = []
for value in user_input:
    try:
        numeric_value = float(value)
        numeric_user_input.append(numeric_value)
    except ValueError:
        st.error(f"입력값 '{value}'은(는) 숫자로 변환할 수 없습니다.")

# 예측 버튼
if st.button("예측하기"):
    # 입력 데이터를 2D 배열로 변환
    if len(numeric_user_input) == len(feature_ngols):  # feature_names를 feature_ngols로 변경
        numeric_user_input = np.array(numeric_user_input).reshape(1, -1) 
        predictions = model1.predict(numeric_user_input)

        # 예측 결과 출력
        if predictions is not None:
            st.subheader('예측 결과')
            st.write(predictions)
