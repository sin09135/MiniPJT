import streamlit as st
import joblib
import numpy as np
import pandas as pd

def load_data_and_models():
    # 모델 불러오기
    model1 = joblib.load("models/gm_model.pkl")
    model2 = joblib.load("models/ngm_model.pkl")

    # 데이터 프레임 불러오기
    df1 = pd.read_csv('data/골목상권.csv')
    model_cols = ['매출','기준_년_코드','상권_구분_코드_명','상권_코드','상권_코드_명','시간대1', '시간대2', '시간대3', '시간대4', '시간대5', '분기_1', '분기_2', '분기_3',
           '총 가구 수', '총_직장인구_수', '상권내_총_아파트_세대_수', '배후지_총_아파트_세대_수',
            '시간대_생활인구_수', '평일_생활인구_평균', '주말_생활인구_평균', '면적당_점포_수',
           '직장인구/상주인구', '면적당_집객시설_수']

    df1 = df1[model_cols]

    return model1,df1

def main():
    # 데이터와 모델 로드
    model1, df1 = load_data_and_models()

    # 메인 페이지
    st.title('강남구 편의점 매출 예측 서비스')
    st.header('골목상권 구조')

    # 상권코드명 -임시
    unique_market = df1['상권_코드_명'].unique().tolist()
    selected_feature1 = st.sidebar.selectbox("상권코드명 선택", unique_market)

    # 시간대 - 임시
    unique_time = ['시간대1', '시간대2', '시간대3', '시간대4', '시간대5', '시간대6']
    selected_feature2 = st.sidebar.selectbox("시간대 선택", unique_time)

    # 분기 - 임시
    unique_quarter = ['1분기', '2분기', '3분기', '4분기']
    selected_feature3 = st.sidebar.multiselect("분기 선택", unique_quarter)

    # 변수 영역
    feature_names = ['시간대1', '시간대2', '시간대3', '시간대4', '시간대5', '분기_1', '분기_2', '분기_3',
           '총 가구 수', '총_직장인구_수', '상권내_총_아파트_세대_수', '배후지_총_아파트_세대_수',
            '시간대_생활인구_수', '평일_생활인구_평균', '주말_생활인구_평균', '면적당_점포_수',
           '직장인구/상주인구', '면적당_집객시설_수']

    # 각 피처에 대해 슬라이더 또는 텍스트 입력 상자 생성.
    user_input = []
    for i, feature_name in enumerate(feature_names):
        max_value_feature = float(df1[feature_name].max())
        min_value_feature = float(df1[feature_name].min())

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
        if len(numeric_user_input) == len(feature_names):
            numeric_user_input = np.array(numeric_user_input).reshape(1, -1) 
            predictions = model1.predict(numeric_user_input)

            # 예측 결과 출력
            if predictions is not None:
                st.subheader('예측 결과')
                st.write(predictions)

if __name__ == "__main__":
    main()
