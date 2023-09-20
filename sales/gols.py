import streamlit as st
import joblib
import numpy as np
import pandas as pd

# 데이터 및 모델 로드

model1 = joblib.load("models/gm_model.pkl")
model2 = joblib.load("models/ngm_model.pkl")

df = pd.read_csv('data/전체_수정_streamlit용.csv')
df1 = pd.read_csv('data/골목상권.csv')
df2 = pd.read_csv('data/비골목상권.csv')
    
model_cols = [
        '매출','기준_년_코드','상권_구분_코드_명','상권_코드','상권_코드_명','시간대1', '시간대2', '시간대3', '시간대4', '시간대5', 
        '분기_1', '분기_2', '분기_3','총 가구 수', '총_직장인구_수', '상권내_총_아파트_세대_수', '배후지_총_아파트_세대_수',
        '시간대_생활인구_수', '평일_생활인구_평균', '주말_생활인구_평균', '면적당_점포_수', '직장인구/상주인구', '면적당_집객시설_수'
    ]

st.write(type(df['분기']))
df1 = df1[model_cols]

# 전역변수
feature_names_gol = df1.iloc[:, 5:].columns.tolist()
feature_names_ngol = df2.iloc[:, 5:].columns.tolist()

# 기본값 설정
def get_default_values(df, selected_feature1, selected_feature2, selected_feature3):
    feature_names_gol = df1.iloc[:, 5:].columns.tolist()  # df1을 사용할 때
    feature_names_ngol = df2.iloc[:, 5:].columns.tolist()
    default_values = []

    # 특정 조건에 해당하는 데이터 추출
    if selected_feature1 in df['상권_코드_명'].unique().tolist():
        condition = (df['기준_년_코드'] == 2022)  & (df['상권_코드_명'] == selected_feature1) & (df['시간대'] == selected_feature2) & (df['분기'] == selected_feature3)
        selected_data = df.loc[condition, feature_names_gol]

        if not selected_data.empty:
            default_values = selected_data.iloc[0].tolist()  # 첫 번째 행의 값을 리스트로 변환
        else:
            # 조건에 해당하는 데이터가 없는 경우, 기본값 설정
            default_values = [0.0] * len(feature_names_gol)  # 기본값을 0.0으로 설정 (수정 필요)

        return default_values
    else:
        condition = (df['기준_년_코드'] == 2022)  & (df['상권_코드_명'] == selected_feature1) & (df['시간대'] == selected_feature2) & (df['분기'] == selected_feature3)
        selected_data = df.loc[condition, feature_names_ngol]
        st.dataframe(selected_data)

        if not selected_data.empty:
            default_values = selected_data.iloc[0].tolist()  # 첫 번째 행의 값을 리스트로 변환
        else:
            # 조건에 해당하는 데이터가 없는 경우, 기본값 설정
            default_values = [0.0] * len(feature_names_ngol)  # 기본값을 0으로 예외 처리

        return default_values


# 메인 함수
def main():
   
    # Streamlit 애플리케이션 설정
    st.title('강남구 편의점 매출 예측 서비스')
    st.header('골목상권 구조')

    # 사용자 입력 옵션 설정
    unique_market = df['상권_코드_명'].unique().tolist() 
    selected_feature1 = st.sidebar.selectbox("상권코드명 선택", unique_market) # str
    
    unique_time = df['시간대'].unique().tolist() 
    selected_feature2 = st.sidebar.selectbox("시간대 선택", unique_time) #int
    selected_feature2 = float(selected_feature2)

    unique_quarter = df['분기'].unique().tolist()
    selected_feature3 = st.sidebar.selectbox("분기 선택", unique_quarter) #int

    # st.write(type(selected_feature2))
    # st.write(type(selected_feature1))
    # st.write(type(selected_feature3))

    # 입력 항목 설정
    user_input = []
    if selected_feature1 in df1['상권_코드_명'].unique().tolist():
        st.write(df1['상권_코드_명'].unique().tolist())
        for i, feature_name in enumerate(feature_names_gol):
            max_value_feature = float(df1[feature_name].max())
            min_value_feature = float(df1[feature_name].min())

            default_value = get_default_values(df1, selected_feature1, selected_feature2, selected_feature3)  
            user_input.append(st.slider(f"{feature_name}:", min_value=min_value_feature, max_value=max_value_feature, value=default_value))
    else:
        for i, feature_name in enumerate(feature_names_ngol):
            max_value_feature = float(df2[feature_name].max())
            min_value_feature = float(df2[feature_name].min())

            default_value = get_default_values(df2, selected_feature1, selected_feature2, selected_feature3)  
            user_input.append(st.slider(f"{feature_name}:", min_value=min_value_feature, max_value=max_value_feature, value=default_value))


    # 입력값을 숫자로 변환
    numeric_user_input = []
    for value in user_input:
        try:
            numeric_value = float(value)
            numeric_user_input.append(numeric_value)
        except ValueError:
            st.error(f"입력값 '{value}'은(는) 숫자로 변환할 수 없습니다.")

    # 입력값 출력
    st.subheader('입력값')
    st.write(user_input)

    # 예측 버튼
   
    if st.button("예측하기"):
    # 입력 데이터를 2D 배열로 변환하고 모델로 예측 수행
        numeric_user_input = np.array(user_input).reshape(1, -1)

        if selected_feature1 in df1['상권_코드_명'].unique().tolist():
            if len(numeric_user_input) == len(feature_names_gol):
                if selected_feature1 in df1['상권_코드_명'].unique().tolist():
                    predictions = model1.predict(numeric_user_input)
                else : 
                    predictions = model2.predict(numeric_user_input)

                # 예측 결과 출력
                st.subheader('예측 결과')
                st.write(predictions)
        else:
            if len(numeric_user_input) == len(feature_names_ngol):
                if selected_feature1 in df1['상권_코드_명'].unique().tolist():
                    predictions = model1.predict(numeric_user_input)
                else : 
                    predictions = model2.predict(numeric_user_input)

                # 예측 결과 출력
                st.subheader('예측 결과')
                st.write(predictions)

if __name__ == "__main__":
    main()
