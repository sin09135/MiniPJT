# -*- coding:utf-8 -*-

# import streamlit as st
# from main_page import run_main_page
# from sub_page import run_sub_page
# # from 폴더명.파일명 import 함수명

# def main():

#     st.markdown("# Hello World")
#     menu = ["Home", "메인 페이지", "상세 페이지", "About"]
#     choice = st.sidebar.selectbox("메뉴", menu)

#     if choice == "Home":
#         st.subheader("Home")
#     elif choice == "메인 페이지":
#         #st.subheader("탐색적 자료 분석")
#         run_main_page()
#     elif choice == "상세 페이지":
#         #st.subheader("머신러닝")
#         run_sub_page()
#     elif choice == "About":
#         st.subheader("About")
#     else:
#         pass

# if __name__ == "__main__":
#     main()

# import streamlit as st
# from streamlit_option_menu import option_menu
# import main_page, sub_page
# import streamlit.components.v1 as html
# from  PIL import Image
# import numpy as np
# import pandas as pd
# import plotly.express as px
# import io

# st.set_page_config(
#     page_title = "강남구 편의점 매출 예측 서비스"
# )

# class MultiApp:

#     def __init__(self):
#         self.apps = []
#     def add_app(self, title, function):
#         self.apps.append({
#             "title":title,
#             "function":function
#         })
#     def run():
#         with st.sidebar:
#             choose = option_menu("Pages", ["About", "메인 페이지", "상세 페이지"],
#                                 icons=['Book', 'Bar chart', 'Search'],
#                                 menu_icon="app-indicator", default_index=0,
#                                 styles={
#                 "container": {"padding": "5!important", "background-color": "#fafafa"},
#                 "icon": {"color": "white", "font-size": "25px"}, 
#                 "nav-link": {"font-size": "16px", "text-align": "left", "margin":"0px", "--hover-color": "#eee"},
#                 "nav-link-selected": {"background-color": "#02ab21"},
#             })

#             if choose == "메인 페이지":
#                 main_page.run_main_page()
#             if choose == "상세 페이지":
#                 sub_page.run_sub_page()
#     run()
import streamlit as st
from streamlit_option_menu import option_menu

st.header('Home')
st.write('프로젝트 소개, 앱 이용방법 등을 소개하는 페이지')

def main_page():
    st.header("프로젝트 소개")

def sub_page():
    st.subheader("Sub")

# with st.sidebar:
#     selected_page = option_menu(
#         "selected Page",
#         ("main","sub"),
#         icons = ['house','gear']
#     )

# if selected_page == "main":
#     main_page()
# if selected_page == "sub":
#     sub_page()
