import streamlit as st
import pandas as pd

# 설문 문항 및 가중치 리스트
survey_questions = [
    ("나는 변수를 선언하고, 데이터 타입(문자열, 리스트, 딕셔너리 등)의 차이를 이해하여 적절히 사용할 수 있다.", 3),
    ("나는 if-else 조건문과 for/while 반복문을 활용하여 주어진 문제를 해결하는 코드를 작성할 수 있다.", 3),
    ("나는 업무에서 반복적으로 수행하는 작업(엑셀 데이터 정리, 보고서 자동화 등)을 파이썬으로 자동화할 수 있다.", 4),
    ("나는 필요한 파이썬 라이브러리(Numpy, Pandas 등)를 설치하고, 환경 설정을 올바르게 구성할 수 있다.", 4),
    ("나는 패키지 설치 중 발생하는 오류를 분석하고 해결할 수 있으며, 가상환경을 활용할 수 있다.", 4),
    ("나는 평균, 중앙값, 표준편차 등의 개념을 이해하고, 이를 사용하여 데이터의 분포를 설명할 수 있다.", 4),
    ("나는 Pandas, Matplotlib, Seaborn을 활용하여 데이터의 기초적인 탐색적 분석(EDA)을 수행할 수 있다.", 4),
    ("나는 파이썬을 사용하여 회귀 분석을 수행하고, 변수 간의 상관관계를 해석할 수 있다.", 4),
    ("나는 데이터 시각화(히스토그램, 상자그림 등)를 통해 데이터의 특성을 파악하고 보고할 수 있다.", 5),
    ("나는 통계적 가설 검정(예: T-test, 카이제곱 검정)의 개념을 이해하고 실제 데이터를 기반으로 수행할 수 있다.", 5),
    ("나는 지도 학습(Supervised Learning)과 비지도 학습(Unsupervised Learning)의 차이를 이해하고 있으며, 적절한 상황에서 활용할 수 있다.", 5),
    ("나는 주요 머신러닝 알고리즘(로지스틱 회귀, 랜덤포레스트, SVM 등)의 기본 개념을 이해하고 있으며, 이를 실제 데이터에 적용할 수 있다.", 5),
    ("나는 결측치 처리(삭제, 대체)와 정규화/표준화 등의 데이터 전처리 기법을 이해하고 있으며, 데이터 품질을 개선할 수 있다.", 5),
    ("나는 파이썬의 Scikit-Learn을 사용하여 모델을 학습시키고, 평가 지표(정확도, RMSE 등)를 분석할 수 있다.", 5),
    ("나는 과적합(Overfitting)과 과소적합(Underfitting)의 개념을 이해하고, 하이퍼파라미터 조정을 통해 모델을 개선할 수 있다.", 5),
    ("나는 설명 가능한 AI(XAI)의 개념을 이해하고 있으며, 모델의 예측 결과를 해석할 수 있는 기법을 알고 있다.", 7),
    ("나는 딥러닝(Deep Learning)의 기본 구조(신경망, CNN, RNN 등)를 이해하고 있으며, 실제 적용 경험이 있다.", 7),
    ("나는 머신비전(Computer Vision) 관련 기술(OpenCV, TensorFlow/Keras 등)을 사용하여 이미지 데이터를 분석한 경험이 있다.", 7),
    ("나는 하이퍼파라미터 튜닝(Grid Search, Bayesian Optimization 등)을 활용하여 모델의 성능을 향상시킬 수 있다.", 7),
    ("나는 최적화 기법(Gradient Descent, Adam Optimizer 등)을 이해하고, 모델의 성능을 개선할 수 있다.", 7)
]

responses = []

st.title("📝 Python 데이터 분석 역량 평가")
st.write("각 문항에 대해 아래 중 하나를 선택하세요.")
st.write("\n")

options = ["아주 그렇지 않다", "그렇지 않다", "보통이다", "그렇다", "아주 그렇다"]

def calculate_score(weight, response):
    """환산 점수 계산"""
    return weight * ((options.index(response) + 1) / 5)

def determine_level(total_score):
    """총점에 따라 레벨 결정"""
    if total_score < 25:
        return "Lv1-1"
    elif total_score < 50:
        return "Lv1-2"
    elif total_score < 75:
        return "Lv2"
    else:
        return "Lv3"

# 사용자 정보 입력
st.sidebar.header("수강생 정보 입력")
name = st.sidebar.text_input("이름을 입력하세요")
email = st.sidebar.text_input("이메일을 입력하세요")

# 설문을 5개씩 나누어 보여주기
for i in range(0, len(survey_questions), 5):
    with st.expander(f"질문 {i+1} ~ {i+5 if i+5 < len(survey_questions) else len(survey_questions)}"):
        for question, weight in survey_questions[i:i+5]:
            response = st.radio(question, options, index=2)
            responses.append((weight, response))

if st.button("결과 확인하기"):
    total_score = sum(calculate_score(weight, response) for weight, response in responses)
    level = determine_level(total_score)
    
    st.write("## 🎉 평가 완료! 🎉")
    st.write(f"총점: **{total_score:.2f}**점입니다.")
    st.write(f"당신의 배정된 레벨은 **{level}** 입니다.")
    st.write(f"지금 바로 **{level}** 과정을 수강신청하고 한 단계 더 성장해볼까요? 💪")
    
    # 수강생 데이터 저장
    data = {"이름": [name], "이메일": [email], "총점": [total_score], "레벨": [level]}
    df = pd.DataFrame(data)
    
    # CSV 파일 저장
    df.to_csv("survey_results.csv", mode='a', header=False, index=False)
    
    st.write("📊 설문 응답이 저장되었습니다! 관리자 페이지에서 확인하세요.")

