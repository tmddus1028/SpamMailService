import pandas as pd
from sklearn.naive_bayes import GaussianNB

# 1. CSV 파일 경로 입력
# 본인 CSV 파일 경로로 바꾸세요.
csv_path = r"C:\Users\USER\Desktop\spam_mail\spam.csv"

# 2. CSV 파일 읽기
data = pd.read_csv(csv_path, encoding="utf-8-sig")

# 3. 데이터 확인
print("=== 데이터 미리보기 ===")
print(data.head())

print("\n=== 열 이름 확인 ===")
print(data.columns)

print("\n=== 라벨 개수 확인 ===")
print(data["label"].value_counts())


# 4. 메일 문장을 숫자 특징으로 바꾸는 함수
def extract_features(text):
    text = str(text)

    # 기본 특징
    length = len(text)                      # 글자 수
    exclamation_count = text.count("!")     # 느낌표 개수

    # 스팸 관련 단어 포함 여부
    has_free = 1 if "무료" in text else 0
    has_click = 1 if "클릭" in text or "누르" in text else 0
    has_prize = 1 if "당첨" in text or "상품" in text or "보상" in text else 0
    has_urgent = 1 if "긴급" in text or "지금" in text else 0
    has_link = 1 if "http" in text or "www" in text or "링크" in text else 0

    return [
        length,
        exclamation_count,
        has_free,
        has_click,
        has_prize,
        has_urgent,
        has_link
    ]


# 5. 입력 데이터 X와 정답 y 만들기
X = data["message"].apply(extract_features).tolist()
y = data["label"]

print("\n=== 숫자로 바뀐 데이터 예시 ===")
print(X[:5])


# 6. GNB 모델 생성
model = GaussianNB()

# 7. 모델 학습
model.fit(X, y)

print("\nGNB 모델 학습 완료!")


# 8. 새 메일 테스트
test_mail = "무료 상품에 당첨되었습니다 지금 링크를 누르세요"

test_features = [extract_features(test_mail)]

prediction = model.predict(test_features)[0]
probability = model.predict_proba(test_features)[0]

print("\n=== 예측 테스트 ===")
print("입력 문장:", test_mail)
print("숫자 특징:", test_features)
print("예측 결과:", prediction)
print("라벨 순서:", model.classes_)
print("확률:", probability)