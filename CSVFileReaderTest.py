import pandas as pd

data = pd.read_csv(r"C:\Users\USER\Desktop\spam_mail\spam.csv", encoding="utf-8-sig")

print("=== 데이터 미리보기 ===")
print(data.head())

print("\n=== 열 이름 확인 ===")
print(data.columns)

print("\n=== 라벨 개수 확인 ===")
print(data["label"].value_counts())