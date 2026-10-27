import pandas as pd

df = pd.read_csv("data/raw/vehicles.csv")

print("So dong, so cot:", df.shape)
print("\nCac cot:", df.columns.tolist())
print("\nThieu du lieu (%):")
print((df.isna().mean() * 100).round(1).sort_values(ascending=False))
print("\nThong ke price / year / odometer:")
print(df[["price", "year", "odometer"]].describe().round(0))
print("\nTop 15 hang xe:")
print(df.manufacturer.value_counts().head(15))
