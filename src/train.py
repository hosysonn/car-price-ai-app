import json
import os
import sys

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split


TRAIN_YEAR = 2026 #nam hien tai dung de tinh tuoi xe.
QUICK = "--quick" in sys.argv #huan luyen nhanh de tim loi code
os.makedirs("models", exist_ok=True)

cols = ["price", "year", "manufacturer", "model", "odometer", "fuel",
        "transmission", "condition", "type", "drive", "cylinders",
        "title_status", "state"]
df = pd.read_csv("data/raw/vehicles.csv", usecols=cols)

df = df.dropna(subset=["price", "year", "manufacturer", "model", "odometer"]) #bo cac dong thieu thong tin cot loi , 3 thuoc tinh con lai thi xu li sau

#loai bo outliers , giu lai cac dong co chi so hop li
df = df[df.price.between(1000, 100000)]
df = df[df.odometer.between(1000, 400000)]
df = df[df.year.between(1995, TRAIN_YEAR)]

df = df.drop_duplicates() #bo trung lap(tranh ro ri du lieu)

#dien unknown cho gtri bi thieu , chuyen chu hoa thanh chu thuong va bo space 2 dau
# df[c] tra ve 1 cot duoi dang Series
for c in ["manufacturer", "model", "fuel", "transmission", "condition","type", "drive", "cylinders", "title_status", "state"]:
    df[c] = df[c].fillna("unknown").astype(str).str.lower().str.strip()


vc = df.manufacturer.value_counts() # dem so xe moi hang

# vc >= 300 cho True/False từng hãng: toyota True, ford True, tesla False.
# vc[vc >= 300] giữ lại các hãng True, còn toyota và ford.
# .index lấy tên hãng: ["toyota", "ford"].
# df.manufacturer.isin([...]) đánh dấu từng dòng xe có thuộc hãng đó không.
# df[mask] giữ lại các dòng True, tức bỏ hãng quá ít dữ liệu.

df = df[df.manufacturer.isin(vc[vc >= 300].index)] 

vc = df.model.value_counts() # dem so xe moi model

# vc[vc < 100].index là danh sách các model có dưới 100 xe.
# df.model.isin(...) đánh dấu các dòng thuộc những model hiếm đó.
# df.loc[mask, "model"] = "other" nghĩa là: tại các dòng True ở cột model, gán giá trị "other".


df.loc[df.model.isin(vc[vc < 100].index), "model"] = "other"

#tao dac trung
df["car_age"] = TRAIN_YEAR - df.year
df["km_per_year"] = df.odometer / (df.car_age + 1)
df = df.drop(columns=["year"]) #trung thon tin car_age
df["brand_model"] = df.manufacturer + "_" + df.model   # "ford_f-150"
df["log_odo"] = np.log1p(df.odometer)

cat_cols = ["manufacturer", "model", "fuel", "transmission", "condition",
            "type", "drive", "cylinders", "title_status", "state","brand_model"]
X = df.drop(columns = "price") # dau vao
Y = np.log1p(df.price) # nhan can predict , 
# np.log1p(price) = log(1 + giá): đây là mẹo quan trọng.
#Giá xe phân phối lệch phải (nhiều xe rẻ, ít xe đắt).
#Học trên log giúp mô hình quan tâm đến sai số tương đối (%) thay vì sai số tuyệt đối, 
#nên xe $3,000 và xe $50,000 được đối xử công bằng hơn.

# lan 1 chia 15% test vaf 85% train , lan 2 chia tiep 17.65% tu train cu lam test va con lai la train moi
# ---> train 70% , test ~ 15% , validate 15%
x_train , x_test , y_train , y_test = train_test_split(X,Y,test_size=0.15,random_state=42)
x_train1 , x_test1 , y_train1 , y_test1 = train_test_split(x_train,y_train,test_size=0.1765,random_state=42)

# huan luyen CatBoost
model = CatBoostRegressor(
    iterations = 300 if QUICK else 2000,
    learning_rate = 0.03,
    depth = 10,
    cat_features = cat_cols,
    early_stopping_rounds = 100,
    verbose = 0,
)

model.fit(x_train1,y_train1,eval_set = (x_test1,y_test1))
pred = np.expm1(model.predict(x_test))
true = np.expm1(y_test.values)

#mectrics 
mae  = mean_absolute_error(true, pred)
mape = float(np.mean(np.abs(pred - true) / true) * 100)
r2   = r2_score(true, pred)

print("mae : ", mae)
print("mape : ", mape)
print("r2 : ", r2)