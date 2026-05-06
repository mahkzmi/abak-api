# train_model.py (نسخه اصلاح شده)
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import joblib

df = pd.read_csv('synthetic_soc_data.csv')

X = df[['ndvi', 'precipitation', 'temperature', 'elevation', 'clay']]
y = df['soc']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=100, max_depth=6, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"✅ مدل دوباره آموزش دید!")
print(f"میانگین خطای مطلق (MAE): {mae:.2f}%")
print(f"ضریب تعیین (R²): {r2:.2f}")

joblib.dump(model, 'carbon_model_v2.pkl')
print("✅ مدل جدید در فایل carbon_model_v2.pkl ذخیره شد.")