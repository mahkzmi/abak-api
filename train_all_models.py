# train_all_models.py
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score, accuracy_score
import joblib

# بارگذاری داده
df = pd.read_csv('soil_data_complete.csv')

# ویژگی‌های ورودی (همون ۵ تا متغیر محیطی)
features = ['ndvi', 'precipitation', 'temperature', 'elevation', 'clay']
X = df[features]

# ============================================================
# مدل ۱: پیش‌بینی EC (شوری) - رگرسیون
# ============================================================
y_ec = df['ec']
X_train, X_test, y_train, y_test = train_test_split(X, y_ec, test_size=0.2, random_state=42)

model_ec = RandomForestRegressor(n_estimators=100, max_depth=6, random_state=42)
model_ec.fit(X_train, y_train)

y_pred = model_ec.predict(X_test)
print(f"✅ مدل EC - MAE: {mean_absolute_error(y_test, y_pred):.2f}, R²: {r2_score(y_test, y_pred):.2f}")
joblib.dump(model_ec, 'model_ec.pkl')

# ============================================================
# مدل ۲: پیش‌بینی pH (اسیدیته) - رگرسیون
# ============================================================
y_ph = df['ph']
X_train, X_test, y_train, y_test = train_test_split(X, y_ph, test_size=0.2, random_state=42)

model_ph = RandomForestRegressor(n_estimators=100, max_depth=6, random_state=42)
model_ph.fit(X_train, y_train)

y_pred = model_ph.predict(X_test)
print(f"✅ مدل pH - MAE: {mean_absolute_error(y_test, y_pred):.2f}, R²: {r2_score(y_test, y_pred):.2f}")
joblib.dump(model_ph, 'model_ph.pkl')

# ============================================================
# مدل ۳: پیش‌بینی مواد آلی (OM) - رگرسیون
# ============================================================
y_om = df['om']
X_train, X_test, y_train, y_test = train_test_split(X, y_om, test_size=0.2, random_state=42)

model_om = RandomForestRegressor(n_estimators=100, max_depth=6, random_state=42)
model_om.fit(X_train, y_train)

y_pred = model_om.predict(X_test)
print(f"✅ مدل OM - MAE: {mean_absolute_error(y_test, y_pred):.2f}, R²: {r2_score(y_test, y_pred):.2f}")
joblib.dump(model_om, 'model_om.pkl')

# ============================================================
# مدل ۴: پیش‌بینی بافت خاک (Texture) - طبقه‌بندی
# ============================================================
texture_mapping = {'sandy': 0, 'sandy_loam': 1, 'loam': 2, 'clay_loam': 3, 'clay': 4}
df['texture_code'] = df['texture'].map(texture_mapping)

y_texture = df['texture_code']
X_train, X_test, y_train, y_test = train_test_split(X, y_texture, test_size=0.2, random_state=42)

model_texture = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42)
model_texture.fit(X_train, y_train)

y_pred = model_texture.predict(X_test)
print(f"✅ مدل بافت خاک - دقت: {accuracy_score(y_test, y_pred):.2f}")
joblib.dump(model_texture, 'model_texture.pkl')
joblib.dump(texture_mapping, 'texture_mapping.pkl')

print("\n✅ همه ۴ مدل با موفقیت ذخیره شدند!")