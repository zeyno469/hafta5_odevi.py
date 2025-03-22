# Gerekli kütüphaneleri yükleme
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import normaltest, boxcox

# Veri setini yükleme
df = sns.load_dataset('tips')

# Veri setini inceleme
print("Veri Setinin İlk 5 Satırı:")
print(df.head())

print("\nVeri Seti Bilgisi:")
print(df.info())

print("\nEksik Değer Kontrolü:")
print(df.isnull().sum())

# Eksik değerleri medyan ile doldurma
df['total_bill'] = df['total_bill'].fillna(df['total_bill'].median())
df['tip'] = df['tip'].fillna(df['tip'].median())

# Kategorik değişkenleri One-Hot Encoding ile dönüştürme
df = pd.get_dummies(df, columns=['sex', 'smoker', 'day', 'time'], drop_first=True)

print("\nDönüştürülmüş Veri Setinin İlk 5 Satırı:")
print(df.head())

# Korelasyon matrisini oluşturma ve görselleştirme
print("\nKorelasyon Matrisi:")
corr_matrix = df.corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title('Korelasyon Matrisi')
plt.show()

# Bağımlı ve bağımsız değişkenleri ayırma
X = df.drop('tip', axis=1)
y = df['tip']

# Veriyi eğitim ve test setlerine ayırma
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Linear Regression modeli oluşturma ve eğitme
model = LinearRegression()
model.fit(X_train, y_train)

# Tahmin yapma
y_pred = model.predict(X_test)

# Model performansını değerlendirme
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"\nModel Performansı: MSE = {mse}, R2 = {r2}")

# Normallik testi (D'Agostino K^2)
stat, p = normaltest(df['tip'])
print(f"\nD\'Agostino K^2 Testi: p-değeri = {p}")

if p < 0.5:
    print("Başarılı: Veri normal dağılıma uygun.")
else:
    print("Başarısız: Veri normal dağılıma uygun değil.")

# Dönüştürme türlerini uygulama
# Box-Cox dönüşümü
df['tip_boxcox'], _ = boxcox(df['tip'] + 1)

# Log dönüşümü
df['tip_log'] = np.log(df['tip'] + 1)

# Karekök dönüşümü
df['tip_sqrt'] = np.sqrt(df['tip'])

print("\nDönüştürülmüş Verilerin İlk 5 Satırı:")
print(df[['tip', 'tip_boxcox', 'tip_log', 'tip_sqrt']].head())

# Dönüştürülmüş verilerin dağılımını görselleştirme
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
sns.histplot(df['tip_boxcox'], kde=True)
plt.title('Box-Cox Dönüşümü')

plt.subplot(1, 3, 2)
sns.histplot(df['tip_log'], kde=True)
plt.title('Log Dönüşümü')

plt.subplot(1, 3, 3)
sns.histplot(df['tip_sqrt'], kde=True)
plt.title('Karekök Dönüşümü')

plt.show()
