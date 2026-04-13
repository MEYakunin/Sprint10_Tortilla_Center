from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, mean_absolute_percentage_error
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Генерируем случайные данные
X, y = make_regression(
    n_samples=200,
    n_features=5,
    noise=15,
    random_state=42
)

'''
fig, axes = plt.subplots(1, 5, figsize=(28, 4))

for i in range(5):
    axes[i].scatter(X[:, i], y, color='dodgerblue', edgecolor='k', alpha=0.7)

plt.tight_layout()
plt.show()
'''
# Создаем модель
model = LinearRegression()

# Обучение модели
model.fit(X, y)

# Вывод коэффицентов модели
print(f'массив коэффициентов для каждого признака: {model.coef_}')
print(f'скаляр, свободный член модели (константа): {model.intercept_}')

# Предскозание моделью данных
y_pred = model.predict(X)

# Вывод предсказаных значений
print(y_pred[:10])

# Вычеслим оценку качества модели через MSE
mse = mean_squared_error(y, y_pred)

# Вычеслим оценку качества модели через R^2
r2 = r2_score(y, y_pred)

# Выводим значения метрик
print(f"MSE (Mean Squared Error): {mse:.3f}")
print(f"R² (Coefficient of Determination): {r2:.3f}")

# Расчитаем оставшиеся виды метрик
mae = mean_absolute_error(y, y_pred)
rmse = np.sqrt(mse)
non_zero_idx = y != 0
mape = mean_absolute_percentage_error(y[non_zero_idx], y_pred[non_zero_idx]) * 100

print("\nMAE:", round(mae, 3))
print("RMSE:", round(rmse, 3))
print("MAPE:", round(mape, 3))