from sklearn.linear_model import SGDRegressor
from sklearn.datasets import make_regression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Генерируем случайные данные
X, y = make_regression(
    n_samples=200,
    n_features=5,
    noise=15,
    random_state=42
)

# Создаем модель
model_sgd = SGDRegressor(
    learning_rate='constant',
    eta0=0.01,
    max_iter=1000,
    tol=1e-3,
    random_state=42
)

# Обучение модели на данных
model_sgd.fit(X, y)

# Вывод коэффициентов и свободного члена
print("Коэффициенты:", model_sgd.coef_)
print("Свободный член:", model_sgd.intercept_)

# Получение и вывод первых 10 предсказаний
y_pred_sgd = model_sgd.predict(X)
print(y_pred_sgd[:10])

# Cтроим график рассеяния между y и y_pred_sgd
plt.scatter(y, y_pred_sgd)
plt.show()

# Вычисление метрик
mse = mean_squared_error(y, y_pred_sgd) # Ваша строка кода здесь
mae = mean_absolute_error(y, y_pred_sgd) # Ваша строка кода здесь
rmse = np.sqrt(mse) # Ваша строка кода здесь
r2 = r2_score(y, y_pred_sgd) # Ваша строка кода здесь
# Для MAPE исключаем нулевые значения y, чтобы избежать деления на 0
non_zero_idx = y != 0
mape = np.mean(np.abs((y[non_zero_idx] - y_pred_sgd[non_zero_idx]) / y[non_zero_idx])) * 100 # Ваша строка кода здесь

# Вывод результатов
print("MSE:", round(mse, 3))
print("MAE:", round(mae, 3))
print("RMSE:", round(rmse, 3))
print("R²:", round(r2, 3))
print("MAPE:", round(mape, 3))