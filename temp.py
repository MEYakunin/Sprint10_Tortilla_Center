# Обучаем Dummy (всегда предсказывает среднее)
dummy = DummyRegressor(strategy='mean')
dummy.fit(X_train, y_train)

# Предсказания и оценка
y_pred_dummy = dummy.predict(X_val)

# Функция для сбора метрик
def get_metrics(y_true, y_pred, model_name, dataset_name='default'):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    # MAPE (избегаем деления на ноль)
    mape = np.mean(np.abs((y_true - y_pred) / np.maximum(np.abs(y_true), 1e-8))) * 100
    
    return pd.DataFrame({
        'Model': [model_name],
        'Dataset': [dataset_name],
        'MAE': [round(mae, 2)],
        'MSE': [round(mse, 2)],
        'RMSE': [round(rmse, 2)],
        'R2': [round(r2, 2)],
        'MAPE (%)': [round(mape, 2)]
    })

# Сохраняем результаты в DataFrame
results = get_metrics(y_val, y_pred_dummy, 'Dummy (mean)')
print("Базовый уровень (Dummy):")
print(results)

#--------------------------------------------------------------------------------------

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.pipeline import make_pipeline

# Варианты масштабирования
scalers = {
    'no_scale': None,               # без масштабирования
    'standard': StandardScaler(),
    'minmax': MinMaxScaler()
}

# Варианты потерь для SGD
loss_functions = ['squared_error', 'huber', 'epsilon_insensitive']

#--------------------------------------------------------------------------------------


# Список для хранения всех результатов
all_results = [results]  # уже есть Dummy

# Перебираем масштабирования
for scale_name, scaler in scalers.items():
    
    # ---- LinearRegression ----
    if scaler:
        lr_pipe = make_pipeline(scaler, LinearRegression())
    else:
        lr_pipe = LinearRegression()
    
    lr_pipe.fit(X_train, y_train)
    y_pred = lr_pipe.predict(X_val)
    all_results.append(get_metrics(y_val, y_pred, 'LinearRegression', scale_name))
    
    # ---- Ridge (L2) - пробуем разные alpha ----
    for alpha in [0.1, 1.0, 10.0]:
        if scaler:
            ridge_pipe = make_pipeline(scaler, Ridge(alpha=alpha, random_state=42))
        else:
            ridge_pipe = Ridge(alpha=alpha, random_state=42)
        ridge_pipe.fit(X_train, y_train)
        y_pred = ridge_pipe.predict(X_val)
        all_results.append(get_metrics(y_val, y_pred, f'Ridge(alpha={alpha})', scale_name))
    
    # ---- Lasso (L1) ----
    for alpha in [0.01, 0.1, 1.0]:
        if scaler:
            lasso_pipe = make_pipeline(scaler, Lasso(alpha=alpha, random_state=42, max_iter=5000))
        else:
            lasso_pipe = Lasso(alpha=alpha, random_state=42, max_iter=5000)
        lasso_pipe.fit(X_train, y_train)
        y_pred = lasso_pipe.predict(X_val)
        all_results.append(get_metrics(y_val, y_pred, f'Lasso(alpha={alpha})', scale_name))
    
    # ---- SGDRegressor с разными loss ----
    for loss in loss_functions:
        for penalty in ['l2', 'l1', 'elasticnet']:
            if scaler:
                sgd_pipe = make_pipeline(scaler, SGDRegressor(loss=loss, penalty=penalty, 
                                                              max_iter=1000, random_state=42, tol=1e-3))
            else:
                sgd_pipe = SGDRegressor(loss=loss, penalty=penalty, max_iter=1000, 
                                        random_state=42, tol=1e-3)
            sgd_pipe.fit(X_train, y_train)
            y_pred = sgd_pipe.predict(X_val)
            model_name = f'SGD({loss},{penalty})'
            all_results.append(get_metrics(y_val, y_pred, model_name, scale_name))

# Объединяем все результаты
final_results = pd.concat(all_results, ignore_index=True)


#--------------------------------------------------------------------------------------


# Сортируем по MAE (чем меньше, тем лучше)
final_results_sorted = final_results.sort_values('MAE').reset_index(drop=True)

print("\n=== ИТОГОВАЯ ТАБЛИЦА РЕЗУЛЬТАТОВ ===")
pd.set_option('display.float_format', '{:.2f}'.format)
print(final_results_sorted.to_string())

# Проверка, что базовая модель не лучшая
dummy_mae = final_results_sorted[final_results_sorted['Model'] == 'Dummy (mean)']['MAE'].values[0]
better_than_dummy = final_results_sorted[final_results_sorted['MAE'] < dummy_mae]

print(f"\nБазовый MAE = {dummy_mae:.4f}")
print(f"Моделей лучше базовой: {len(better_than_dummy)}")

# Лучшая модель
best_model = final_results_sorted.iloc[0]
print(f"\nЛучшая модель: {best_model['Model']} на датасете {best_model['Dataset']}")
print(f"MAE = {best_model['MAE']:.4f}, R2 = {best_model['R2']:.4f}")