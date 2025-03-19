import os, time, numpy as np, tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, ReduceLROnPlateau
from sklearn.preprocessing import StandardScaler

# Для файла с прогнозами forecast.txt
MONTHS = ["Январь", "Февраль", "Март", "Апрель", "Май", "Июнь",
          "Июль", "Август", "Сентябрь", "Октябрь", "Ноябрь", "Декабрь"]

# Загружаем данные
def load_data(filepath):
    data = np.loadtxt(filepath)
    if data.ndim != 2 or data.shape[1] != 12:
        raise ValueError("Каждая строка файла должна содержать ровно 12 чисел! Ну, серьезно.")
    return data.astype(np.float32)

# Немного магии нормализации – чтобы данные не ругались.
def normalize_data(data):
    scaler = StandardScaler()
    return scaler.fit_transform(data), scaler

# Составляем модель добавляем три слоя, приправляем Dropout и BatchNorm.
def build_model(input_dim=12):
    model = Sequential([
        Dense(128, activation='relu', input_shape=(input_dim,)),
        BatchNormalization(), Dropout(0.2),  # Немного стабилизации
        Dense(128, activation='relu'),
        BatchNormalization(), Dropout(0.2),
        Dense(128, activation='relu'),
        BatchNormalization(), Dropout(0.2),
        Dense(input_dim)  # Выходной слой
    ])
    # Оптимизатор Adam с темпом 0.001
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  loss='mse', metrics=['mae'])
    return model

# Создаём датасет, чтобы тензоры не скучали
def create_dataset(X, y, batch_size):
    return tf.data.Dataset.from_tensor_slices((X, y))\
              .cache().shuffle(len(X)).batch(batch_size)\
              .prefetch(tf.data.AUTOTUNE)

# Записываем прогноз в текстовый файл forecast.txt
def save_forecast_as_text(prediction, filepath):
    with open(filepath, "w", encoding="utf-8") as f:
        for i, temp in enumerate(prediction):
            month = MONTHS[i] if i < len(MONTHS) else f"Месяц {i+1}"
            f.write(f"{month}   {temp:.2f} C°\n")
    print(f"Прогноз сохранён в файл: {filepath}  # Вот так просто.")

# Главная функция обучения или загрузки модели
def train_or_load(data_file, output_file, model_file="model.h5", load_existing=False,
                  epochs=300, batch_size=8, logs_dir="logs"):
    data = load_data(data_file)
    if len(data) < 2:
        raise ValueError("Недостаточно данных (минимум 2 года). Да, так бывает.")
    data, scaler = normalize_data(data)
    X, y = data[:-1], data[1:]
    split = int(0.8 * len(X))
    train_ds = create_dataset(X[:split], y[:split], batch_size)
    test_ds  = create_dataset(X[split:], y[split:], batch_size)

    # Проверка на GPU – надеемся, они у нас есть.
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print("Обнаружены GPU:", gpus)
    else:
        print("GPU не обнаружены – используем добротный CPU.")

    if load_existing and os.path.exists(model_file):
        model = load_model(model_file)
        print(f"Модель загружена из {model_file}")
        history = None
    else:
        model = build_model()
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=50, restore_best_weights=True, verbose=1),
            ModelCheckpoint(model_file, monitor='val_loss', save_best_only=True, verbose=1),
            TensorBoard(log_dir=logs_dir, histogram_freq=1),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, verbose=1)
        ]
        start = time.time()
        history = model.fit(train_ds, epochs=epochs, validation_data=test_ds, callbacks=callbacks, verbose=1)
        print(f"Обучение завершено за {time.time()-start:.2f} секунд")
    loss, mae = model.evaluate(test_ds, verbose=0)
    print(f"Результаты: Потери (MSE): {loss:.4f}, MAE: {mae:.4f}")

    raw_pred = model.predict(data[-1].reshape(1, 12))[0]
    prediction = scaler.inverse_transform(raw_pred.reshape(1, -1))[0]
    print("Прогноз температур на следующий год:", prediction)
    save_forecast_as_text(prediction, output_file)
    return history, model, prediction, scaler
