
def zadachki(number):
        if number == 0:
            return'''
            1 -  classification/bank.csv + матрицы + оптимизаторы
            2 - classification/bank.csv + матрицы + дропауты
            3 - classification/bank.csv. + матрицы + несбалансиров
            4 - regression/bike_cnt.csv + BatchNorm
            5 -  regression/gold.csv + оптимизаторы
            6 - images/sign_language.zip + матрицы + PCA
            7 -  images/sign_language.zip + разные conv
            8 - images/eng_handwritten.zip + ранняя остановка
            9 - images/clothes_multi.zip + разные лейблы
            10 - images/chars.zip + аугментация'''
        elif number == 1:
            return '''import cv2
import numpy as np
import matplotlib.pyplot as plt

def enhance_image_all_methods(image_path):
    # Загружаем изображение в градациях серого
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print("Ошибка загрузки изображения")
        return

    methods = {}

    # 1. Инверсия изображения
    inverted = 255 - img
    methods['Inversion'] = inverted

    # 2. Степенное преобразование (гамма-коррекция)
    gamma = 2.2
    gamma_corrected = np.array(255 * (img / 255) ** gamma, dtype='uint8')
    methods[f'Gamma Correction (γ={gamma})'] = gamma_corrected

    # 3. Логарифмическое преобразование
    c = 255 / np.log(1 + np.max(img))
    log_transformed = np.array(c * np.log(1 + img), dtype='uint8')
    methods['Log Transformation'] = log_transformed

    # 4. Эквализация гистограммы
    hist_eq = cv2.equalizeHist(img)
    methods['Histogram Equalization'] = hist_eq

    # 5. Адаптивное выравнивание гистограммы с ограничением контраста (CLAHE)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahe_img = clahe.apply(img)
    methods['CLAHE'] = clahe_img

    # 6. Контрастное растяжение
    min_val, max_val = np.min(img), np.max(img)
    contrast_stretched = np.array(255 * (img - min_val) / (max_val - min_val), dtype='uint8')
    methods['Contrast Stretching'] = contrast_stretched

    # 7. Сигмоидальная коррекция
    gain = 10
    cutoff = 128
    sigmoid = 255 / (1 + np.exp(-gain * ((img - cutoff) / 255.0)))
    sigmoid_corrected = np.array(sigmoid, dtype='uint8')
    methods['Sigmoid Correction'] = sigmoid_corrected

    # Вывод результатов
    n = len(methods)
    plt.figure(figsize=(14, 3 * n))
    
    for i, (title, result) in enumerate(methods.items()):
        # Изображение
        plt.subplot(n, 2, 2*i + 1)
        plt.imshow(result, cmap='gray')
        plt.title(title)
        plt.axis('off')

        # Гистограмма
        plt.subplot(n, 2, 2*i + 2)
        plt.hist(result.ravel(), bins=256, range=(0, 256), color='black')
        plt.title(f'{title} Histogram')
        plt.tight_layout()

    plt.show()

    # Выводы
    print("\nВыводы по методам:")
    print("- Инверсия меняет яркое на тёмное и наоборот.")
    print("- Гамма-коррекция регулирует освещенность: γ>1 делает темнее, γ<1 — светлее.")
    print("- Логарифмическое преобразование усиливает темные детали.")
    print("- Эквализация гистограммы улучшает контраст на изображениях с узким динамическим диапазоном.")
    print("- CLAHE работает локально и полезен при неравномерном освещении.")
    print("- Контрастное растяжение расширяет интенсивности на весь диапазон (0–255).")
    print("- Сигмоидальная коррекция улучшает контраст в средней зоне, сглаживая крайние значения.")

# Пример использования:
# enhance_image_all_methods('path_to_your_image.jpg')
'''
        elif number == 2:
            return '''
            import pandas as pd
import numpy as np

import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay

import matplotlib.pyplot as plt

# Загрузка данных
data = pd.read_csv("/content/bank.csv")

# Разделяем признаки и целевую переменную
X = data.drop("deposit", axis=1)
y = data["deposit"]

# Определяем категориальные и числовые столбцы
categorical_cols = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'poutcome']
numerical_cols = ['age', 'balance', 'day', 'duration', 'campaign', 'pdays', 'previous']

# Предобработка данных: стандартизация числовых и кодирование категориальных признаков
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_cols),
        ('cat', OneHotEncoder(), categorical_cols)
    ]
)

X_processed = preprocessor.fit_transform(X)
y = (y == 'yes').astype(int)  # Преобразуем целевую переменную в бинарный формат (0 и 1)

# Разделение на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.2, random_state=42)

# Преобразование данных в тензоры
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test.values, dtype=torch.float32)

train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)

# Определение модели без Dropout
class ModelWithoutDropout(nn.Module):
    def __init__(self, input_dim):
        super(ModelWithoutDropout, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return self.sigmoid(x)

# Определение модели с Dropout
class ModelWithDropout(nn.Module):
    def __init__(self, input_dim):
        super(ModelWithDropout, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.dropout1 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(64, 32)
        self.dropout2 = nn.Dropout(0.3)
        self.fc3 = nn.Linear(32, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout1(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        return self.sigmoid(x)

# Гиперпараметры
batch_size = 64
epochs = 40
learning_rate = 0.01

# Функция обучения модели
def train_model(model, train_loader, criterion, optimizer):
    loss_history = []
    for epoch in range(epochs):
        epoch_loss = 0
        model.train()
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X).squeeze()
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        loss_history.append(epoch_loss / len(train_loader))
    return loss_history

# Обучение модели без Dropout
model_without_dropout = ModelWithoutDropout(X_train.shape[1])
optimizer_without_dropout = optim.Adam(model_without_dropout.parameters(), lr=learning_rate)
criterion = nn.BCELoss()
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

loss_history_without_dropout = train_model(model_without_dropout, train_loader, criterion, optimizer_without_dropout)

# Обучение модели с Dropout
model_with_dropout = ModelWithDropout(X_train.shape[1])
optimizer_with_dropout = optim.Adam(model_with_dropout.parameters(), lr=learning_rate)

loss_history_with_dropout = train_model(model_with_dropout, train_loader, criterion, optimizer_with_dropout)
# Обучение модели с Dropout
model_with_dropout = ModelWithDropout(X_train.shape[1])
optimizer_with_dropout = optim.Adam(model_with_dropout.parameters(), lr=learning_rate)

loss_history_with_dropout = train_model(model_with_dropout, train_loader, criterion, optimizer_with_dropout)


# Графическое сравнение
plt.figure(figsize=(10, 6))
plt.plot(range(1, epochs + 1), loss_history_without_dropout, label='Without Dropout', marker='o')
plt.plot(range(1, epochs + 1), loss_history_with_dropout, label='With Dropout', marker='s')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training Loss Comparison')
plt.legend()
plt.grid()
plt.show()

# Оценка моделей на тестовом наборе
def evaluate_model(model, X_test, y_test):
    model.eval()
    with torch.no_grad():
        y_pred = model(X_test).squeeze()
        y_pred_classes = (y_pred >= 0.5).int()
    conf_matrix = confusion_matrix(y_test, y_pred_classes)
    print("Classification Report:\n", classification_report(y_test, y_pred_classes))
    disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=['no', 'yes'])
    disp.plot(cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.show()

print("Evaluation for Model Without Dropout:")
evaluate_model(model_without_dropout, X_test, y_test)

print("Evaluation for Model With Dropout:")
evaluate_model(model_with_dropout, X_test, y_test)'''
        elif number == 3:
            return '''
            import pandas as pd
import numpy as np

import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt

# Загрузка данных
data = pd.read_csv("/content/bank.csv")

# Определяем признаки и целевую переменную
X = data.drop("deposit", axis=1)
y = data["deposit"]

# Определяем категориальные и числовые столбцы
categorical_cols = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'poutcome']
numerical_cols = ['age', 'balance', 'day', 'duration', 'campaign', 'pdays', 'previous']

# Предобработка данных
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_cols),
        ('cat', OneHotEncoder(), categorical_cols)
    ]
)

X_processed = preprocessor.fit_transform(X)
y = (y == 'yes').astype(int)  # Преобразуем целевую переменную в бинарный формат (0 и 1)

# Разделение на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.2, random_state=42)

# Преобразование данных в тензоры
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test.values, dtype=torch.float32)

train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)

# Определение модели
class Model(nn.Module):
    def __init__(self, input_dim):
        super(Model, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)  # Один выходной нейрон для бинарной классификации
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return self.sigmoid(x)

# Гиперпараметры
batch_size = 64
epochs = 40
learning_rate = 0.01

# Расчет весов классов
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y),
    y=y
)
class_weights = torch.tensor(class_weights[1], dtype=torch.float32)

# Обучение модели
def train_model(model, optimizer, criterion, train_loader):
    loss_history = []
    for epoch in range(epochs):
        epoch_loss = 0
        model.train()
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X).squeeze()
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        loss_history.append(epoch_loss / len(train_loader))
        print(f'Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss:.4f}')
    return loss_history

# Сравнение обычной и взвешенной функции потерь
models = {
    "Unweighted Loss": (Model(X_train.shape[1]), nn.BCELoss()),
    "Weighted Loss": (Model(X_train.shape[1]), nn.BCELoss(weight=class_weights))
}

results = {}
for name, (model, criterion) in models.items():
    print(f"\n--- Training with {name} ---")
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    loss_history = train_model(model, optimizer, criterion, train_loader)

    # Оценка модели на тестовом множестве
    model.eval()
    with torch.no_grad():
        y_pred = model(X_test).squeeze()
        y_pred_classes = (y_pred >= 0.5).int()


    # Матрица ошибок и отчет классификации
    conf_matrix = confusion_matrix(y_test, y_pred_classes)
    class_report = classification_report(y_test, y_pred_classes, output_dict=True)
    results[name] = {
        "loss_history": loss_history,
        "conf_matrix": conf_matrix,
        "class_report": class_report
    }

    print(f"Confusion Matrix for {name}:")
    print(conf_matrix)
    print(f"\nClassification Report for {name}:\n")
    print(classification_report(y_test, y_pred_classes))

    # Отображение матрицы ошибок
    disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=['no', 'yes'])
    disp.plot(cmap=plt.cm.Blues)
    plt.title(f"Confusion Matrix ({name})")
    plt.show()

# Графическое сравнение функций потерь
plt.figure(figsize=(10, 6))
for name, result in results.items():
    plt.plot(range(1, epochs + 1), result["loss_history"], label=name)
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training Loss Comparison (Weighted vs Unweighted Loss)')
plt.legend()
plt.grid()
plt.show()
'''
        elif number == 4:
            return '''
            import pandas as pd
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

# 1. Load and preprocess data
data = pd.read_csv("/content/bike_cnt.csv")

# Assuming 'cnt' is the target column
y = data['cnt']
X = data.drop(columns=['cnt'])

# Identify categorical and numerical columns
categorical_cols = X.select_dtypes(include=['object']).columns
numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_cols),
        ('cat', OneHotEncoder(), categorical_cols)
    ])

X_processed = preprocessor.fit_transform(X).toarray()
y = y.values.reshape(-1, 1)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.2, random_state=42)

# Convert to PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)

# Create datasets and dataloaders
train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# 2. Define the model
class RegressionModel(nn.Module):
    def __init__(self, input_dim, use_batchnorm=False):
        super(RegressionModel, self).__init__()
        self.use_batchnorm = use_batchnorm
        self.fc1 = nn.Linear(input_dim, 128)
        self.bn1 = nn.BatchNorm1d(128) if use_batchnorm else None
        self.fc2 = nn.Linear(128, 64)
        self.bn2 = nn.BatchNorm1d(64) if use_batchnorm else None
        self.fc3 = nn.Linear(64, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.fc1(x)
        if self.use_batchnorm:
            x = self.bn1(x)
        x = self.relu(x)
        x = self.fc2(x)
        if self.use_batchnorm:
            x = self.bn2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x

# 3. Train the model
def train_model(model, optimizer, criterion, train_loader, epochs=150):
    losses = []
    for epoch in range(epochs):
        epoch_loss = 0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        losses.append(epoch_loss / len(train_loader))
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss / len(train_loader):.4f}")
    return losses

# 4. Evaluate the model
def evaluate_model(model, X, y):
    with torch.no_grad():
        predictions = model(X).numpy()
        return r2_score(y.numpy(), predictions)

# Initialize and train models
input_dim = X_train.shape[1]

# Model without BatchNorm
model_no_bn = RegressionModel(input_dim, use_batchnorm=False)
criterion = nn.MSELoss()
optimizer = optim.Adam(model_no_bn.parameters(), lr=0.001)
losses_no_bn = train_model(model_no_bn, optimizer, criterion, train_loader)

# Model with BatchNorm
model_with_bn = RegressionModel(input_dim, use_batchnorm=True)
optimizer = optim.Adam(model_with_bn.parameters(), lr=0.001)
losses_with_bn = train_model(model_with_bn, optimizer, criterion, train_loader)

# Evaluate on test set
r2_no_bn = evaluate_model(model_no_bn, X_test, y_test)
r2_with_bn = evaluate_model(model_with_bn, X_test, y_test)

# 5. Plot results
plt.figure(figsize=(10, 5))
plt.plot(losses_no_bn, label='Without BatchNorm')
plt.plot(losses_with_bn, label='With BatchNorm')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss with/without BatchNorm')
plt.legend()
plt.show()


print(f"R^2 without BatchNorm on test set: {r2_no_bn:.4f}")
print(f"R^2 with BatchNorm on test set: {r2_with_bn:.4f}")
'''
        elif number == 5:
            return '''
            import pandas as pd
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

# 1. Load data
data = pd.read_csv("gold.csv")

# Define target and feature columns
target_columns = ['Gold_T-7', 'Gold_T-14', 'Gold_T-22', 'Gold_T+22']
X = data.drop(columns=target_columns)
y = data[target_columns]

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler_X = StandardScaler()
X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)

# Convert data to PyTorch tensors
X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32)

# Create datasets and dataloaders
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# 2. Define the model
class MultiTargetRegressionModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MultiTargetRegressionModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)  # Reduced number of neurons
        self.fc2 = nn.Linear(64, 32)  # Reduced number of neurons
        self.fc3 = nn.Linear(32, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Initialize the model
input_dim = X_train_tensor.shape[1]
output_dim = y_train_tensor.shape[1]
model = MultiTargetRegressionModel(input_dim, output_dim)

# Loss function and optimizers
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)  # Reduced learning rate

# 3. Train the model
def train_model(model, optimizer, criterion, train_loader, epochs=100):
    losses = []
    for epoch in range(epochs):
        epoch_loss = 0
        model.train()
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        losses.append(epoch_loss / len(train_loader))
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss / len(train_loader):.4f}")
    return losses

# 4. Evaluate the model
def evaluate_model(model, X, y):
    model.eval()
    with torch.no_grad():
        predictions = model(X).numpy()
        return r2_score(y, predictions, multioutput='uniform_average')

# Train the model
losses = train_model(model, optimizer, criterion, train_loader)

# Evaluate on test set
r2 = evaluate_model(model, X_test_tensor, y_test_tensor.numpy())
print(f"R^2 on test set: {r2:.4f}")

# 5. Plot training losses
plt.figure(figsize=(10, 5))
plt.plot(losses, label='Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.legend()
plt.show()
'''
        elif number == 6:
            return '''
            import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms, datasets
from sklearn.metrics import f1_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as np
from zipfile import ZipFile

# Разархивация датасета
zip_path = "images/sign_language.zip"
data_dir = "sign_language"
if not os.path.exists(data_dir):
    with ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall("images")

# 1. Загрузка данных
transform = transforms.Compose([
    transforms.Resize((64, 64)),  # Приводим изображения к одному размеру
    transforms.ToTensor(),       # Преобразуем в тензоры
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Нормализация
])

full_dataset = datasets.ImageFolder(data_dir, transform=transform)
num_classes = len(full_dataset.classes)

# 2. Разделение данных
train_size = int(0.7 * len(full_dataset))
test_size = len(full_dataset) - train_size
train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64)

# 3. Сверточная нейронная сеть
class CNN(nn.Module):
    def __init__(self, num_classes):
        super(CNN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * (64 // 8)**2, 128),  # 64 // 8 — это уменьшение размера после трех слоев MaxPool2d
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)
        return x

# 4. Обучение модели
def train_model(model, train_loader, num_epochs=10):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    train_losses = []
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        for images, labels in train_loader:

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        train_losses.append(epoch_loss / len(train_loader))
        print(f"Epoch {epoch + 1}: Loss = {epoch_loss / len(train_loader):.4f}")

    return train_losses

# 5. Оценка модели
def evaluate_model(model, test_loader):
    model.eval()
    y_true, y_pred = [], []

    with torch.no_grad():
        for images, labels in test_loader:

            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    micro_f1 = f1_score(y_true, y_pred, average='micro')
    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred, target_names=full_dataset.classes)
    return micro_f1, report, cm

# 6. Визуализация скрытых представлений
def visualize_pca(model, test_loader):
    model.eval()
    features, labels = [], []

    with torch.no_grad():
        for images, label in test_loader:

            output = model.conv(images).view(images.size(0), -1)
            features.append(output.cpu().numpy())
            labels.extend(label.numpy())

    features = np.concatenate(features)
    pca = PCA(n_components=2)
    reduced_features = pca.fit_transform(features)

    plt.figure(figsize=(10, 8))
    for class_idx in range(num_classes):
        idx = np.array(labels) == class_idx
        plt.scatter(reduced_features[idx, 0], reduced_features[idx, 1], label=full_dataset.classes[class_idx], alpha=0.7)

    plt.title("PCA of Hidden Representations")
    plt.xlabel("PCA 1")
    plt.ylabel("PCA 2")
    plt.legend()
    plt.show()

# 7. Запуск эксперимента
model = CNN(num_classes=num_classes)

# Обучение модели
train_losses = train_model(model, train_loader)

# График функции потерь
plt.plot(train_losses, label="Training Loss")
plt.title("Training Loss per Epoch")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.show()

# Оценка качества
micro_f1, report, cm = evaluate_model(model, test_loader)
print(report)

# Матрица ошибок
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=full_dataset.classes, yticklabels=full_dataset.classes)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# Визуализация PCA
visualize_pca(model, test_loader)
'''
        elif number == 7:
            return '''
            import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms, datasets
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
from zipfile import ZipFile

# Разархивация датасета
zip_path = "images/sign_language.zip"
data_dir = "sign_language"
if not os.path.exists(data_dir):
    with ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall("images")


# 1. Загрузка данных и предобработка
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

dataset = datasets.ImageFolder(data_dir, transform=transform)
num_classes = len(dataset.classes)

# Разделение данных на обучающее и тестовое множества
train_size = int(0.7 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64)

# 2. Сверточная нейронная сеть
class CNN(nn.Module):
    def __init__(self, num_blocks, num_classes):
        super(CNN, self).__init__()
        layers = []
        in_channels = 3

        for _ in range(num_blocks):
            layers.extend([
                nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2)
            ])
            in_channels = 32

        self.conv = nn.Sequential(*layers)
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * (64 // (2 ** num_blocks))**2, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)
        return x

# 3. Функция обучения
def train_model(model, train_loader, num_epochs=10):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(num_epochs):
        model.train()
        for images, labels in train_loader:

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

# 4. Функция оценки
def evaluate_model(model, test_loader):
    model.eval()
    y_true, y_pred = [], []

    with torch.no_grad():
        for images, labels in test_loader:

            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    micro_f1 = f1_score(y_true, y_pred, average='micro')
    return micro_f1

# 5. Проведение экспериментов
results = []

for num_blocks in range(1, 5):  # Перебираем количество сверточных блоков
    print(f"Training with {num_blocks} convolutional blocks...")
    model = CNN(num_blocks, num_classes)
    train_model(model, train_loader)
    micro_f1 = evaluate_model(model, test_loader)
    results.append((num_blocks, micro_f1))
    print(f"Micro F1 for {num_blocks} blocks: {micro_f1:.4f}")

# 6. Визуализация результатов
blocks, scores = zip(*results)
plt.plot(blocks, scores, marker='o', label="Micro F1 Score")
plt.title("Micro F1 vs Number of Convolutional Blocks")
plt.xlabel("Number of Convolutional Blocks")
plt.ylabel("Micro F1 Score")
plt.grid()
plt.legend()
plt.show()
'''
        elif number == 8:
            return '''
            import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, Subset
from torchvision import transforms, datasets
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
from tqdm import tqdm
from zipfile import ZipFile

# Разархивация датасета
zip_path = "images/eng_handwritten.zip"
data_dir = "eng_handwritten"
if not os.path.exists(data_dir):
    with ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall("images")

# 1. Загрузка данных и предобработка
transform = transforms.Compose([
    transforms.CenterCrop((32, 32)),  # Обрезка центральной области
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

dataset = datasets.ImageFolder(data_dir, transform=transform)
num_classes = len(dataset.classes)

# Разделение данных: 70% - обучающая выборка, 15% - валидационная, 15% - тестовая
train_size = int(0.7 * len(dataset))
val_size = int(0.15 * len(dataset))
test_size = len(dataset) - train_size - val_size
train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=128)
test_loader = DataLoader(test_dataset, batch_size=128)

# 2. Сверточная нейронная сеть
class CNN(nn.Module):
    def __init__(self, num_classes):
        super(CNN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * (32 // 8)**2, 128),  # 128 // 8 - уменьшение после трех MaxPool2d
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)
        return x

# 3. Обучение модели с ранней остановкой
def train_model_with_early_stopping(model, train_loader, val_loader, patience=5, num_epochs=50):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    best_f1 = 0.0
    best_model_state = None
    patience_counter = 0

    for epoch in tqdm(range(num_epochs)):
        # Обучение
        model.train()
        for images, labels in train_loader:

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        # Валидация
        model.eval()
        y_true, y_pred = [], []
        with torch.no_grad():
            for images, labels in val_loader:
                outputs = model(images)
                _, preds = torch.max(outputs, 1)
                y_true.extend(labels.cpu().numpy())
                y_pred.extend(preds.cpu().numpy())

        val_f1 = f1_score(y_true, y_pred, average='micro')
        print(f"Epoch {epoch + 1}: Validation Micro F1 = {val_f1:.4f}")

        # Проверка ранней остановки
        if val_f1 > best_f1:
            best_f1 = val_f1
            best_model_state = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered")
                break

    # Возврат лучшей модели
    model.load_state_dict(best_model_state)
    return model

# 4. Оценка модели
def evaluate_model(model, test_loader):
    model.eval()
    y_true, y_pred = [], []

    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            y_true.extend(labels.numpy())
            y_pred.extend(preds.numpy())

    micro_f1 = f1_score(y_true, y_pred, average='micro')
    return micro_f1

# 5. Запуск эксперимента
model = CNN(num_classes)
model = train_model_with_early_stopping(model, train_loader, val_loader, patience=3, num_epochs=20)
test_f1 = evaluate_model(model, test_loader)

print(f"Final Test Micro F1: {test_f1:.4f}")
'''
        elif number == 9:
            return '''
            import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from zipfile import ZipFile

# Разархивация датасета
zip_path = "clothes_multi.zip"
data_dir = "clothes_multi"
if not os.path.exists(data_dir):
    with ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall("images")

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

dataset = datasets.ImageFolder(root=data_dir, transform=transform)

class_names = dataset.classes
color_to_idx = {}
item_to_idx = {}
colors = set()
items = set()

for class_name in class_names:
    color, item = class_name.split('_')
    colors.add(color)
    items.add(item)

color_to_idx = {color: idx for idx, color in enumerate(sorted(colors))}
item_to_idx = {item: idx for idx, item in enumerate(sorted(items))}

def get_color_and_item_labels(target):
    class_name = class_names[target]
    color, item = class_name.split('_')
    return color_to_idx[color], item_to_idx[item]

class MultiLabelDataset(torch.utils.data.Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, target = self.dataset[idx]
        color_label, item_label = get_color_and_item_labels(target)
        return img, torch.tensor([color_label, item_label])

multi_label_dataset = MultiLabelDataset(dataset)

train_size = int(0.8 * len(multi_label_dataset))
test_size = len(multi_label_dataset) - train_size
train_dataset, test_dataset = random_split(multi_label_dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


# Модель для многозадачной классификации
class MultiTaskModel(nn.Module):
    def __init__(self, num_color_classes, num_clothing_classes):
        super(MultiTaskModel, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.fc_color = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 32 * 32, 128),
            nn.ReLU(),
            nn.Linear(128, num_color_classes)
        )
        self.fc_clothing = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 32 * 32, 128),
            nn.ReLU(),
            nn.Linear(128, num_clothing_classes)
        )

    def forward(self, x):
        features = self.conv(x)
        color_output = self.fc_color(features)
        clothing_output = self.fc_clothing(features)
        return color_output, clothing_output

# Функция вычисления micro F1
def calculate_micro_f1(y_true, y_pred):
    return f1_score(y_true, y_pred, average='micro')

# Логика обучения модели
def train_model(model, train_loader, num_epochs):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in tqdm(range(num_epochs)):
        model.train()
        train_loss = 0
        for imgs, labels in train_loader:
            color_labels, item_labels = labels[:, 0], labels[:, 1]

            optimizer.zero_grad()
            color_out, item_out = model(imgs)

            loss = criterion(color_out, color_labels) + criterion(item_out, item_labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}")

# Функция тестирования
def evaluate_model(model, data_loader):
    model.eval()
    true_color, true_item = [], []
    pred_color, pred_item = [], []

    with torch.no_grad():
        for imgs, labels in data_loader:
            color_labels, item_labels = labels[:, 0], labels[:, 1]

            color_out, item_out = model(imgs)
            true_color.extend(color_labels.numpy())
            true_item.extend(item_labels.numpy())

            pred_color.extend(color_out.argmax(1).numpy())
            pred_item.extend(item_out.argmax(1).numpy())

    f1_color = calculate_micro_f1(true_color, pred_color)
    f1_item = calculate_micro_f1(true_item, pred_item)
    micro_f1 = (f1_color + f1_item) / 2
    return micro_f1

# Основной процесс
num_colors = len(color_to_idx)
num_items = len(item_to_idx)
model = MultiTaskModel(num_colors, num_items)

# Обучение модели
train_model(model, train_loader, num_epochs=10)

# Оценка модели
train_f1 = evaluate_model(model, train_loader)
test_f1 = evaluate_model(model, test_loader)

print(f"F1 на обучающем множестве: {train_f1:.4f}")
print(f"F1 на тестовом множестве: {test_f1:.4f}")
'''
        elif number == 10:
            return '''
            import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms, datasets
from sklearn.metrics import f1_score
from zipfile import ZipFile

# Разархивация датасета
zip_path = "images/chars.zip"
data_dir = "chars"
if not os.path.exists(data_dir):
    with ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall("images")

# 1. Загрузка и предобработка данных
basic_transform = transforms.Compose([
    transforms.Resize((64, 64)),  # Изменяем размер изображений
    transforms.ToTensor(),  # Преобразуем в тензор
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Нормализация
])

augment_transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

dataset = datasets.ImageFolder(data_dir, transform=basic_transform)
num_classes = len(dataset.classes)

# Разделение данных на обучающее (70%) и тестовое (30%) множество
train_size = int(0.7 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

# Датасет с аугментацией
augmented_train_dataset = datasets.ImageFolder(data_dir, transform=augment_transform)
augmented_train_dataset = torch.utils.data.Subset(augmented_train_dataset, train_dataset.indices)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64)
augmented_train_loader = DataLoader(augmented_train_dataset, batch_size=64, shuffle=True)

# 2. Сверточная нейронная сеть
class CNN(nn.Module):
    def __init__(self, num_classes):
        super(CNN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * (64 // 8)**2, 128),  # Учитываем уменьшение размера после MaxPool2d
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)
        return x

# 3. Функция обучения модели
def train_model(model, train_loader, num_epochs=10):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(num_epochs):
        model.train()
        for images, labels in train_loader:
            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch + 1}/{num_epochs} complete.")

    return model

# 4. Функция оценки модели
def evaluate_model(model, test_loader):
    model.eval()
    y_true, y_pred = [], []

    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            y_true.extend(labels.numpy())
            y_pred.extend(preds.numpy())

    micro_f1 = f1_score(y_true, y_pred, average='micro')
    return micro_f1

# 5. Обучение и оценка
# Базовый набор данных
model_basic = CNN(num_classes)
model_basic = train_model(model_basic, train_loader, num_epochs=10)
f1_basic = evaluate_model(model_basic, test_loader)
print(f"Micro F1 Score (basic dataset): {f1_basic:.4f}")

# Расширенный набор данных (с аугментацией)
model_augmented = CNN(num_classes)
model_augmented = train_model(model_augmented, augmented_train_loader, num_epochs=20)
f1_augmented = evaluate_model(model_augmented, test_loader)
print(f"Micro F1 Score (augmented dataset): {f1_augmented:.4f}")
'''