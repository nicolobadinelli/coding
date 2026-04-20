
def cv2(number):
        if number == 0:
            return'''
            1007 - что-то с луной(размывание границ и тд)
            1005 - убрать фон с картинки
            1002 - Убрать шум, выделить границы и усилить их(соль и перец)
            1012 - На встроенном датасете FashionMNIST обучить классификацию без сверток
            1011 - Используя детекцию людей (например YOLO) определить кто из них находится 
            слишком близко друг от друга (по какому-то установленному порогу), например для контроля осблюдения дистанции в общественных местах. 
            Выводить предупреждение, если количество собравшихся больше двух. Продемонстрировать на изображении этих людей рамками красного цвета. Изображение 1.jpg
            101 - сегментация по порогу
            102 - всякое с гистограмами
            103 - морфологические операции
            104 - все фильтры(+ручные)
            105 - методы выделения границ
            106 - сапостовление шаблонов
            107 - Классификация
            108 - Классификация VGG16
            109 - SSD детектор с возможностью менять порог уверенности и указывать 
            классы объектов и сохранением в json class, score, bbox + вывод рандомных 5 с боксами
            110 - детекция Faster-Rnn
            111 - Детекция Yolo
            112 - семантическая сегментация ResNet
            113 - инстанс сегментация Yolo
            114 - Сегментация MaskRnn с разными цветами для каждого объекта
            1 -  улучшение изображения(инверсия, гамма, логарифмическое, степенное...)
            2 - афинные преобразование
            5 -  преобразования фурье
            6 - трекинг(deepsort/sort)
            7 -  заполнение отверстий
            8 - Сегментация по rgb-каналам по гистограмам
            9 - Кластеризация kmeans
            10 - Кастомный датасет
            11 - Ручный фильтры медиана, мин макс
            13 - Классификация VGG16
            14 - SSD детектор с возможностью менять порог уверенности и указывать 
            классы объектов и сохранением в json class, score, bbox + вывод рандомных 5 с боксами
            15 - Yolo на видео с bbox
            16 - Unet
            17 - Mask R-CNN обнаруживает ток опред.классы с опред. цветами, отрисовывает bbox и считает колво
            18 - адаптация под видео + сохранение бинарных масок для каждого объекта (PNG) и аннотаций в COCO-формате (JSON)
            19 - ByteTrack + вывод его со всякими модификациями в определенном окне, считается количество объектов и тд
            20 - deepsort с подсчетом объектов
            21 - алгоритма Sort( + замена на YOLO) id объекта и его класс. подсчет и вывод в консоль: 
            Общего числа уникальных объектов за видео. 
            Среднего времени жизни трека (в кадрах). 
            Числа ложных срабатываний'''
        elif number == 1007:
            return '''
import numpy as np
import cv2
import matplotlib.pyplot as plt

image = cv2.imread('moon.png', cv2.IMREAD_GRAYSCALE)
image

eq_img = cv2.equalizeHist(image)
eq_img

from scipy.ndimage import gaussian_laplace
from PIL import Image

im1 = gaussian_laplace(image, sigma=1)
im2 = gaussian_laplace(image, sigma=0.5)


Image.fromarray(im1)

Image.fromarray(im2)

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
clahe_img = clahe.apply(image)

laplacian = cv2.Laplacian(clahe_img, cv2.CV_64F)
laplacian_abs = cv2.convertScaleAbs(laplacian)

enhanced_img = cv2.addWeighted(clahe_img, 0.8, laplacian_abs, 0.2, 0)

enhanced_img

clahe_img'''
        elif number == 1005:
            return '''
import cv2
import numpy as np
import matplotlib.pyplot as plt
image_path = 'writing_on_the_wall.jpeg'
image = cv2.imread(image_path)

plt.figure(figsize=(12,8))
plt.imshow(image)
plt.axis(False);

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 15, 10) # адаптивная бинаризация (выделение темного текста на светлом фоне)

kernel_large = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)) # открытие и закрытие (удаление мелких шумов и артефактов фона)
morph = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_large, iterations=1)
morph = cv2.morphologyEx(morph, cv2.MORPH_CLOSE, kernel_large, iterations=1)

contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

min_area = 50 
mask_filtered = np.zeros_like(gray)
for cnt in contours: # фильтрация контуров по площади (крупные контуры - буквы, вероятнее всего )
    area = cv2.contourArea(cnt)
    if area > min_area:
        cv2.drawContours(mask_filtered, [cnt], -1, 255, thickness=cv2.FILLED)

result_filtered = cv2.bitwise_and(image, image, mask=mask_filtered) # маскирование 
white_bg = np.full_like(image, 255)
final_clean = np.where(mask_filtered[:, :, None] == 255, result_filtered, white_bg)

final_clean_rgb = cv2.cvtColor(final_clean, cv2.COLOR_BGR2RGB)
plt.figure(figsize=(12,8))
plt.imshow(final_clean_rgb)
plt.axis(False);'''
        elif number == 1002:
            return '''
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
image_path = "/content/salt_and_papper.png"
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
# Медианная фильтрация для удаления импульсного шума
denoised = cv2.medianBlur(image, 3)

# Выделение границ с помощью оператора Собеля
sobel_x = cv2.Sobel(denoised, cv2.CV_64F, 1, 0, ksize=3)
sobel_y = cv2.Sobel(denoised, cv2.CV_64F, 0, 1, ksize=3)
sobel_edges = cv2.magnitude(sobel_x, sobel_y)
sobel_edges = np.uint8(np.clip(sobel_edges, 0, 255))

# Усиление границ: наложение границ на исходное изображение
edges = cv2.Canny(denoised, 50, 150)
edges_dilated = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
enhanced_edges = cv2.bitwise_or(denoised, edges_dilated)

fig, axs = plt.subplots(1, 4, figsize=(16, 4))
axs[0].imshow(image, cmap='gray')
axs[0].set_title('Изначальное изображение')
axs[1].imshow(denoised, cmap='gray')
axs[1].set_title('Удаление шума медианный фильтр')
axs[2].imshow(sobel_edges, cmap='gray')
axs[2].set_title('Границы метод Собеля')
axs[3].imshow(enhanced_edges, cmap='gray')
axs[3].set_title('Усиленные границы')
# axs[3].axis('off')
for ax in axs:
    ax.axis('off')
plt.tight_layout()
plt.show()'''
        elif number == 1012:
            return '''
import torch
import torchvision as tv
import torch.nn as nn
import torchvision.transforms as T

import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from tqdm.notebook import tqdm
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Device: {device}")
transform = T.Compose([
    T.ToTensor(),
    T.Normalize((0.5,), (0.5,)),
])

train_dataset = tv.datasets.FashionMNIST(root='mnist', download=True, train=True, transform=transform)
test_dataset = tv.datasets.FashionMNIST(root='mnist', download=True, train=False, transform=transform)

batch_size = 128
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

classes = train_dataset.classes

classes

plt.figure(figsize=(16, 8))

n = 10

for i in range(n):
    img, label = train_dataset[i]

    plt.subplot(1, n, i+1)

    plt.imshow(img.squeeze(), cmap='gray')

    plt.title(classes[label])
    plt.axis('off')

plt.show()

class FCModel(nn.Module):
    def __init__(self, inp_dim, out_dim):
        super().__init__()
        self.flatten = nn.Flatten()
        self.net = nn.Sequential(
            nn.Linear(inp_dim, 256),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, out_dim)
        )

    def forward(self, x):
        x = self.flatten(x)
        x = self.net(x)
        return x

model = FCModel(28 * 28, len(classes)).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

def train_epoch(model, dataloader, criterion, optimizer):
    model.train()

    epoch_loss = 0
    correct = 0
    total = len(dataloader) * dataloader.batch_size

    for imgs, labels in dataloader:
        imgs, labels = imgs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(imgs)

        loss = criterion(outputs, labels)
        loss.backward()

        optimizer.step()

        epoch_loss += loss.item()

        p, predicted = torch.max(outputs, 1)

        correct += (predicted == labels).sum().item()

    return epoch_loss / total, correct / total

def eval_epoch(model, dataloader, criterion):
    model.eval()

    epoch_loss = 0.0
    correct = 0
    total = len(dataloader) * dataloader.batch_size

    with torch.no_grad():
        for imgs, labels in dataloader:
            imgs, labels = imgs.to(device), labels.to(device)

            outputs = model(imgs)

            loss = criterion(outputs, labels)
            epoch_loss += loss.item()

            p, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()

    return epoch_loss / total, correct / total
    
num_epochs = 10

train_losses, train_accs = [], []
test_losses, test_accs = [], []

for epoch in tqdm(range(num_epochs)):
    train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer)
    test_loss, test_acc = eval_epoch(model, test_loader, criterion)

    train_losses.append(train_loss)
    train_accs.append(train_acc)
    test_losses.append(test_loss)
    test_accs.append(test_acc)

    print(f"Epoch {epoch+1:>3} | Train loss: {train_loss:.6f} | Train Accuracy: {train_acc:.3f} | Test loss: {test_loss:.6f} | Test Accuracy: {test_acc:.3f}")
t = range(1, num_epochs + 1)

plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)

plt.plot(t, train_losses, label='Train loss')
plt.plot(t, test_losses, label='Test loss')
plt.xlabel('Epoch')
plt.title('Loss')

plt.grid(alpha=0.5, linestyle=':')
plt.legend()

plt.subplot(1, 2, 2)

plt.plot(t, train_accs, label='Train accuracy')
plt.plot(t, test_accs, label='Test accuracy')
plt.xlabel('Epoch')
plt.title('Accuracy')

plt.grid(alpha=0.5, linestyle=':')
plt.legend()

plt.show()
img, label = test_dataset[0]
model.eval()

with torch.no_grad():
    img_input = img.unsqueeze(0).to(device)
    output = model(img_input)
    p, pred = torch.max(output, 1)

plt.imshow(img.squeeze(), cmap='gray')

color = 'green' if pred == label else 'red'

plt.title(f"True: {classes[label]} | Pred: {classes[pred]}", color=color)
plt.axis('off')
plt.show()

model.eval()

plt.figure(figsize=(18, 6))

n = 30

with torch.no_grad():
    for i in range(n):
        img, label = test_dataset[i]
        img_input = img.unsqueeze(0).to(device)

        output = model(img_input)
        pred = torch.argmax(output, dim=1).item()

        plt.subplot(3, n // 3, i + 1)
        plt.imshow(img.squeeze(), cmap='gray')

        color = 'green' if pred == label else 'red'

        plt.title(f"True: {classes[label]}\nPred: {classes[pred]}", color=color)

        plt.axis('off')

plt.show()'''
        elif number == 1011:
            return '''
!pip install ultralytics
%matplotlib inline
import cv2
import numpy as np
from ultralytics import YOLO
from matplotlib import pyplot as plt
from itertools import combinations
# загружаем изображение
image = cv2.imread('./1.jpg')
plt.imshow(image)
plt.show()
model = YOLO('yolov8s.pt')
results = model(image)[0]
results
human_boxes = []
for result in results.boxes:
    if int(result.cls) == 0:  # class 0 это человек в COCO
        x1, y1, x2, y2 = map(int, result.xyxy[0])
        conf = float(result.conf)  # Точность
        center = ((x1 + x2) // 2, (y1 + y2) // 2)
        human_boxes.append(((x1, y1, x2, y2), center, conf))

human_boxes
# расстояния между людьми
threshold = 50  # порог дистанции в пикселях
too_close = set()

for (i, (boxA, centerA)), (j, (boxB, centerB)) in combinations(enumerate(human_boxes), 2):
    dist = np.linalg.norm(np.array(centerA) - np.array(centerB))
    if dist < threshold:
        too_close.add(i)
        too_close.add(j)
too_close
# визуализация
for idx, (box, _, conf) in enumerate(human_boxes):
    color = (255, 0, 0) if idx in too_close else (0, 255, 0)
    x1, y1, x2, y2 = box
    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

    # выводим точности
    label = f"{conf:.2f}"
    cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, color, 2)

# WARNING
if len(too_close) > 2:
    cv2.putText(image, "WARNING: Group Detected!", (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 0), 3)

# сохранение и вывод результата
cv2.imwrite("output.jpg", image)
plt.imshow(image)
plt.show()'''
        elif number == 114:
            return '''
import torch
import torchvision
from torchvision.transforms import functional as F
from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt
from google.colab.patches import cv2_imshow
import random

# 1. Загрузка предобученной модели
model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)
model.eval()

image_path = '/content/test2.jpg'
try:
    img_pil = Image.open(image_path).convert("RGB")
except FileNotFoundError:
    print(f"Ошибка: Изображение не найдено по пути {image_path}")
    exit()

img_tensor = F.to_tensor(img_pil)
img_tensor = img_tensor.to(device)

# 3. Выполнение предсказания
with torch.no_grad():
    prediction = model([img_tensor])

# 4. Обработка и визуализация результатов
results = prediction[0]

boxes = results['boxes'].cpu().numpy()
labels = results['labels'].cpu().numpy()
scores = results['scores'].cpu().numpy()
masks = results['masks'].cpu().numpy() # Маски имеют размер (N, 1, H, W)

# Преобразуем исходное изображение обратно в формат OpenCV (BGR) для визуализации
img_cv2 = np.array(img_pil)[:, :, ::-1].copy() # PIL Image (RGB) -> NumPy array (RGB) -> BGR for cv2

detection_threshold = 0.7
COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]
person_class_id = COCO_INSTANCE_CATEGORY_NAMES.index('person')

# Функция для генерации случайного цвета
def get_random_color():
    r = random.randint(0, 255)
    g = random.randint(0, 255)
    b = random.randint(0, 255)
    return (b, g, r) # Возвращаем в формате BGR для OpenCV

# Итерируем по обнаруженным объектам
if len(boxes) > 0:
    output_image = img_cv2.copy()

    for i in range(len(boxes)):
        score = scores[i]
        label = labels[i]

        if label == person_class_id and score > detection_threshold:
            box = boxes[i].astype(int)
            mask = masks[i, 0] # Маски имеют размер (N, 1, H, W), берем 0-й канал

            # Генерируем уникальный цвет для текущего объекта
            color = get_random_color()
            # Рисуем bounding box выбранным цветом
            cv2.rectangle(output_image, (box[0], box[1]), (box[2], box[3]), color, 2)
            # Накладываем маску
            mask_scaled = (mask > 0.5).astype(np.uint8) * 255 # Бинаризуем маску (0 или 1), затем масштабируем до 0/255

            # Создаем цветную маску
            mask_colored = np.zeros_like(output_image)
            mask_colored[:, :, 0] = mask_scaled * (color[0] / 255.0) # B
            mask_colored[:, :, 1] = mask_scaled * (color[1] / 255.0) # G
            mask_colored[:, :, 2] = mask_scaled * (color[2] / 255.0) # R
            mask_colored = mask_colored.astype(np.uint8)

            # Добавляем маску к изображению с полупрозрачностью
            # Используем output_image для последовательного наложения
            output_image = cv2.addWeighted(output_image, 1, mask_colored, 0.5, 0)

            # Добавляем текст с меткой и уверенностью
            text = f"Person: {score:.2f}"
            cv2.putText(output_image, text, (box[0], box[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
else:
    print(f"Объекты не обнаружены с порогом уверенности {detection_threshold}")

# 5. Отображение результата
print(f"--- Предсказание Mask R-CNN (Pretrained COCO) для {image_path} ---")
cv2_imshow(output_image) # Используем output_image для отображения'''
        elif number == 113:
            return '''
from ultralytics import YOLO
import cv2
import numpy as np
import matplotlib.pyplot as plt
from google.colab.patches import cv2_imshow
import random

def get_random_color():
    r = random.randint(50, 255)
    g = random.randint(50, 255)
    b = random.randint(50, 255)
    return (b, g, r) # Возвращаем в формате BGR для OpenCV

# --- Загружаем предобученную модель YOLOv8-segmentation (nano версия) ---
model = YOLO("yolov8n-seg.pt")

image_path = "test2.jpg"
person_id = [k for k, v in model.names.items() if v == 'person'][0] if 'person' in model.names.values() else None

# --- Выполняем предсказание ---
results = model(
    image_path,
    classes=[person_id], # Обнаружить только людей
    conf=0.5,             # Минимальная уверенность 50%
    iou=0.6,              # Порог NMS IoU 60%
    imgsz=1280,           # Изменить размер изображения до 1280x1280
    device='cpu',         # Использовать CPU для инференса
    verbose=True          # Показывать подробный вывод
)
# Получаем объект Results для первого изображения
result = results[0]

print(f"\n--- Визуализация предсказаний для {image_path} ---")

# Загружаем оригинальное изображение для рисования поверх него
original_img = cv2.imread(image_path)
if original_img is None:
    print(f"Ошибка: Изображение не найдено по пути {image_path}")
    exit()

output_image = original_img.copy()

class_counts = {}
# --- Детальная информация об обнаруженных объектах и их визуализация ---
print("\n--- Детализация обнаруженных объектов ---")
if result.boxes:
    names = result.names

    for i in range(len(result.boxes)):
        box = result.boxes.xyxy[i].cpu().numpy().astype(int) # bbox (x1, y1, x2, y2)
        score = result.boxes.conf[i].item()                   # Уверенность
        class_id = int(result.boxes.cls[i].item())            # ID класса
        class_name = names[class_id]                          # Имя класса

        # Обновляем счетчик классов
        if class_name not in class_counts:
            class_counts[class_name] = 0
        class_counts[class_name] += 1

        # Генерируем случайный цвет для текущего объекта
        color = get_random_color()

        # Выводим детали объекта
        print(f"Объект {i+1}:")
        print(f"  Класс: {class_name}")
        print(f"  Уверенность: {score:.2f}")
        print(f"  Bounding Box (x1,y1,x2,y2): {box}")

        # Рисуем bounding box
        cv2.rectangle(output_image, (box[0], box[1]), (box[2], box[3]), color, 2)

        # Добавляем текст с меткой и уверенностью
        text = f"{class_name}: {score:.2f}"
        cv2.putText(output_image, text, (box[0], box[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        # Если есть маски сегментации для этого объекта
        if result.masks:
            # Маски находятся в result.masks.data (тензор NxHxW)
            # Применяем маску к изображению
            mask = result.masks.data[i].cpu().numpy() # Получаем маску для текущего объекта
            # Масштабируем маску до 0-255 и преобразуем в uint8
            mask_scaled = (mask * 255).astype(np.uint8)

            # Изменяем размер маски до исходного изображения (если она была уменьшена)
            # YOLOv8 маски обычно имеют размер 160x160, а затем интерполируются.
            # result.masks.data уже интерполирована до оригинального размера изображения.
            # Убедимся, что размеры маски соответствуют изображению для попиксельного наложения
            mask_resized = cv2.resize(mask_scaled, (original_img.shape[1], original_img.shape[0]), interpolation=cv2.INTER_LINEAR)

            # Создаем цветную маску
            mask_colored = np.zeros_like(output_image, dtype=np.uint8)
            # Накладываем цвет на те пиксели, где маска > 0
            mask_colored[mask_resized > 128] = color # Используем 128 как порог для бинарной маски (можно 0)

            # Накладываем маску на изображение с прозрачностью
            alpha = 0.4 # Прозрачность маски
            output_image = cv2.addWeighted(output_image, 1, mask_colored, alpha, 0)

    # Выводим общую статистику
    print("\n--- Общая статистика обнаруженных объектов ---")
    if class_counts:
        for class_name, count in class_counts.items():
            print(f"- Класс '{class_name}': {count} объект(ов)")
    else:
        print("Объектов не обнаружено с заданными параметрами.")

else:
    print("Объектов не обнаружено.")

# Отображение результата
cv2_imshow(output_image)'''
        elif number == 112:
            return '''
import torch
import torchvision
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import cv2
from google.colab.patches import cv2_imshow

PASCAL_VOC_CLASSES = [
    '__background__', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
    'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
    'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
]

def create_pascal_voc_colormap():
    colormap = []
    for i in range(len(PASCAL_VOC_CLASSES)):
        random.seed(i * 12345)
        r = random.randint(0, 255)
        g = random.randint(0, 255)
        b = random.randint(0, 255)
        # Добавляем немного яркости для лучшей видимости
        colormap.append((r, g, b))
    colormap[0] = (0, 0, 0) # Черный для фона
    return colormap

COLORMAP = create_pascal_voc_colormap()

model = torchvision.models.segmentation.deeplabv3_resnet101(pretrained=True)
model.eval()
image_path = "/content/test2.jpg"

try:
    image = Image.open(image_path).convert("RGB")
except FileNotFoundError:
    print(f"Ошибка: Изображение не найдено по пути {image_path}")
    exit()

preprocess = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
input_tensor = preprocess(image).unsqueeze(0) # Добавляем батч-размерность
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)
input_tensor = input_tensor.to(device)

with torch.no_grad():
    output = model(input_tensor)["out"][0] # Получаем логиты масок
# Применяем argmax, чтобы получить предсказанный класс для каждого пикселя
output_predictions = output.argmax(0)
predicted_mask_np = output_predictions.byte().cpu().numpy()

# --- 5. Визуализация с цветовой кодировкой и статистика ---
print("\n--- Визуализация и статистика сегментации ---")

segmented_image_colored = np.zeros((*predicted_mask_np.shape, 3), dtype=np.uint8)
pixel_counts_per_class = {}

for class_id, class_name in enumerate(PASCAL_VOC_CLASSES):
    # Находим все пиксели, принадлежащие текущему классу
    mask_for_class = (predicted_mask_np == class_id)
    num_pixels = np.sum(mask_for_class)

    if num_pixels > 0:
        pixel_counts_per_class[class_name] = num_pixels
        # Присваиваем цвет пикселям этого класса
        color = COLORMAP[class_id]
        segmented_image_colored[mask_for_class] = color

# Накладываем сегментированное изображение на оригинал для лучшей наглядности (с прозрачностью)
original_img_np = np.array(image) # PIL Image (RGB) -> NumPy array (RGB)
# cv2_imshow ожидает BGR, поэтому конвертируем
original_img_bgr = original_img_np[:, :, ::-1].copy()

alpha = 0.5 # Прозрачность наложения
plt.figure(figsize=(15, 7))
plt.subplot(1, 2, 1)
plt.imshow(original_img_np)
plt.title('Оригинальное изображение')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(original_img_np)
plt.imshow(segmented_image_colored, alpha=alpha) # Накладываем цветную маску
plt.title('Сегментированное изображение (DeepLabV3)')
plt.axis('off')

# Создаем легенду для классов
handles = []
labels = []
for class_id, class_name in enumerate(PASCAL_VOC_CLASSES):
    if class_name in pixel_counts_per_class or class_id == 0: # Добавляем фон в легенду
        color = np.array(COLORMAP[class_id]) / 255.0 # Нормализуем для matplotlib
        handles.append(plt.Rectangle((0, 0), 1, 1, fc=color))
        labels.append(class_name)

# Устанавливаем легенду за пределами графика
plt.legend(handles, labels, loc='center left', bbox_to_anchor=(1, 0.5), fontsize='small', ncol=1)
plt.tight_layout(rect=[0, 0, 0.85, 1]) # Adjust layout to make room for the legend
plt.show()

# Выводим статистику по количеству пикселей
print("\n--- Статистика пикселей по классам ---")
if pixel_counts_per_class:
    total_pixels = predicted_mask_np.size
    for class_name, count in sorted(pixel_counts_per_class.items(), key=lambda item: item[1], reverse=True):
        percentage = (count / total_pixels) * 100
        print(f"- Класс '{class_name}': {count} пикселей ({percentage:.2f}%)")
    print(f"Всего пикселей в изображении: {total_pixels}")
else:
    print("Пиксели классов не обнаружены.")'''
        elif number == 111:
            return '''
from ultralytics import YOLO
import cv2
import numpy as np
import matplotlib.pyplot as plt
from google.colab.patches import cv2_imshow
import random

def get_random_color():
    r = random.randint(50, 255)
    g = random.randint(50, 255)
    b = random.randint(50, 255)
    return (b, g, r) # Return in BGR format for OpenCV

model = YOLO('yolov8n.pt')
print(f"Model: {model.model_name}")

image_path = "/content/test2.jpg"

results = model(
    image_path,
    conf=0.7,             # Only show detections with 70% confidence or higher
    iou=0.5,              # Moderate NMS threshold
    imgsz=960,            # Resize image to 960x960
    device='cuda:0',      # Explicitly use the first GPU
    classes=[0, 2],       # Only detect 'person' (ID 0) and 'car' (ID 2)
    save=True,            # Save annotated image to disk
    save_txt=True,        # Save detection details as text files
    project='my_detection_run', # Custom directory for saved results
    name='image_output',  # Custom subdirectory name
    verbose=True          # Show detailed output during inference
)
result = results[0]

# Load the original image for drawing on it
original_img = cv2.imread(image_path)
if original_img is None:
    print(f"Error: Image not found at path {image_path}")
    exit()

output_image = original_img.copy()
class_counts = {}

print("\n--- Details of Detected Objects ---")
if result.boxes:
    # result.boxes contains: xyxy (bbox), conf (confidence), cls (class ID)
    names = result.names # Dictionary mapping class ID to class name

    for i in range(len(result.boxes)):
        box = result.boxes.xyxy[i].cpu().numpy().astype(int) # bbox (x1, y1, x2, y2)
        score = result.boxes.conf[i].item()                   # Confidence score
        class_id = int(result.boxes.cls[i].item())            # Class ID
        class_name = names[class_id]                          # Class name

        # Update class counter
        if class_name not in class_counts:
            class_counts[class_name] = 0
        class_counts[class_name] += 1

        # Generate a unique color for the current object
        color = get_random_color()

        # Print object details
        print(f"Object {i+1}:")
        print(f"  Class: {class_name}")
        print(f"  Confidence: {score:.2f}")
        print(f"  Bounding Box (x1,y1,x2,y2): {box}")

        # Draw bounding box with the chosen color
        cv2.rectangle(output_image, (box[0], box[1]), (box[2], box[3]), color, 2)

        # Add text label with class name and confidence
        text = f"{class_name}: {score:.2f}"
        cv2.putText(output_image, text, (box[0], box[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)


cv2_imshow(output_image)

print("\n--- Overall Statistics of Detected Objects ---")
if class_counts:
    for class_name, count in class_counts.items():
        print(f"- Class '{class_name}': {count} object(s)")
else:
    print("No objects detected with the given parameters.")

output_filename = 'output_detection.jpg'
cv2.imwrite(output_filename, output_image)
print(f"\nResult saved to: {output_filename}")'''
        elif number == 110:
            return '''
import torchvision
from PIL import Image
import matplotlib.pyplot as plt

# Загрузка модели
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
model.eval()  # Переводим в режим inference

# Загрузка и преобразование изображения
image = Image.open('image.jpg')
transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
])
img_tensor = transform(image).unsqueeze(0)

# Детекция
with torch.no_grad():
    predictions = model(img_tensor)

# Визуализация
boxes = predictions[0]['boxes'].numpy()
labels = predictions[0]['labels'].numpy()
scores = predictions[0]['scores'].numpy()

# Отображаем только объекты с уверенностью > 0.7
for box, label, score in zip(boxes, labels, scores):
    if score > 0.7:
        x1, y1, x2, y2 = box
        plt.gca().add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor='red', linewidth=2))
        plt.text(x1, y1, f"{label}: {score:.2f}", color='white', backgroundcolor='red')

plt.imshow(image)
plt.show()'''
        elif number == 106:
            return '''
import cv2
from google.colab.patches import cv2_imshow
image_path = 'заяцволк.jpg'
template_path = 'шаблон.PNG'

image = cv2.imread(image_path)
template = cv2.imread(template_path, 0)
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
result = cv2.matchTemplate(gray_image, template, cv2.TM_CCOEFF_NORMED)

threshold = 0.8
locations = np.where(result >= threshold)

for pt in zip(*locations[::-1]):
    cv2.rectangle(image, pt, (pt[0] + template.shape[1], pt[1] + template.shape[0]), (0, 255, 0), 2)

# Отображение результата
cv2_imshow(image)'''
        elif number == 105:
            return '''
image_path = '/content/drive/MyDrive/Финашка/Машинное зрение/1 сем/dogs.jpg'
image = cv2.imread(image_path)
img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(img_gray, 50, 170)
cv2_imshow(edges)

#Усиление границ
image_path = '/content/drive/MyDrive/Финашка/Машинное зрение/1 сем/заяцволк.jpg'
image = cv2.imread(image_path)
img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(img_gray, 50, 100)
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
enhanced_edges = cv2.dilate(edges, kernel, iterations=1)
cv2_imshow(enhanced_edges)'''
        elif number == 102:
            return '''
import cv2
import numpy as np
import matplotlib.pyplot as plt
from google.colab.patches import cv2_imshow # Это функция для отображения изображений в Google Colab

def enhance_image_all_methods(image_path):
    # Загружаем изображение в градациях серого
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Ошибка загрузки изображения по пути: {image_path}")
        return

    # Создаем словарь для хранения результатов методов
    methods = {}

    # --- Добавляем оригинальное изображение и гистограмму ---
    methods['Original'] = img # Добавляем оригинал в словарь для единого цикла отображения

    # 1. Инверсия изображения
    inverted = 255 - img
    methods['Inversion'] = inverted

    # 2. Степенное преобразование (гамма-коррекция)
    gamma = 2.2
    gamma_corrected = np.array(255 * (img / 255) ** gamma, dtype='uint8')
    methods[f'Gamma Correction (γ={gamma})'] = gamma_corrected

    # 3. Логарифмическое преобразование
    # Чтобы избежать log(0), добавляем 1 перед логарифмом
    # Константа 'c' для масштабирования результата до диапазона [0, 255]
    c = 255 / np.log(1 + np.float32(np.max(img)))
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
    # Избегаем деления на ноль, если все пиксели имеют одинаковое значение
    if (max_val - min_val) == 0:
        contrast_stretched = np.zeros_like(img) # Все пиксели станут 0 (черным)
    else:
        contrast_stretched = np.array(255 * (img - min_val) / (max_val - min_val), dtype='uint8')
    methods['Contrast Stretching'] = contrast_stretched

    # 7. Сигмоидальная коррекция
    gain = 10
    cutoff = 128
    sigmoid = 255 / (1 + np.exp(-gain * ((img - cutoff) / 255.0)))
    sigmoid_corrected = np.array(sigmoid, dtype='uint8')
    methods['Sigmoid Correction'] = sigmoid_corrected

    # --- Вывод результатов ---

    # Вывод оригинального изображения с cv2_imshow
    print("--- Оригинальное изображение ---")
    cv2_imshow(img)
    print("\n") # Добавляем пустую строку для отступа

    # Вывод всех обработанных изображений и их гистограмм с matplotlib
    # +1 к n, чтобы учесть "Original"
    n_rows = len(methods)
    plt.figure(figsize=(14, 3 * n_rows))

    for i, (title, result) in enumerate(methods.items()):
        # Изображение
        plt.subplot(n_rows, 2, 2*i + 1)
        plt.imshow(result, cmap='gray')
        plt.title(title)
        plt.axis('off')

        # Гистограмма
        plt.subplot(n_rows, 2, 2*i + 2)
        plt.hist(result.ravel(), bins=256, range=(0, 256), color='black')
        plt.title(f'{title} Histogram')
        plt.xlabel('Pixel Intensity')
        plt.ylabel('Frequency')
        # Добавляем tight_layout в конце цикла или после subplots для корректного размещения
        plt.tight_layout()

    plt.show()

    # Выводы
    print("\n--- Выводы по методам улучшения изображения ---")
    print("- **Оригинал:** Представляет собой исходное изображение и его распределение яркости.")
    print("- **Инверсия:** Меняет яркие области на темные и наоборот. Полезно для выделения деталей на темном фоне или для создания негативного эффекта.")
    print("- **Гамма-коррекция (γ):** Регулирует общую яркость и контрастность изображения. Если γ > 1, изображение становится темнее (усиливаются светлые тона), если γ < 1 — светлее (усиливаются темные тона).")
    print("- **Логарифмическое преобразование:** Увеличивает динамический диапазон темных областей изображения, делая их более различимыми, при этом сжимая светлые области. Полезно для изображений с большим динамическим диапазоном, где есть как очень темные, так и очень светлые участки.")
    print("- **Эквализация гистограммы:** Распределяет интенсивности пикселей таким образом, чтобы гистограмма стала более плоской. Это приводит к улучшению контраста, особенно на изображениях с узким динамическим диапазоном (когда пиксели сосредоточены в небольшой части диапазона яркости).")
    print("- **CLAHE (Адаптивное выравнивание гистограммы с ограничением контраста):** Работает локально, разделяя изображение на небольшие блоки и применяя эквализацию к каждому блоку. Ограничение контраста (clipLimit) предотвращает чрезмерное усиление шума. Это особенно полезно для изображений с неравномерным освещением или когда эквализация всей гистограммы создает нежелательные артефакты.")
    print("- **Контрастное растяжение:** Расширяет диапазон интенсивности пикселей исходного изображения до полного диапазона (0–255). Это максимально увеличивает контраст, растягивая существующие значения яркости.")
    print("- **Сигмоидальная коррекция:** Применяет сигмоидальную функцию для изменения интенсивности пикселей, что приводит к улучшению контраста в средней части диапазона яркости при сглаживании очень темных и очень светлых значений. Параметры `gain` (крутизна кривой) и `cutoff` (точка перегиба) позволяют контролировать форму кривой.")'''
        elif number == 101:
            return '''
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.filters import threshold_otsu, threshold_local, threshold_yen, threshold_li, threshold_isodata, threshold_triangle
from skimage.exposure import histogram
from google.colab.patches import cv2_imshow

def segment_image_by_histogram(img_path):
    # --- 1. Загрузка изображения ---
    image_bgr = cv2.imread(img_path)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    image_gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    # --- 2. Построение гистограммы grayscale ---
    plt.figure(figsize=(6, 4))
    hist, hist_centers = histogram(image_gray)
    plt.plot(hist_centers, hist, color='black')
    plt.title('Grayscale Histogram')
    plt.xlabel('Intensity Value')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.show()

    # --- 3. Пороговые методы для grayscale ---
    gray_otsu = threshold_otsu(image_gray)
    gray_yen = threshold_yen(image_gray)
    gray_local = threshold_local(image_gray, block_size=31, method='gaussian', offset=5)

    print(f"Grayscale thresholds:")
    print(f" - Otsu: {gray_otsu}")
    print(f" - Renyi (Yen): {gray_yen}")

    # --- 4. Маски ---
    mask_gray_otsu = (image_gray >= gray_otsu).astype(np.uint8) * 255
    mask_gray_yen = (image_gray >= gray_yen).astype(np.uint8) * 255
    mask_gray_adaptive = (image_gray >= gray_local).astype(np.uint8) * 255

    # --- 5. Применение масок ---
    segmented_gray_otsu = cv2.bitwise_and(image_rgb, image_rgb, mask=mask_gray_otsu)
    segmented_gray_yen = cv2.bitwise_and(image_rgb, image_rgb, mask=mask_gray_yen)
    segmented_gray_adaptive = cv2.bitwise_and(image_rgb, image_rgb, mask=mask_gray_adaptive)

    # --- 6. Визуализация grayscale результатов ---
    titles = ['Gray Otsu', 'Gray Renyi (Yen)', 'Gray Adaptive']
    masks = [mask_gray_otsu, mask_gray_yen, mask_gray_adaptive]
    results = [segmented_gray_otsu, segmented_gray_yen, segmented_gray_adaptive]

    plt.figure(figsize=(12, 8))
    for i in range(3):
        plt.subplot(3, 2, i * 2 + 1)
        plt.title(f"{titles[i]} - Mask")
        plt.imshow(masks[i], cmap='gray')
        plt.axis('off')

        plt.subplot(3, 2, i * 2 + 2)
        plt.title(f"{titles[i]} - Segmented Image")
        plt.imshow(results[i])
        plt.axis('off')
    plt.tight_layout()
    plt.show()

    # --- 7.  RGB-каналы --
    R, G, B = cv2.split(image_rgb)
    channels = [R, G, B]
    channel_names = ['Red', 'Green', 'Blue']
    colors = ['red', 'green', 'blue']

    # --- 8. Гистограммы RGB ---
    plt.figure(figsize=(12, 4))
    for i, (channel, color, name) in enumerate(zip(channels, colors, channel_names)):
        plt.subplot(1, 3, i + 1)
        plt.hist(channel.ravel(), bins=256, range=(0, 256), color=color)
        plt.title(f'{name} channel histogram')
    plt.tight_layout()
    plt.show()

    # --- 9. Порог Оцу и Реньи для RGB ---
    otsu_thresholds = [threshold_otsu(c) for c in channels]
    print(f"Оцу RGB пороги: {otsu_thresholds}")
    mean_otsu = int(np.mean(otsu_thresholds))

    renyi_thresholds = [threshold_yen(c) for c in channels]
    print(f"Реньи RGB пороги: {renyi_thresholds}")
    mean_renyi = int(np.mean(renyi_thresholds))

    # --- 10. Адаптивный порог RGB ---
    block_size = 31
    adapt_thresh_r = cv2.adaptiveThreshold(R, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                           cv2.THRESH_BINARY, block_size, 5)
    adapt_thresh_g = cv2.adaptiveThreshold(G, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                           cv2.THRESH_BINARY, block_size, 5)
    adapt_thresh_b = cv2.adaptiveThreshold(B, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                           cv2.THRESH_BINARY, block_size, 5)
    adaptive_mask = cv2.bitwise_and(cv2.bitwise_and(adapt_thresh_r, adapt_thresh_g), adapt_thresh_b)

    def make_mask(thresh):
        binary_r = (R >= thresh).astype(np.uint8)
        binary_g = (G >= thresh).astype(np.uint8)
        binary_b = (B >= thresh).astype(np.uint8)
        return (binary_r & binary_g & binary_b) * 255

    mask_otsu = make_mask(mean_otsu).astype(np.uint8)
    mask_renyi = make_mask(mean_renyi).astype(np.uint8)

    segmented_otsu = cv2.bitwise_and(image_rgb, image_rgb, mask=mask_otsu)
    segmented_renyi = cv2.bitwise_and(image_rgb, image_rgb, mask=mask_renyi)
    segmented_adaptive = cv2.bitwise_and(image_rgb, image_rgb, mask=adaptive_mask)

    # --- 11. Визуализация RGB результатов ---
    titles = ['RGB Otsu', 'RGB Renyi (Yen)', 'RGB Adaptive']
    masks = [mask_otsu, mask_renyi, adaptive_mask]
    results = [segmented_otsu, segmented_renyi, segmented_adaptive]

    plt.figure(figsize=(12, 8))
    for i in range(3):
        plt.subplot(3, 2, i * 2 + 1)
        plt.title(f"{titles[i]} - Mask")
        plt.imshow(masks[i], cmap='gray')
        plt.axis('off')

        plt.subplot(3, 2, i * 2 + 2)
        plt.title(f"{titles[i]} - Segmented Image")
        plt.imshow(results[i])
        plt.axis('off')
    plt.tight_layout()
    plt.show()



import cv2
import numpy as np
import matplotlib.pyplot as plt
from google.colab.patches import cv2_imshow

def manual_threshold_segmentation(img_path, channel='gray', manual_thresh=90):
    # --- 1. Загрузка изображения ---
    image = cv2.imread(img_path)

    # --- 2. Выбор канала (по умолчанию — grayscale) ---
    if channel == 'gray':
        channel_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    elif channel == 'r':
        channel_img = image[:, :, 2]
    elif channel == 'g':
        channel_img = image[:, :, 1]
    elif channel == 'b':
        channel_img = image[:, :, 0]
    else:
        raise ValueError("Канал должен быть: 'gray', 'r', 'g' или 'b'")

    # --- 3. Построение гистограммы ---
    plt.figure(figsize=(6, 4))
    plt.hist(channel_img.ravel(), bins=256, range=(0, 256), color='black')
    plt.axvline(manual_thresh, color='red', linestyle='--', label=f'Threshold = {manual_thresh}')
    plt.title(f'Histogram of {channel.upper()} channel')
    plt.xlabel('Intensity Value')
    plt.ylabel('Pixel Count')
    plt.legend()
    plt.show()

    # --- 4. Применение ручного порога ---
    _, binary_mask = cv2.threshold(channel_img, manual_thresh, 255, cv2.THRESH_BINARY)

    # --- 5. Наложение маски на оригинал ---
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    segmented = cv2.bitwise_and(image_rgb, image_rgb, mask=binary_mask)

    # --- 6. Визуализация результата ---
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.title("Binary Mask")
    plt.imshow(binary_mask, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.title("Segmented Image")
    plt.imshow(segmented)
    plt.axis('off')

    plt.tight_layout()
    plt.show()

'''
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
            import cv2
import numpy as np
import matplotlib.pyplot as plt

def affine_transformations(image_path):
    img = cv2.imread(image_path)
    if img is None:
        print("Ошибка загрузки изображения")
        return

    rows, cols = img.shape[:2]
    methods = {}

    # 1. Перенос (сдвиг)
    dx, dy = 50, 30
    M_translate = np.float32([[1, 0, dx], [0, 1, dy]])
    translated = cv2.warpAffine(img, M_translate, (cols, rows))
    methods['Translation (dx=50, dy=30)'] = translated

    # 2. Вращение
    angle = 45
    center = (cols // 2, rows // 2)
    M_rotate = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(img, M_rotate, (cols, rows))
    methods['Rotation (45°)'] = rotated

    # 3. Масштабирование (увеличение)
    scale_factor = 1.5
    scaled = cv2.resize(img, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_LINEAR)
    methods['Scaling 1.5x (Bilinear)'] = scaled

    # 4. Интерполяции (уменьшение)
    half_size = (cols // 2, rows // 2)

    resized_nearest = cv2.resize(img, half_size, interpolation=cv2.INTER_NEAREST)
    methods['Resize 0.5x (Nearest)'] = resized_nearest

    resized_bilinear = cv2.resize(img, half_size, interpolation=cv2.INTER_LINEAR)
    methods['Resize 0.5x (Bilinear)'] = resized_bilinear

    resized_bicubic = cv2.resize(img, half_size, interpolation=cv2.INTER_CUBIC)
    methods['Resize 0.5x (Bicubic)'] = resized_bicubic

    # Отображение изображений + гистограмм
    n = len(methods)
    plt.figure(figsize=(12, 4 * n))

    for i, (title, result) in enumerate(methods.items()):
        # Преобразование в градации серого для гистограммы
        gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)

        # Изображение
        plt.subplot(n, 2, 2 * i + 1)
        plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
        plt.title(title)
        plt.axis('off')

        # Гистограмма яркости
        plt.subplot(n, 2, 2 * i + 2)
        plt.hist(gray.ravel(), bins=256, range=(0, 256), color='black')
        plt.title(f'{title} - Histogram')

    plt.tight_layout()
    plt.show()

    # Выводы
    print("\nВыводы:")
    print("- Аффинные преобразования сохраняют прямые линии и параллельность.")
    print("- Перенос просто сдвигает изображение, гистограмма почти не меняется.")
    print("- Вращение и масштабирование могут изменить распределение яркости, особенно если происходит обрезка.")
    print("- Интерполяции влияют на гладкость: Bicubic — самая мягкая, Nearest — грубая.")
    print("- При сильном масштабировании bicubic может давать наилучший визуальный результат.")
'''
        elif number == 104:
            return '''
            import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage

def all_filters(image_path):
    # Загрузка изображения
    img = cv2.imread(image_path)
    if img is None:
        print("Ошибка загрузки изображения")
        return

    # Перевод в градации серого (упрощает обработку фильтрами)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Словарь для хранения результатов
    methods = {}

    # === 1. Гауссов фильтр ===
    # Сглаживает изображение, устраняет мелкий шум
    # ksize=(5,5) — размер ядра, sigmaX — стандартное отклонение по X
    gaussian = cv2.GaussianBlur(gray, (5, 5), sigmaX=1)
    methods['Gaussian Blur'] = gaussian

    # === 2. Фильтр среднего (усреднение) ===
    # Каждый пиксель заменяется средним значением соседей
    mean = cv2.blur(gray, (5, 5))
    methods['Mean Filter'] = mean

    # === 3. Медианный фильтр ===
    # Хорошо справляется с шумом «соль-перец»
    # ksize=5 — размер окна 5x5
    median = cv2.medianBlur(gray, 5)
    methods['Median Filter'] = median

    # === 4. Билатеральный фильтр ===
    # Сохраняет края, сглаживая шум
    # d — диаметр области, sigmaColor — фильтрация по цвету, sigmaSpace — по пространству
    bilateral = cv2.bilateralFilter(gray, d=9, sigmaColor=75, sigmaSpace=75)
    methods['Bilateral Filter'] = bilateral

    # === 5. Фильтр Собеля (градиенты) ===
    # Вычисляет производные по x и y, усиливает границы
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)  # по X
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)  # по Y
    sobel = cv2.magnitude(sobelx, sobely)                # итоговая амплитуда градиента
    sobel = np.uint8(np.clip(sobel, 0, 255))
    methods['Sobel Filter (Edges)'] = sobel

    # === 6. Лапласиан ===
    # Второй производный оператор, подчеркивает быстрые изменения
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    laplacian = np.uint8(np.clip(np.abs(laplacian), 0, 255))
    methods['Laplacian Filter'] = laplacian

    # === 7. Unsharp Masking (маска резкости) ===
    # Повышает резкость изображения путём вычитания размытой копии
    blurred = cv2.GaussianBlur(gray, (9, 9), 10.0)
    unsharp = cv2.addWeighted(gray, 1.5, blurred, -0.5, 0)
    # gray * 1.5 - blurred * 0.5
    methods['Unsharp Masking'] = unsharp

    # === 8. Фильтр Прюитта (Prewitt) ===
    # Похож на Собеля, но проще по вычислениям
    kernelx = np.array([[1, 0, -1],
                        [1, 0, -1],
                        [1, 0, -1]], dtype=np.float32)  # по X

    kernely = np.array([[1,  1,  1],
                        [0,  0,  0],
                        [-1, -1, -1]], dtype=np.float32)  # по Y

    prewitt_x = ndimage.convolve(gray.astype(np.float32), kernelx)
    prewitt_y = ndimage.convolve(gray.astype(np.float32), kernely)
    prewitt = np.hypot(prewitt_x, prewitt_y)  # вычисляем амплитуду градиента
    prewitt = np.uint8(np.clip(prewitt, 0, 255))
    methods['Prewitt Filter (Edges)'] = prewitt

    # === 9. Детектор границ Кэнни (Canny) ===
    # Мощный алгоритм с порогами, подавлением немаксимумов
    # Пороговые значения 100 и 200
    canny = cv2.Canny(gray, 100, 200)
    methods['Canny Edge Detector'] = canny

    # === Визуализация изображений и гистограмм ===
    n = len(methods)
    plt.figure(figsize=(12, 4 * n))

    for i, (title, result) in enumerate(methods.items()):
        # Отображение результата фильтра
        plt.subplot(n, 2, 2 * i + 1)
        plt.imshow(result, cmap='gray')
        plt.title(title)
        plt.axis('off')

        # Гистограмма яркости
        plt.subplot(n, 2, 2 * i + 2)
        plt.hist(result.ravel(), bins=256, range=(0, 256), color='black')
        plt.title(f'{title} - Histogram')

    plt.tight_layout()
    plt.show()

    # === Выводы ===
    print("\nВыводы:")
    print("- Gaussian, Median, Mean — убирают шум. Median особенно хорош при шуме 'соль-перец'.")
    print("- Bilateral — уникален: сглаживает, но сохраняет края.")
    print("- Sobel и Prewitt — градиентные фильтры, полезны для поиска направлений границ.")
    print("- Laplacian — выделяет быстрые изменения интенсивности.")
    print("- Unsharp Masking — улучшает резкость за счёт вычитания размытой копии.")
    print("- Canny — мощный фильтр, надёжно выделяет границы с подавлением шумов.")
'''
        elif number == 103:
            return '''
            import cv2
import numpy as np
import matplotlib.pyplot as plt

def morphological_operations(image_path):
    # Загрузка изображения
    img = cv2.imread(image_path)
    if img is None:
        print("Ошибка загрузки изображения")
        return

    # Перевод в оттенки серого
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Бинаризация изображения для морфологических операций
    # (лучше работает на чётком черно-белом)
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

    # Создание структурирующего элемента (ядра)
    # 3x3 квадрат, можно изменить на другие формы и размеры
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

    # === 1. Эрозия ===
    # Уменьшает яркие области, "съедает" границы
    erosion = cv2.erode(binary, kernel, iterations=1)

    # === 2. Расширение ===
    # Увеличивает яркие области, "расширяет" границы
    dilation = cv2.dilate(binary, kernel, iterations=1)

    # === 3. Открытие ===
    # Последовательность: эрозия → расширение
    # Удаляет мелкие объекты/шум, сохраняет крупные формы
    opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    # === Отображение результатов ===
    images = {
        "Исходное (бинарное)": binary,
        "Эрозия": erosion,
        "Расширение": dilation,
        "Открытие": opening
    }

    plt.figure(figsize=(10, 10))

    for i, (title, result) in enumerate(images.items()):
        # Показываем изображение
        plt.subplot(len(images), 2, 2 * i + 1)
        plt.imshow(result, cmap='gray')
        plt.title(title)
        plt.axis('off')

        # Гистограмма
        plt.subplot(len(images), 2, 2 * i + 2)
        plt.hist(result.ravel(), bins=256, range=(0, 256), color='black')
        plt.title(f'{title} - Histogram')

    plt.tight_layout()
    plt.show()

    # === Выводы ===
    print("\nВыводы:")
    print("- Эрозия удаляет мелкие объекты и уменьшает яркие участки.")
    print("- Расширение заполняет пробелы, увеличивает яркие области.")
    print("- Открытие удаляет шум, сохраняя форму более крупных объектов.")
    print("- Подходит для предобработки перед выделением контуров или OCR.")
'''
        elif number == 5:
            return '''
            import cv2
import numpy as np
import matplotlib.pyplot as plt

def fourier_transform_analysis(image_path):
    # Загрузка изображения в градациях серого
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print("Ошибка загрузки изображения")
        return

    # === 1. Прямое 2D-преобразование Фурье ===
    f = np.fft.fft2(img)  # комплексное 2D FFT
    fshift = np.fft.fftshift(f)  # перенос нуля частот в центр
    magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1)  # лог-масштаб амплитуды

    # === 2. Обратное преобразование Фурье ===
    f_ishift = np.fft.ifftshift(fshift)  # сдвиг обратно
    img_back = np.fft.ifft2(f_ishift)  # обратное FFT
    img_back = np.abs(img_back)  # берем только действительную часть

    # === Отображение результатов ===
    images = {
        "Исходное изображение": img,
        "Амплитудный спектр (лог)": magnitude_spectrum,
        "Обратное преобразование": img_back
    }

    plt.figure(figsize=(12, 8))

    for i, (title, image) in enumerate(images.items()):
        # Показываем изображение
        plt.subplot(len(images), 2, 2 * i + 1)
        plt.imshow(image, cmap='gray')
        plt.title(title)
        plt.axis('off')

        # Гистограмма яркости
        plt.subplot(len(images), 2, 2 * i + 2)
        plt.hist(image.ravel(), bins=256, range=(0, 256), color='black')
        plt.title(f"{title} - Histogram")

    plt.tight_layout()
    plt.show()

    # === Выводы ===
    print("\nВыводы:")
    print("- Преобразование Фурье переводит изображение в частотную область.")
    print("- Центр спектра — низкие частоты (фон), края — высокие (детали, шум).")
    print("- Логарифмическое усиление помогает увидеть слабые частоты.")
    print("- Обратное преобразование восстанавливает изображение с высокой точностью.")
    print("- Частотный анализ полезен для фильтрации, выделения текстур и восстановления.")
'''
        elif number == 6:
            return '''
            from ikomia.dataprocess.workflow import Workflow
import cv2
import collections
import os

def run_tracking_workflow(
    input_video_path: str,
    output_video_path: str,
    tracker_type: str = "deepsort",  # "deepsort", "sort", "ocsort"
    detection_model: str = "infer_yolo_v7",  # см. список моделей ниже
    categories_to_track: str = "person,car"
):
    """
    Запуск пайплайна отслеживания объектов с использованием Ikomia Workflow.

    Аргументы:
    ----------
    input_video_path : str
        Путь к входному видеофайлу.
    output_video_path : str
        Путь для сохранения выходного видео.
    tracker_type : str
        Тип алгоритма трекинга:
            - "deepsort" (точный, appearance-based)
            - "sort" (простой и быстрый, без appearance)
            - "ocsort" (современный, устойчивый к окклюзии)

    detection_model : str
        Название модели для детекции объектов. Поддерживаются:
            - "infer_yolo_v7"
            - "infer_yolo_v5"
            - "infer_yolo_v8"
            - "infer_yolox"
            - "infer_yolo_nas"
            - "infer_efficientdet"
            - "infer_ssd"
            - "infer_torchvision_frcnn"  (Faster R-CNN)
            - "infer_torchvision_retinanet"

        Если модель не установлена — установить через:
            from ikomia.utils import ik
            ik.install("название_модуля")

    categories_to_track : str
        Классы объектов, которые нужно отслеживать, через запятую.
        Пример: "person,car,dog"
    """

    if not os.path.exists(input_video_path):
        print(f"Файл не найден: {input_video_path}")
        return

    wf = Workflow()

    # Добавление детектора объектов
    detector = wf.add_task(name=detection_model, auto_connect=True)
    detector.set_parameters({})  # можно задать параметры при необходимости

    # Выбор трекера
    tracker_plugins = {
        "deepsort": "infer_deepsort",
        "sort": "infer_sort",
        "ocsort": "infer_ocsort"
    }

    if tracker_type.lower() not in tracker_plugins:
        print(f"Ошибка: неизвестный трекер '{tracker_type}'")
        return

    tracking = wf.add_task(name=tracker_plugins[tracker_type.lower()], auto_connect=True)

    # Общие параметры
    common_params = {
        "categories": categories_to_track,
        "iou_threshold": "0.3"
    }

    if tracker_type == "deepsort":
        common_params.update({
            "conf_thres": "0.5",
            "max_age": "50",
            "min_hits": "3",
            "cosine_threshold": "0.2",
            "nn_budget": "100",
            "use_cuda": "True"
        })

    tracking.set_parameters(common_params)

    # Работа с видео
    stream = cv2.VideoCapture(input_video_path)
    if not stream.isOpened():
        print("Ошибка: не удалось открыть видео.")
        return

    frame_width = int(stream.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(stream.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_rate = stream.get(cv2.CAP_PROP_FPS)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, frame_rate, (frame_width, frame_height))

    frame_count = 0

    while True:
        ret, frame = stream.read()
        if not ret:
            print("Видео завершено или ошибка чтения.")
            break

        wf.run_on(array=frame)

        image_out = tracking.get_output(0)
        obj_detect_out = tracking.get_output(1)

        img_out = image_out.get_image_with_graphics(obj_detect_out)
        img_res = cv2.cvtColor(img_out, cv2.COLOR_RGB2BGR)

        # Подсчёт объектов
        class_counts = collections.defaultdict(int)
        for detection in obj_detect_out.get_objects():
            class_counts[detection.label] += 1

        # Отображение статистики на кадре
        y_offset = 30
        for class_name, count in class_counts.items():
            cv2.putText(img_res, f"{class_name}: {count}", (10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            y_offset += 30

        out.write(img_res)
        frame_count += 1

    stream.release()
    out.release()
    print(f"\nОбработка завершена. Видео сохранено в: {output_video_path}")
'''
        elif number == 7:
            return '''
            import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread('/content/drive/MyDrive/Финашка/Машинное зрение/3 сем/text4.jpg', cv2.IMREAD_GRAYSCALE)
kernel = np.ones((20, 20), np.uint8)
closed_image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title('Исходное изображение')
plt.imshow(image, cmap='gray')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title('Изображение после закрытия')
plt.imshow(closed_image, cmap='gray')
plt.axis('off')

plt.show()
'''
        elif number == 8:
            return '''
            import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.filters import threshold_otsu, threshold_local
from skimage.filters import threshold_yen, threshold_li, threshold_isodata, threshold_triangle
from skimage.exposure import histogram


def segment_image_by_histogram(img_path):
    # --- 1. Загрузка изображения и преобразование в RGB ---
    image_bgr = cv2.imread(img_path)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    # --- 2. Разделение на каналы ---
    R, G, B = cv2.split(image_rgb)
    channels = [R, G, B]
    channel_names = ['Red', 'Green', 'Blue']
    colors = ['red', 'green', 'blue']

    # --- 3. Построение гистограмм ---
    plt.figure(figsize=(12, 4))
    for i, (channel, color, name) in enumerate(zip(channels, colors, channel_names)):
        plt.subplot(1, 3, i + 1)
        plt.hist(channel.ravel(), bins=256, range=(0, 256), color=color)
        plt.title(f'{name} channel histogram')
    plt.tight_layout()
    plt.show()

    # --- 4. Порог по Оцу для каждого канала ---
    otsu_thresholds = [threshold_otsu(c) for c in channels]
    print(f"Пороги по Оцу (R, G, B): {otsu_thresholds}")
    mean_otsu = int(np.mean(otsu_thresholds))

    # --- 5. Порог по Реньи (используем threshold_yen как приближение) ---
    renyi_thresholds = [threshold_yen(c) for c in channels]
    print(f"Пороги по Реньи (R, G, B): {renyi_thresholds}")
    mean_renyi = int(np.mean(renyi_thresholds))

    # --- 6. Адаптивный порог (локальный) для каждого канала и объединение ---
    block_size = 31  # должен быть нечетным
    adapt_thresh_r = cv2.adaptiveThreshold(R, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                           cv2.THRESH_BINARY, block_size, 5)
    adapt_thresh_g = cv2.adaptiveThreshold(G, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                           cv2.THRESH_BINARY, block_size, 5)
    adapt_thresh_b = cv2.adaptiveThreshold(B, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                           cv2.THRESH_BINARY, block_size, 5)
    adaptive_mask = cv2.bitwise_and(cv2.bitwise_and(adapt_thresh_r, adapt_thresh_g), adapt_thresh_b)

    # --- 7. Маски по порогам Оцу и Реньи ---
    def make_mask(thresh):
        binary_r = (R >= thresh).astype(np.uint8)
        binary_g = (G >= thresh).astype(np.uint8)
        binary_b = (B >= thresh).astype(np.uint8)
        return (binary_r & binary_g & binary_b) * 255

    mask_otsu = make_mask(mean_otsu).astype(np.uint8)
    mask_renyi = make_mask(mean_renyi).astype(np.uint8)

    # --- 8. Применение масок к изображению ---
    segmented_otsu = cv2.bitwise_and(image_rgb, image_rgb, mask=mask_otsu)
    segmented_renyi = cv2.bitwise_and(image_rgb, image_rgb, mask=mask_renyi)
    segmented_adaptive = cv2.bitwise_and(image_rgb, image_rgb, mask=adaptive_mask)

    # --- 9. Визуализация результатов ---
    titles = ['Otsu Threshold', 'Renyi (Yen) Threshold', 'Adaptive Threshold']
    masks = [mask_otsu, mask_renyi, adaptive_mask]
    results = [segmented_otsu, segmented_renyi, segmented_adaptive]

    plt.figure(figsize=(12, 8))
    for i in range(3):
        plt.subplot(3, 2, i*2 + 1)
        plt.title(f"{titles[i]} - Mask")
        plt.imshow(masks[i], cmap='gray')
        plt.axis('off')

        plt.subplot(3, 2, i*2 + 2)
        plt.title(f"{titles[i]} - Segmented Image")
        plt.imshow(results[i])
        plt.axis('off')
    plt.tight_layout()
    plt.show()


import cv2
import numpy as np
import matplotlib.pyplot as plt

def manual_threshold_segmentation(img_path, channel='gray', manual_thresh=90):
    # --- 1. Загрузка изображения ---
    image = cv2.imread(img_path)

    # --- 2. Выбор канала (по умолчанию — grayscale) ---
    if channel == 'gray':
        channel_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    elif channel == 'r':
        channel_img = image[:, :, 2]
    elif channel == 'g':
        channel_img = image[:, :, 1]
    elif channel == 'b':
        channel_img = image[:, :, 0]
    else:
        raise ValueError("Канал должен быть: 'gray', 'r', 'g' или 'b'")

    # --- 3. Построение гистограммы ---
    plt.figure(figsize=(6, 4))
    plt.hist(channel_img.ravel(), bins=256, range=(0, 256), color='black')
    plt.axvline(manual_thresh, color='red', linestyle='--', label=f'Threshold = {manual_thresh}')
    plt.title(f'Histogram of {channel.upper()} channel')
    plt.xlabel('Intensity Value')
    plt.ylabel('Pixel Count')
    plt.legend()
    plt.show()

    # --- 4. Применение ручного порога ---
    _, binary_mask = cv2.threshold(channel_img, manual_thresh, 255, cv2.THRESH_BINARY)

    # --- 5. Наложение маски на оригинал ---
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    segmented = cv2.bitwise_and(image_rgb, image_rgb, mask=binary_mask)

    # --- 6. Визуализация результата ---
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.title("Binary Mask")
    plt.imshow(binary_mask, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.title("Segmented Image")
    plt.imshow(segmented)
    plt.axis('off')

    plt.tight_layout()
    plt.show()


'''
        elif number == 9:
            return '''
            import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
# Вывод для colab
from google.colab.patches import cv2_imshow

img = cv2.imread('.jpeg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Преобразуем в 2D массив пикселей
pix = img.reshape((-1, 3))
k = 2 # Число кластеров (K)
# Кластеризация
kmeans = KMeans(n_clusters=k)
kmeans.fit(pix)

# Получаем лейблы и центроиды
labels = kmeans.labels_
centroids = kmeans.cluster_centers_

# Каждый пиксель относим к отдельному центроиду
seg_image = centroids[labels].reshape(img.shape)

plt.subplot(121)
plt.imshow(img)
plt.title('Начальное')
plt.subplot(122)
plt.imshow(seg_image.astype(np.uint8))
plt.title('После кластеризации')
plt.show()
'''
        elif number == 10:
            return '''
            class CustomImageDataset:
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = []
        self.image_paths = []
        self.labels = []

        self.classes = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}

        for cls_name in self.classes:
            cls_dir = os.path.join(root_dir, cls_name)
            for file_name in os.listdir(cls_dir):
                if file_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.image_paths.append(os.path.join(cls_dir, file_name))
                    self.labels.append(self.class_to_idx[cls_name])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path)
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label

    def get_class_name(self, label):
        return self.classes[label]

if __name__ == "__main__":
    dataset = CustomImageDataset(root_dir="/content/drive/MyDrive/Финашка/Машинное зрение/4 сем/dataset")
    print(f"Размер датасета: {len(dataset)} изображений")
    print(f"Классы: {dataset.classes}")

    try:
        img, label = dataset[200]
        plt.imshow(img)
        plt.title(f"Класс: {dataset.get_class_name(label)} (id={200})")
        plt.axis('off')
        plt.show()
    except IndexError:
        print(f"Изображение с id = 200 не найдено (датасет содержит {len(dataset)} изображений)")
'''
        elif number == 11:
            return '''
import time

def median_filter(img, size):
    height, width = img.shape
    radius = size // 2
    filtered_img = np.zeros_like(img, dtype=np.uint8)

    for row in range(height):
        for col in range(width):
            r_start = max(row - radius, 0)
            r_end = min(row + radius + 1, height)
            c_start = max(col - radius, 0)
            c_end = min(col + radius + 1, width)

            window = img[r_start:r_end, c_start:c_end]
            filtered_img[row, col] = int(np.median(window))

    return filtered_img


gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

start_time = time.perf_counter()
res = median_filter(gray_image, 5)
end_time = time.perf_counter()
print(f"Время выполнения: {end_time - start_time:.6f} секунд")

cv2_imshow(res)
'''
        elif number == 107:
            return '''
import os
import io
import gdown
import torch
from torch.utils.data import DataLoader, Subset, random_split
from torchvision import datasets, transforms
from google.oauth2 import service_account
from PIL import Image
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import time
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomRotation(30),
    transforms.ToTensor()
])

dataset = datasets.ImageFolder(root='/content/drive/MyDrive/Финашка/Машинное зрение/4 сем/dataset', transform=transform)
train_size = int(0.7 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

num_images = len(dataset)
num_classes = len(dataset.classes)
image_size = dataset[0][0].size()

class CNN(nn.Module):
    def __init__(self, num_classes):
        super(CNN, self).__init__()

        # Сверточные слои
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1)  # Вход 3x300x300, выход 32x150x150
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1) # Вход 32x300x300, выход 64x75x75
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)       # Уменьшение разрешения в 2 раза
        self.sigmoid = nn.Sigmoid()

        # Полносвязные слои
        self.fc1 = nn.Linear(16384, 128)
        self.fc2 = nn.Linear(128, num_classes)

        # Активации и нормализация
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)  # Разворачиваем в вектор
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.sigmoid(self.fc2(x))
        return x


# Установка параметров
batch_size = 32
num_epochs = 10
learning_rate = 0.001

model = CNN(num_classes).to(device)
criterion = nn.BCELoss()  #nn.CrossEntropyLoss()(для многоклассовой)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

def calculate_accuracy(loader, model):
    correct = 0
    total = 0
    model.eval()
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total

train_losses = []
train_accuracies = []
test_accuracies = []
epoch_times = []

for epoch in range(num_epochs):
    start_time = time.time()
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Statistics
        running_loss += loss.item() * images.size(0)

    # Среднее значение функции потерь за эпоху
    train_loss = running_loss / len(train_loader.dataset)
    train_losses.append(train_loss)

    # Точность на обучающем множестве
    train_acc = calculate_accuracy(train_loader, model)
    train_accuracies.append(train_acc)

    # Точность на тестовом множестве
    test_acc = calculate_accuracy(test_loader, model)
    test_accuracies.append(test_acc)

    # Время выполнения эпохи
    epoch_time = time.time() - start_time
    epoch_times.append(epoch_time)

    print(f"Эпоха [{epoch+1}/{num_epochs}], Время: {epoch_time:.2f} сек, Потеря: {train_loss:.4f}, "
          f"Точность на обучении: {train_acc:.2f}%, Точность на тесте: {test_acc:.2f}%")

# Графики потерь и точности
epochs = np.arange(1, num_epochs+1)

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(epochs, train_losses, label='Train Loss')
plt.title('Изменение потерь во времени')
plt.xlabel('Эпоха')
plt.ylabel('Потеря')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(epochs, train_accuracies, label='Train Accuracy')
plt.plot(epochs, test_accuracies, label='Test Accuracy')
plt.title('Изменение точности во времени')
plt.xlabel('Эпоха')
plt.ylabel('Точность (%)')
plt.legend()

plt.show()

# Вывод количества параметров модели
model_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Количество параметров модели: {model_params}")



#Более сложные модели

class CNN_1(nn.Module):
    def __init__(self, num_classes):
        super(CNN_1, self).__init__()

        # Сверточные слои
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1)  # Вход 3x300x300, выход 32x150x150
        #self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1) # Вход 32x300x300, выход 64x75x75
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)       # Уменьшение разрешения в 2 раза

        # Полносвязные слои
        self.fc1 = nn.Linear(131072, num_classes)

        # Активации и нормализация
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        #x = self.pool(self.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)  # Разворачиваем в вектор
        #x = self.relu(self.fc1(x))
        #x = self.dropout(x)
        x = self.fc1(x)
        return x

class CNN_2(nn.Module):
    def __init__(self, num_classes):
        super(CNN_2, self).__init__()

        # Сверточные слои
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1)  # Вход 3x300x300, выход 32x150x150
        #self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1) # Вход 32x300x300, выход 64x75x75
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)       # Уменьшение разрешения в 2 раза

        # Полносвязные слои
        self.fc1 = nn.Linear(131072, 128)
        self.fc2 = nn.Linear(128, num_classes)

        # Активации и нормализация
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        #x = self.pool(self.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)  # Разворачиваем в вектор
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class CNN_3(nn.Module):
    def __init__(self, num_classes):
        super(CNN_3, self).__init__()

        # Сверточные слои
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1)  # Вход 3x300x300, выход 32x150x150
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1) # Вход 32x300x300, выход 64x75x75
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)       # Уменьшение разрешения в 2 раза

        # Полносвязные слои
        self.fc1 = nn.Linear(16384, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, num_classes)

        # Активации и нормализация
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)  # Разворачиваем в вектор
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.relu(self.fc3(x))
        x = self.dropout(x)
        x = self.fc4(x)
        return x'''

        elif number == 108:
            return '''
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader, random_split
import time
import matplotlib.pyplot as plt
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Преобразования изображений
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # VGG16 требует 224x224
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Нормализация для ImageNet
])

# Загрузка данных
dataset = datasets.ImageFolder(root='/content/drive/MyDrive/Финашка/Машинное зрение/4 сем/dataset', transform=transform)
train_size = int(0.7 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

num_classes = len(dataset.classes)

class VGG16Transfer(nn.Module):
    def __init__(self, num_classes, num_additional_layers=0):
        super(VGG16Transfer, self).__init__()

        # Загрузка предобученной VGG16
        self.vgg16 = models.vgg16(pretrained=True)

        # Замораживаем все слои
        for param in self.vgg16.parameters():
            param.requires_grad = False

        # Заменяем классификатор
        num_features = self.vgg16.classifier[6].in_features
        layers = list(self.vgg16.classifier.children())[:-1]  # Удаляем последний слой

        # Добавляем новые слои в зависимости от параметра
        if num_additional_layers > 0:
            additional_layers = []
            for i in range(num_additional_layers):
                additional_layers.append(nn.Linear(num_features, num_features//2))
                additional_layers.append(nn.ReLU(inplace=True))
                additional_layers.append(nn.Dropout(0.5))
                num_features = num_features // 2
            layers.extend(additional_layers)

        layers.append(nn.Linear(num_features, num_classes))
        self.vgg16.classifier = nn.Sequential(*layers)

    def forward(self, x):
        return self.vgg16(x)

# Параметры обучения
batch_size = 32
num_epochs = 15
learning_rate = 0.001

# Словарь для хранения результатов
results = {}

# Тестируем разное количество дополнительных слоёв
for num_layers in [0, 1, 2, 3]:
    print(f"\nTraining model with {num_layers} additional layers...")

    model = VGG16Transfer(num_classes, num_additional_layers=num_layers).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    train_losses = []
    train_accuracies = []
    test_accuracies = []

    for epoch in range(num_epochs):
        start_time = time.time()
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_loss = running_loss / len(train_loader.dataset)
        train_acc = 100 * correct / total
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)

        # Тестовая точность
        model.eval()
        test_correct = 0
        test_total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                test_total += labels.size(0)
                test_correct += (predicted == labels).sum().item()

        test_acc = 100 * test_correct / test_total
        test_accuracies.append(test_acc)

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {train_loss:.4f}, "
              f"Train Acc: {train_acc:.2f}%, Test Acc: {test_acc:.2f}%")

    results[num_layers] = {
        'train_losses': train_losses,
        'train_accuracies': train_accuracies,
        'test_accuracies': test_accuracies
    }

# Визуализация результатов
plt.figure(figsize=(15, 5))

# График потерь
plt.subplot(1, 3, 1)
for num_layers, res in results.items():
    plt.plot(res['train_losses'], label=f'{num_layers} layers')
plt.title('Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

# График точности на обучающей выборке
plt.subplot(1, 3, 2)
for num_layers, res in results.items():
    plt.plot(res['train_accuracies'], label=f'{num_layers} layers')
plt.title('Training Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.legend()

# График точности на тестовой выборке
plt.subplot(1, 3, 3)
for num_layers, res in results.items():
    plt.plot(res['test_accuracies'], label=f'{num_layers} layers')
plt.title('Test Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.legend()

plt.tight_layout()
plt.show()'''
        elif number == 109:
            return '''
import torch
import torchvision
from torchvision import transforms
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import warnings
import numpy as np

# Игнорируем предупреждения
warnings.filterwarnings("ignore")

class SSDObjDetector:
    def __init__(self, model_name='ssd300_vgg16', confidence_threshold=0.5):

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self._load_pretrained_model(model_name)
        self.confidence_threshold = confidence_threshold
        self.transform = self._get_transform()
        self.target_classes = None

        self.CLASSES = [
            '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
            'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
            'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
            'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
            'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
            'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
            'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana',
            'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut',
            'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table', 'N/A', 'N/A',
            'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
            'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
            'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
        ]

    def set_target_classes(self, class_names):

        if class_names is None:
            self.target_classes = None
            return

        invalid_classes = set(class_names) - set(self.CLASSES)
        if invalid_classes:
            raise ValueError(f"Неизвестные классы: {invalid_classes}. Доступные классы: {self.CLASSES}")

        self.target_classes = class_names

    def _load_pretrained_model(self, model_name):
        """Загрузка предобученной модели"""
        if model_name == 'ssd300_vgg16':
            model = torchvision.models.detection.ssd300_vgg16(pretrained=True)
        elif model_name == 'ssdlite320_mobilenet_v3_large':
            model = torchvision.models.detection.ssdlite320_mobilenet_v3_large(pretrained=True)
        else:
            raise ValueError(f"Unknown model name: {model_name}")

        model.eval()
        return model.to(self.device)

    def _get_transform(self):
        """Трансформации для входного изображения"""
        return transforms.Compose([transforms.ToTensor()])

    def set_confidence_threshold(self, threshold):
        self.confidence_threshold = threshold

    def detect(self, image_path, output_path=None):
        """
        Детектирование объектов на изображении с фильтрацией по целевым классам

        Args:
            image_path (str): путь к входному изображению
            output_path (str, optional): путь для сохранения результата

        Returns:
            dict: словарь с результатами детектирования
        """

        image = Image.open(image_path).convert("RGB")
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            predictions = self.model(image_tensor)

        boxes = predictions[0]['boxes'].cpu().numpy()
        labels = predictions[0]['labels'].cpu().numpy()
        scores = predictions[0]['scores'].cpu().numpy()

        keep = scores >= self.confidence_threshold
        boxes = boxes[keep]
        labels = labels[keep]
        scores = scores[keep]

        detections = []
        if self.target_classes:
            target_class_indices = [self.CLASSES.index(cls) for cls in self.target_classes]
            class_mask = np.isin(labels, target_class_indices)
            boxes = boxes[class_mask]
            labels = labels[class_mask]
            scores = scores[class_mask]

        detections = []
        for box, label, score in zip(boxes, labels, scores):
            class_name = self.CLASSES[label]

            detections.append({
                "class": class_name,
                "score": float(score),
                "bbox": [float(coord) for coord in box.tolist()]
            })

        draw = ImageDraw.Draw(image)
        for box, label, score in zip(boxes, labels, scores):
            box_coords = [(box[0], box[1]), (box[2], box[3])]
            draw.rectangle(box_coords, outline="red", width=3)
            text = f"{self.CLASSES[label]}: {score:.2f}"
            draw.text((box[0], box[1]), text, fill="red")

        if output_path:
            image.save(output_path)
        else:
            plt.imshow(image)
            plt.axis('off')
            plt.show()

        return {
            'boxes': boxes,
            'labels': [self.CLASSES[l] for l in labels],
            'scores': scores
        }, detections


#Вывод рандомных 5 изображений

import os
import random
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw

def display_random_detections(detector, dataset_path, num_images=5):
    """
    Отображает случайные изображения из датасета с детекциями

    Args:
        detector (SSDObjDetector): инициализированный детектор объектов
        dataset_path (str): путь к папке с изображениями
        num_images (int): количество изображений для отображения
    """
    # Получаем список всех изображений в датасете
    image_files = [f for f in os.listdir(dataset_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    # Выбираем случайные изображения
    selected_images = random.sample(image_files, min(num_images, len(image_files)))

    # Создаем grid для отображения
    plt.figure(figsize=(15, 10))

    for i, img_file in enumerate(selected_images, 1):
        img_path = os.path.join(dataset_path, img_file)

        # Детектируем объекты
        results = detector.detect(img_path)

        # Загружаем изображение для отрисовки
        image = Image.open(img_path).convert("RGB")
        draw = ImageDraw.Draw(image)

        # Рисуем bounding boxes и подписи
        for box, label, score in zip(results['boxes'], results['labels'], results['scores']):
            # Рисуем прямоугольник
            draw.rectangle([(box[0], box[1]), (box[2], box[3])], outline="red", width=2)
            # Добавляем текст с классом и уверенностью
            text = f"{label}: {score:.2f}"
            draw.text((box[0], box[1]), text, fill="red")

        # Отображаем изображение
        plt.subplot(2, 3, i) if num_images > 3 else plt.subplot(1, num_images, i)
        plt.imshow(image)
        plt.title(f"Image: {img_file}")
        plt.axis('off')

    plt.tight_layout()
    plt.show()

# Пример использования:
if __name__ == "__main__":
    # Инициализация детектора
    detector = SSDObjDetector(model_name='ssd300_vgg16', confidence_threshold=0.5)

    # Укажите путь к вашему датасету
    dataset_path = "path/to/your/dataset"  # Замените на реальный путь

    # Отображаем 5 случайных изображений
    display_random_detections(detector, dataset_path, num_images=5)'''
        elif number == 15:
            return '''
from ultralytics import YOLO
import cv2

model = YOLO('yolov8n.pt')
# Путь к исходному и выходному видео
input_video = '/content/IMG_7235.MP4'
output_video = 'output_video_detected.mp4'

# Открываем видео
cap = cv2.VideoCapture(input_video)
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Создаем VideoWriter для сохранения
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

# Обрабатываем кадры
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, conf=0.5)  # conf - порог уверенности

    annotated_frame = results[0].plot()  # Автоматическая отрисовка bbox

    out.write(annotated_frame)

cap.release()
out.release()'''
        elif number == 16:
            return '''
!pip install torch torchvision segmentation-models-pytorch

import torch.nn as nn
from torchvision.datasets import VOCSegmentation
import torch
import torchvision.transforms as transforms
from torchvision.datasets import VOCSegmentation
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp
from tqdm import tqdm


transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

train_dataset = VOCSegmentation(root='./data', year='2012', image_set='train', download=True, transform=transform, target_transform=transform)
val_dataset = VOCSegmentation(root='./data', year='2012', image_set='val', download=True, transform=transform, target_transform=transform)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

model = smp.Unet(
    encoder_name="vgg16",
    in_channels=3,
    classes=21,
)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

num_epochs = 10

for epoch in tqdm(range(num_epochs)):
    model.train()
    running_loss = 0.0
    for images, masks in train_loader:
        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks.squeeze(1).long())
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader)}")

    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for images, masks in val_loader:
            images = images.to(device)
            masks = masks.to(device)
            outputs = model(images)
            loss = criterion(outputs, masks.squeeze(1).long())
            val_loss += loss.item()

    print(f"Validation Loss: {val_loss/len(val_loader)}")'''
        elif number == 17:
            return '''
import torch
import torchvision
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.transforms import functional as F
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import cv2
import colorsys
import pandas as pd

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def visualize_prediction(image, prediction, score_threshold=0.5):
    # Перенос тензоров на CPU и конвертация в numpy
    prediction = {k: v.cpu() for k, v in prediction.items()}

    # Фильтрация по порогу уверенности
    keep = prediction['scores'] > score_threshold
    masks = prediction['masks'][keep].squeeze(1).numpy() > 0.5
    boxes = prediction['boxes'][keep].int().numpy()
    labels = prediction['labels'][keep].numpy()
    scores = prediction['scores'][keep].numpy()

    image_np = np.array(image)

    # Специально подобранные цвета для часто встречающихся классов
    class_colors = {
        1: (0, 255, 0),    # person - зеленый
        2: (255, 0, 0),     # bicycle - синий
        3: (0, 0, 255),     # car - красный
    }

    # Для остальных классов используем автоматическую генерацию цвета
    max_class = max(COCO_CLASSES.keys()) if hasattr(COCO_CLASSES, 'keys') else len(COCO_CLASSES)
    for i in range(1, max_class + 1):
        if i not in class_colors:
            # Генерируем уникальный цвет на основе хеша названия класса
            class_name = COCO_CLASSES[i] if i < len(COCO_CLASSES) else str(i)
            hue = hash(class_name) % 360
            class_colors[i] = tuple(int(c * 255) for c in colorsys.hsv_to_rgb(hue/360, 0.8, 0.8))

    # Конвертируем изображение в BGR для OpenCV
    result = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

    # Рисуем маски и bounding boxes
    for i, (mask, box, label, score) in enumerate(zip(masks, boxes, labels, scores)):
        class_name = COCO_CLASSES[label]
        color = class_colors.get(label, (255, 255, 255))  # Белый по умолчанию

        # Рисуем маску (полупрозрачную)
        colored_mask = np.zeros_like(result)
        colored_mask[mask] = color
        result = cv2.addWeighted(result, 1, colored_mask, 0.3, 0)

        # Рисуем bounding box (более толстый для важных классов)
        thickness = 3 if class_name in ['person', 'car', 'bicycle'] else 2
        cv2.rectangle(result, (box[0], box[1]), (box[2], box[3]), color, thickness)

        # Формируем текст подписи
        text = f"{class_name} {score:.2f}"

        # Вычисляем позицию текста
        text_y = box[1] - 10 if box[1] - 10 > 10 else box[1] + 20
        (text_width, text_height), _ = cv2.getTextSize(
            text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

        # Рисуем подложку для текста
        cv2.rectangle(result,
                     (box[0], text_y - text_height - 4),
                     (box[0] + text_width, text_y + 2),
                     color, -1)

        # Рисуем текст
        cv2.putText(result, text,
                   (box[0], text_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                   (255, 255, 255), 1, cv2.LINE_AA)

    # Конвертируем обратно в RGB для matplotlib
    result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)

    # Отображаем результат
    plt.figure(figsize=(12, 8))
    plt.imshow(result)
    plt.axis('off')
    plt.show()


# Список классов COCO
COCO_CLASSES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana',
    'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut',
    'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table', 'N/A', 'N/A',
    'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book', 'clock',
    'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
] # N/A - Это placeholder-ы для неиспользуемых индексов (в оригинальном датасете COCO есть "пропуски" в нумерации категорий).

MY_CLASSES = ['person', 'bicycle']
class_ids = [COCO_CLASSES.index(cls) for cls in MY_CLASSES]

# Загрузка предобученной модели Mask R-CNN
model = maskrcnn_resnet50_fpn(pretrained=True) # С предобученными весами на COCO
model.eval()  # Переводим модель в режим оценки

# 2. Загрузка и подготовка изображения
def load_image(image_path):
    image = Image.open(image_path).convert("RGB")
    image_tensor = F.to_tensor(image).unsqueeze(0).to(device) # сразу добавляем размерность батча
    return image, image_tensor

image_path = "/content/IMG_7294.JPG"  # Путь к изображению
original_image, image_tensor = load_image(image_path)

# 3. Выполнение предсказания
with torch.no_grad():
    predictions = model(image_tensor)

    # Фильтруем предсказания, оставляя только person и car
    masks = []
    boxes = []
    labels = []
    scores = []

    for i in range(len(predictions[0]['labels'])):
        label = predictions[0]['labels'][i].item()
        if label in class_ids:
            masks.append(predictions[0]['masks'][i])
            boxes.append(predictions[0]['boxes'][i])
            labels.append(predictions[0]['labels'][i])
            scores.append(predictions[0]['scores'][i])

    # Создаем новый словарь предсказаний только с нужными классами
    filtered_predictions = [{
        'masks': torch.stack(masks) if masks else torch.tensor([]),
        'boxes': torch.stack(boxes) if boxes else torch.tensor([]),
        'labels': torch.stack(labels) if labels else torch.tensor([]),
        'scores': torch.stack(scores) if scores else torch.tensor([])
    }]

# все классы
visualize_prediction(original_image, predictions[0], score_threshold=0.8)
# 'person', 'bicycle', 'car'
visualize_prediction(original_image, filtered_predictions[0], score_threshold=0.8)


def class_statistics(predictions, score_threshold=0.5):

    # Получаем предсказания и фильтруем по порогу
    pred = predictions[0]
    keep = pred['scores'] > score_threshold
    filtered_labels = pred['labels'][keep].cpu().numpy()

    # Считаем количество объектов по классам
    unique_classes, counts = np.unique(filtered_labels, return_counts=True)

    stats = []
    for class_id, count in zip(unique_classes, counts):
        class_name = COCO_CLASSES[class_id]
        stats.append({'Класс': class_name, 'Количество': count})

    return pd.DataFrame(stats), counts.sum()

df, res = class_statistics(predictions, score_threshold=0.8)
print(f'Всего объектов: {res}')
df'''
        elif number == 18:
            return '''
import os
import cv2
import torch
import torchvision
import numpy as np
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.transforms import functional as F
from PIL import Image
import json
from tqdm import tqdm

# Настройки
video_path = 'input_video.mp4'
output_dir = 'output'
mask_dir = os.path.join(output_dir, 'masks')
os.makedirs(mask_dir, exist_ok=True)

coco_output = {
    "images": [],
    "annotations": [],
    "categories": []
}

# Классы COCO
COCO_CLASSES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana',
    'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut',
    'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table', 'N/A', 'N/A',
    'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book', 'clock',
    'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

class_ids = [1, 2]  # person и bicycle
category_id_map = {cls_id: idx + 1 for idx, cls_id in enumerate(class_ids)}  # COCO category_id → local

# Категории для COCO JSON
for coco_id in class_ids:
    coco_output["categories"].append({
        "id": category_id_map[coco_id],
        "name": COCO_CLASSES[coco_id]
    })

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = maskrcnn_resnet50_fpn(pretrained=True).to(device)
model.eval()

cap = cv2.VideoCapture(video_path)
frame_idx = 0
annotation_id = 1
image_id = 1

with torch.no_grad():
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Преобразование кадра
        image_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        image_tensor = F.to_tensor(image_pil).unsqueeze(0).to(device)

        predictions = model(image_tensor)[0]

        # Фильтрация по score и классу
        keep = (predictions['scores'] > 0.5) & \
               torch.tensor([lbl in class_ids for lbl in predictions['labels']]).to(device)

        masks = predictions['masks'][keep].squeeze(1).cpu().numpy()
        boxes = predictions['boxes'][keep].cpu().numpy().astype(int)
        labels = predictions['labels'][keep].cpu().numpy()

        # Добавляем информацию об изображении
        height, width = frame.shape[:2]
        coco_output["images"].append({
            "id": image_id,
            "file_name": f"frame_{frame_idx:05d}.jpg",
            "width": width,
            "height": height
        })

        for i, (mask, box, label) in enumerate(zip(masks, boxes, labels)):
            # Сохраняем маску
            mask_bin = (mask > 0.5).astype(np.uint8) * 255
            mask_filename = f"mask_{frame_idx:05d}_{i}.png"
            cv2.imwrite(os.path.join(mask_dir, mask_filename), mask_bin)

            # COCO аннотация
            y_indices, x_indices = np.where(mask_bin)
            if len(x_indices) < 6:  # polygon должен иметь хотя бы 3 точки
                continue

            segmentation = [x.tolist() for x in np.stack([x_indices, y_indices], axis=1).flatten().reshape(-1, 2)]
            coco_output["annotations"].append({
                "id": annotation_id,
                "image_id": image_id,
                "category_id": category_id_map[label],
                "bbox": [int(box[0]), int(box[1]), int(box[2] - box[0]), int(box[3] - box[1])],
                "area": int(np.sum(mask_bin > 0)),
                "iscrowd": 0,
                "segmentation": [np.array(segmentation).flatten().tolist()]
            })
            annotation_id += 1

        frame_idx += 1
        image_id += 1

# Сохраняем COCO аннотации
with open(os.path.join(output_dir, 'annotations.json'), 'w') as f:
    json.dump(coco_output, f)

cap.release()
print("Обработка завершена.")'''
        elif number == 19:
            return '''
!pip install numpy opencv-python Pillow onemetric
!pip install git+https://github.com/ifzhang/ByteTrack.git --no-deps

!pip install ultralytics

import cv2
import numpy as np
from datetime import datetime
from ultralytics import YOLO
from google.colab.patches import cv2_imshow

# Своя реализация IoU на NumPy
def iou_batch(bboxes1, bboxes2):
    x11, y11, x12, y12 = np.split(bboxes1, 4, axis=1)
    x21, y21, x22, y22 = np.split(bboxes2, 4, axis=1)

    xA = np.maximum(x11, x21.T)
    yA = np.maximum(y11, y21.T)
    xB = np.minimum(x12, x22.T)
    yB = np.minimum(y12, y22.T)

    interArea = np.maximum(0, xB - xA) * np.maximum(0, yB - yA)
    boxAArea = (x12 - x11) * (y12 - y11)
    boxBArea = (x22 - x21) * (y22 - y21)

    iou = interArea / (boxAArea + boxBArea.T - interArea)
    return iou

class SimpleByteTracker:
    def __init__(self, track_thresh=0.5, match_thresh=0.8, max_misses=5):
        self.track_thresh = track_thresh
        self.match_thresh = match_thresh
        self.max_misses = max_misses  # Макс. число кадров без обновления
        self.tracks = []
        self.next_id = 1

    def update(self, detections, img_size):
        valid_dets = [d for d in detections if d[4] >= self.track_thresh]
        matched = set()
        matched_tracks = set()

        # Увеличиваем счётчик пропусков для всех треков
        for track in self.tracks:
            track['misses'] += 1

        if self.tracks and valid_dets:
            track_boxes = np.array([t['bbox'] for t in self.tracks])
            det_boxes = np.array([d[:4] for d in valid_dets])

            iou_matrix = iou_batch(track_boxes, det_boxes)

            for i, track in enumerate(self.tracks):
                best_match = np.argmax(iou_matrix[i])
                if iou_matrix[i, best_match] > self.match_thresh:
                    track['bbox'] = valid_dets[best_match][:4]
                    track['misses'] = 0  # Сброс счётчика
                    matched.add(best_match)
                    matched_tracks.add(i)

        # Удаляем треки, которые долго не обновлялись
        self.tracks = [
            t for t in self.tracks
            if t['misses'] <= self.max_misses
        ]

        # Добавляем новые треки
        for i, det in enumerate(valid_dets):
            if i not in matched:
                self.tracks.append({
                    'id': self.next_id,
                    'bbox': det[:4],
                    'score': det[4],
                    'misses': 0  # Инициализация счётчика
                })
                self.next_id += 1

        return self.tracks



# Переинициализация видеозахвата и записи для объединенных задач
cap = cv2.VideoCapture('/content/pedestrian.mp4')  # Или путь к видеофайлу

# Проверка успешности открытия видеофайла
if not cap.isOpened():
    print("Ошибка: Не удалось открыть видеофайл.")
    # Выход из скрипта, если видео не открывается
    exit()

# Инициализация видеозаписи результата
output_filename = f"tracking_result_combined_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Кодек для записи видео
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # Ширина кадра
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # Высота кадра
fps = int(cap.get(cv2.CAP_PROP_FPS))  # Частота кадров

out = cv2.VideoWriter(output_filename, fourcc, fps, (frame_width, frame_height))

# Проверка успешности инициализации видеозаписи
if not out.isOpened():
    print("Ошибка: Не удалось создать видеозапись.")
    cap.release()  # Освобождение видеозахвата при ошибке
    exit()

# Определение области интереса (ROI) для задачи 1
roi = [100, 100, 900, 700]  # Пример: x1=100, y1=100, x2=900, y2=700

# Длина истории для задачи 3 (визуальный след) и задачи 4 (расчет скорости)
HISTORY_LENGTH_TRAIL = 50  # Для визуального следа
HISTORY_LENGTH_SPEED = 5   # Для расчета скорости (усреднение по нескольким кадрам)

# Модифицированный SimpleByteTracker с историей
class SimpleByteTrackerCombined(SimpleByteTracker):
    def __init__(self, track_thresh=0.5, match_thresh=0.8, max_misses=10, history_length_trail=HISTORY_LENGTH_TRAIL):
        super().__init__(track_thresh, match_thresh, max_misses)
        self.history_length_trail = history_length_trail
        # Словари для хранения истории:
        self._history_trail = {}  # Для визуального следа
        self._history_speed = {}  # Для расчета скорости (последние центроиды)

    def update(self, detections, img_size):
        # Фильтрация детекций по порогу уверенности
        valid_dets = [d for d in detections if d[4] >= self.track_thresh]
        matched = set()
        matched_tracks_indices = set()

        # Обновление счетчика пропусков и истории для всех треков
        current_centroids = {}  # Текущие центроиды для этого кадра

        for i, track in enumerate(self.tracks):
            track['misses'] += 1

            # Расчет текущего центроида
            x1, y1, x2, y2 = track['bbox']
            centroid = (int((x1 + x2) // 2), int((y1 + y2) // 2))
            current_centroids[track['id']] = centroid

            # Обновление истории для визуального следа
            if track['id'] not in self._history_trail:
                self._history_trail[track['id']] = []
            self._history_trail[track['id']].append(centroid)
            # Ограничение длины истории
            self._history_trail[track['id']] = self._history_trail[track['id']][-self.history_length_trail:]

            # Обновление истории для расчета скорости (последние 2 точки)
            if track['id'] not in self._history_speed:
                self._history_speed[track['id'] = []
            self._history_speed[track['id']].append(centroid)
            if len(self._history_speed[track['id']]) > HISTORY_LENGTH_SPEED:
                self._history_speed[track['id']].pop(0)

        # Сопоставление треков и детекций
        if self.tracks and valid_dets:
            track_boxes = np.array([t['bbox'] for t in self.tracks])
            det_boxes = np.array([d[:4] for d in valid_dets])

            if det_boxes.size > 0 and track_boxes.size > 0:
                iou_matrix = iou_batch(track_boxes, det_boxes)

                for i, track in enumerate(self.tracks):
                    best_iou = np.max(iou_matrix[i])
                    best_match_index = np.argmax(iou_matrix[i])

                    if best_iou > self.match_thresh:
                        track['bbox'] = valid_dets[best_match_index][:4]
                        track['score'] = valid_dets[best_match_index][4]
                        track['misses'] = 0
                        matched.add(best_match_index)
                        matched_tracks_indices.add(i)

        # Обработка несопоставленных треков и детекций
        unmatched_tracks_indices = set(range(len(self.tracks))) - matched_tracks_indices
        unmatched_dets_indices = set(range(len(valid_dets))) - matched

        # Добавление новых треков для несопоставленных детекций
        for i in unmatched_dets_indices:
            det = valid_dets[i]
            new_track_id = self.next_id
            self.next_id += 1

            x1, y1, x2, y2 = det[:4]
            centroid = (int((x1 + x2) // 2), int((y1 + y2) // 2))

            self.tracks.append({
                'id': new_track_id,
                'bbox': det[:4],
                'score': det[4],
                'misses': 0
            })
            # Инициализация истории для нового трека
            self._history_trail[new_track_id] = [centroid]
            self._history_speed[new_track_id] = [centroid]

        # Удаление треков с превышением максимального числа пропусков
        self.tracks = [t for t in self.tracks if t['misses'] <= self.max_misses]
        # Сохранение только актуальной истории
        active_ids = {t['id'] for t in self.tracks}
        self._history_trail = {k: v for k, v in self._history_trail.items() if k in active_ids}
        self._history_speed = {k: v for k, v in self._history_speed.items() if k in active_ids}

        return self.tracks

# Инициализация трекера
tracker = SimpleByteTrackerCombined(max_misses=10, history_length_trail=HISTORY_LENGTH_TRAIL)

# Основной цикл обработки видео
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Задача 1: Применение ROI
    x1_roi = max(0, roi[0])
    y1_roi = max(0, roi[1])
    x2_roi = min(frame_width, roi[2])
    y2_roi = min(frame_height, roi[3])

    if x2_roi > x1_roi and y2_roi > y1_roi:
        frame_roi = frame[y1_roi:y2_roi, x1_roi:x2_roi]

        # Детекция объектов в ROI
        results = model(frame_roi, conf=0.5)
        detections = []
        for result in results:
            for box, conf, cls_id in zip(result.boxes.xyxy.cpu().numpy(),
                                       result.boxes.conf.cpu().numpy(),
                                       result.boxes.cls.cpu().numpy()):
                # Преобразование координат в исходную систему
                detections.append([box[0]+x1_roi, box[1]+y1_roi,
                                 box[2]+x1_roi, box[3]+y1_roi, conf, cls_id])
    else:
        detections = []

    # Рисование ROI
    cv2.rectangle(frame, (x1_roi, y1_roi), (x2_roi, y2_roi), (255, 0, 0), 2)

    # Обновление трекера
    tracks = tracker.update(detections, (frame_width, frame_height))

    # Задача 4: Расчет и отображение метрик
    num_active_tracks = len(tracks)
    speeds = {}
    for track in tracks:
        history = tracker._history_speed.get(track['id'], [])
        if len(history) >= 2:
            total_dist = sum(np.sqrt((history[i][0]-history[i-1][0])**2 +
                           (history[i][1]-history[i-1][1])**2)
                           for i in range(1, len(history)))
            speeds[track['id']] = total_dist / (len(history)-1)
        else:
            speeds[track['id']] = 0

    # Отображение количества активных треков
    cv2.putText(frame, f"Активных треков: {num_active_tracks}", (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # Визуализация результатов
    for track in tracks:
        x1, y1, x2, y2 = map(int, track['bbox'])
        track_id = track['id']

        # Задача 2: Цвет bbox в зависимости от пропусков
        miss_ratio = min(track['misses'] / tracker.max_misses, 1.0)
        color = (0, int(255*(1-miss_ratio)), int(255*miss_ratio))
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # Отображение ID и скорости
        text_y = max(y1 - 10, 20)  # Чтобы текст не выходил за границу
        cv2.putText(frame, f"ID: {track_id} Скорость: {speeds[track_id]:.1f}",
                   (x1, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Задача 3: Визуализация истории (зеленый след)
        history = tracker._history_trail.get(track_id, [])
        for i, point in enumerate(history):
            alpha = i / len(history)  # Прозрачность по "возрасту" точки
            cv2.circle(frame, point, 2, (0, int(255*alpha), 0), -1)

    # Запись кадра
    out.write(frame)

# Освобождение ресурсов
cap.release()
out.release()
print(f"Результат сохранен как: {output_filename}")'''
        elif number == 20:
            return '''
!pip install ikomia
from google.colab.patches import cv2_imshow
import matplotlib.pyplot as plt
from ikomia.dataprocess.workflow import Workflow # Создаётся рабочий процесс (Workflow), который будет объединять
# разные алгоритмы обработки в локальной версии для вывода
# from ikomia.utils.displayIO import display
import cv2


input_video_path = '/content/pedestrian.mp4'
output_video_path = 'deepsort_output_video.avi'

# Init your workflow
wf = Workflow()

# Add object detection algorithm
detector = wf.add_task(name="infer_yolo_v7", auto_connect=True)
detector.set_parameters({
    # "model_name": "yolov7",  # можно выбрать другую модель
    # "conf_thres": "0.5",     # порог уверенности детекции
    # "iou_thres": "0.45",     # порог IoU для NMS
    # "input_size": "640"      # размер входного изображения
})

# Add deepsort tracking algorithm
tracking = wf.add_task(name="infer_deepsort", auto_connect=True)

tracking.set_parameters({
    "categories": "all", # отслеживать все обнаруженные классы.
    "conf_thres": "0.5", # игнорировать объекты с уверенностью < 50%.
    "max_age": "50",        # максимальное количество пропущенных кадров
    "min_hits": "3",        # минимальное количество детекций для инициализации трека
    "iou_threshold": "0.2",  # порог IoU для ассоциации
    "cosine_threshold": "0.2",  # порог косинусного расстояния для appearance-метрики (обычно оптимальное 0.3)
    "nn_budget": "100",      # бюджет для хранения appearance-фич
    "use_cuda": "True"       # использование GPU
})

# Open the video file
stream = cv2.VideoCapture(input_video_path)
if not stream.isOpened():
    print("Error: Could not open video.")
    exit()

# Get video properties for the output
frame_width = int(stream.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(stream.get(cv2.CAP_PROP_FRAME_HEIGHT))
frame_rate = stream.get(cv2.CAP_PROP_FPS)

# Define the codec and create VideoWriter object
# The 'XVID' codec is widely supported and provides good quality
fourcc = cv2.VideoWriter_fourcc(*'XVID') # сохраняет результат в новый файл ('XVID' — кодек для AVI).
out = cv2.VideoWriter(output_video_path, fourcc, frame_rate, (frame_width, frame_height))

frame_idx = 0
while True:
    # Read image from stream
    ret, frame = stream.read() # читаем кадр
    frame_idx += 1
    # Test if the video has ended or there is an error
    if not ret:
        print("Info: End of video or error.")
        break

    # Run the workflow on current frame
    wf.run_on(array=frame) # передаём кадр в YOLO + DeepSORT

    # Get results
    image_out = tracking.get_output(0) # получаем кадр с разметкой (Изображение с визуализацией (bounding boxes + ID треков))
    obj_detect_out = tracking.get_output(1) # Структурированные данные о детекциях (метаинформация)


    object_counts = {}
    if obj_detect_out and obj_detect_out.get_objects():
        for detection in obj_detect_out.get_objects():
            class_name = detection.label # Получаем имя класса
            if class_name not in object_counts:
                object_counts[class_name] = 0
            object_counts[class_name] += 1

    # Вывод числа объектов каждого класса (или другие действия с object_counts)
    print(f"Кадр {frame_idx}: Объекты в кадре - {object_counts}")
    # Convert the result to BGR color space for displaying
    img_out = image_out.get_image_with_graphics(obj_detect_out)
    img_res = cv2.cvtColor(img_out, cv2.COLOR_RGB2BGR)

    # Save the resulting frame
    out.write(img_out)

     # Display
    # plt.imshow(img_res)

    # Press 'q' to quit the video processing
    # if cv2.waitKey(1) & 0xFF == ord('q'):
    #     break

# After the loop release everything
stream.release()
out.release()'''
        elif number == 21:
            return '''
!pip install filterpy
import numpy as np
import cv2
import torch
import torchvision
from filterpy.kalman import KalmanFilter
from collections import defaultdict
from scipy.optimize import linear_sum_assignment
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Класс для реализации Kalman Filter
class KalmanBoxTracker(object):
    """
    Этот класс представляет внутреннее состояние отдельных отслеживаемых объектов, наблюдаемых как ограничивающая рамка.
    """
    count = 0

    def __init__(self, bbox):
        """
        Инициализирует трекер с указанной ограничивающей рамкой.
        """
        # Определяем модель фильтра Калмана с 7 параметрами состояния и 4 параметрами измерения
        # dim_x=7 имеется в виду [x, y, w, h] — координаты центра bbox (x,y), ширину (w) и высоту (h) и
        # [vx, vy, vw, vh] — скорости изменения этих параметров (но в коде используется только vx, vy, см. матрицу F).
        # для полного описания надо бы 8 параметров, но в SORT 6 параметров + 7-й параметр  vh и vw объединены в один параметр
        # (предполагается, что они равны или слабо влияют на трекинг).
        # На практике vh часто игнорируется, и остаётся vw (отсюда dim_x=7)
        # 6 бы тоже работало, но в SORT принято 7

        # dim_z=4 В фильтре Калмана для трекинга объектов dim_z=4 означает,
        # что вектор измерений (т.е. данные, которые поступают от детектора) состоит из 4 параметров [x, y, w, h]
        # Вектор состояния x имеет размерность 7 ([x, y, w, h, vx, vy, vw]),
        # но измерения (z) включают только наблюдаемые параметры — координаты и размеры (x, y, w, h).
        # Скорости (vx, vy, vw) нельзя измерить напрямую — они вычисляются фильтром Калмана на
        # основе изменений координат и размеров между кадрами.

        self.kf = KalmanFilter(dim_x=7, dim_z=4)

        # Матрица перехода состояния (предполагаем постоянную скорость)
        # Матрица F описывает, как состояние объекта изменяется от кадра к кадру. Для модели с dim_x=7 имеем:
        self.kf.F = np.array([
          [1,0,0,0,1,0,0],  # x_new = x_prev + vx*dt  В SORT обычно предполагается dt = 1
          [0,1,0,0,0,1,0],  # y_new = y_prev + vy*dt
          [0,0,1,0,0,0,1],  # w_new = w_prev + vw*dt
          [0,0,0,1,0,0,0],  # h_new = h_prev (vh игнорируется)
          [0,0,0,0,1,0,0],  # vx_new = vx_prev
          [0,0,0,0,0,1,0],  # vy_new = vy_prev
          [0,0,0,0,0,0,1]   # vw_new = vw_prev
        ])
        # Строки 1-4: Обновление координат и размеров:
        # x и y меняются на vx и vy (предполагается движение с постоянной скоростью).
        # w меняется на vw, а h остаётся неизменным (или меняется слабо — поэтому в F нет vh).
        # Строки 5-7: Скорости (vx, vy, vw) остаются постоянными (диагональные 1).

        # Матрица измерения
        # Показывает, какие параметры состояния видны в измерениях. Здесь — только [x, y, w, h] (без скоростей).
        self.kf.H = np.array([
            [1,0,0,0,0,0,0],  # измеряем x: z_x = 1*x + 0*y + 0*w + 0*h + 0*vx + 0*vy + 0*vw = x
            [0,1,0,0,0,0,0],  # измеряем y: аналогично, z_y = y
            [0,0,1,0,0,0,0],  # измеряем w: аналогично
            [0,0,0,1,0,0,0]   # измеряем h: аналогично
        ])

        # Ковариация измерения (шум)
        self.kf.R[2:,2:] *= 10.

        # Ковариация состояния (начальная)
        self.kf.P[4:,4:] *= 1000. # Даем высокую неопределенность неизмеряемым начальным скоростям
        self.kf.P *= 10.

        # Шум процесса
        self.kf.Q[-1,-1] *= 0.01
        self.kf.Q[4:,4:] *= 0.01

        # Инициализация состояния
        self.kf.x[:4] = self.convert_bbox_to_z(bbox)

        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1

        self.history = []
        self.hits = 0
        self.hit_streak = 0
        self.age = 0
        self.time_since_update = 0

    def update(self, bbox):
        """
        Обновляет состояние трекера с наблюдаемой ограничивающей рамкой.
        """
        self.time_since_update = 0
        self.history = []
        self.hits += 1
        self.hit_streak += 1
        self.kf.update(self.convert_bbox_to_z(bbox))

    def predict(self):
        """
        Продвигает состояние вектора состояния и возвращает предсказанную ограничивающую рамку.
        """
        if (self.kf.x[6] + self.kf.x[2]) <= 0:
            self.kf.x[6] *= 0.0

        self.kf.predict()
        self.age += 1

        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1

        self.history.append(self.convert_x_to_bbox(self.kf.x))
        return self.history[-1]

    def get_state(self):
        """
        Возвращает текущую ограничивающую рамку.
        """
        return self.convert_x_to_bbox(self.kf.x)

    @staticmethod
    def convert_bbox_to_z(bbox):
        """
        Преобразует ограничивающую рамку в формат (x,y,s,r), где x,y - центр, s - масштаб/площадь, r - соотношение сторон.
        """
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        x = bbox[0] + w/2.
        y = bbox[1] + h/2.
        s = w * h
        r = w / float(h)
        return np.array([x, y, s, r]).reshape((4, 1))

    @staticmethod
    def convert_x_to_bbox(x, score=None):
        """
        Преобразует состояние (x,y,s,r) в ограничивающую рамку (x1,y1,x2,y2).
        """
        w = np.sqrt(x[2] * x[3])
        h = x[2] / w

        if score is None:
            return np.array([x[0]-w/2., x[1]-h/2., x[0]+w/2., x[1]+h/2.]).reshape((1,4))
        else:
            return np.array([x[0]-w/2., x[1]-h/2., x[0]+w/2., x[1]+h/2., score]).reshape((1,5))
# Функция для вычисления IoU (Intersection over Union)
def iou(box1, box2):
    """
    Вычисляет пересечение по объединению между двумя ограничивающими рамками.
    """
    # Определяем координаты пересечения
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    # Вычисляем площадь пересечения
    intersection = max(0, x2 - x1 + 1) * max(0, y2 - y1 + 1)

    # Вычисляем площади обеих рамок
    box1_area = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
    box2_area = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)

    # Вычисляем IoU
    iou = intersection / float(box1_area + box2_area - intersection)

    return iou

# Функция для ассоциации данных
def associate_detections_to_trackers(detections, trackers, iou_threshold=0.3):
    """
    Назначает обнаружения отслеживаемым объектам (оба представлены в виде ограничивающих рамок).

    Возвращает 3 списка:
    1. Совпадения
    2. Несопоставленные обнаружения
    3. Несопоставленные трекеры
    """
    if len(trackers) == 0:
        return np.empty((0,2), dtype=int), np.arange(len(detections)), np.empty((0,5), dtype=int)

    # Матрица IoU
    iou_matrix = np.zeros((len(detections), len(trackers)), dtype=np.float32)

    for d, det in enumerate(detections):
        for t, trk in enumerate(trackers):
            iou_matrix[d, t] = iou(det, trk)

    # Назначение с использованием венгерского алгоритма
    if min(iou_matrix.shape) > 0:
        a = (iou_matrix > iou_threshold).astype(np.int32)
        if a.sum(1).max() == 1 and a.sum(0).max() == 1:
            matched_indices = np.stack(np.where(a), axis=1)
        else:
            # Исправленная строка:
            row_ind, col_ind = linear_sum_assignment(-iou_matrix)
            matched_indices = np.array([[row, col] for row, col in zip(row_ind, col_ind)])
    else:
        matched_indices = np.empty(shape=(0,2))

    # Несопоставленные обнаружения
    unmatched_detections = []
    for d, det in enumerate(detections):
        if d not in matched_indices[:,0]:
            unmatched_detections.append(d)

    # Несопоставленные трекеры
    unmatched_trackers = []
    for t, trk in enumerate(trackers):
        if t not in matched_indices[:,1]:
            unmatched_trackers.append(t)

    # Фильтруем совпадения с низким IoU
    matches = []
    for m in matched_indices:
        if iou_matrix[m[0], m[1]] < iou_threshold:
            unmatched_detections.append(m[0])
            unmatched_trackers.append(m[1])
        else:
            matches.append(m.reshape(1,2))

    if len(matches) == 0:
        matches = np.empty((0,2), dtype=int)
    else:
        matches = np.concatenate(matches, axis=0)

    return matches, np.array(unmatched_detections), np.array(unmatched_trackers)

# Класс SORT
class Sort(object):
    def __init__(self, max_age=1, min_hits=3, iou_threshold=0.3):
        """
        Устанавливает параметры для SORT.
        """
        # max_age - Максимальное число кадров, которое трек может существовать без обновления (без совпадения с детекцией).
        # Если трек не обновляется дольше max_age кадров, он удаляется.

        # min_hits - Минимальное число успешных обновлений трека (hits), прежде чем он начнёт выдаваться как валидный.
        # Помогает отфильтровать ложные треки. Если в трех кадрах трек успешно обновляется, то он подтвержден.

        # iou_threshold - Порог Intersection over Union (IoU) для сопоставления треков и детекций.
        # Если IoU < iou_threshold, трек и детекция считаются разными объектами.

        # trackers - Список активных трекеров (экземпляров KalmanBoxTracker). Каждый трекер соответствует одному отслеживаемому объекту.
        # frame_count - Счётчик кадров. Увеличивается на каждом вызове update()

        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.trackers = []
        self.frame_count = 0

    def update(self, dets=np.empty((0,5))):
        """
        Параметры:
        dets - numpy-массив обнаружений в формате [[x1,y1,x2,y2,score],[x1,y1,x2,y2,score],...]

        Требуется: этот метод должен вызываться один раз для каждого кадра, даже с пустыми обнаружениями.

        Возвращает аналогичный массив, где последний столбец - это ID объекта.

        Примечание: количество возвращаемых объектов может отличаться от количества входных обнаружений.

        Метод update() выполняет основную работу по обновлению треков на каждом кадре видео. Он:

          1. Получает новые детекции объектов (dets).
          2. Сопоставляет их с существующими треками.
          3. Обновляет или удаляет треки.
          4. Возвращает актуальные треки с их ID.
        """
        self.frame_count += 1

        # Получаем предсказанные местоположения из существующих трекеров
        trks = np.zeros((len(self.trackers), 5))
        to_del = []
        ret = []

        for t, trk in enumerate(trks):
            pos = self.trackers[t].predict()[0]
            trk[:] = [pos[0], pos[1], pos[2], pos[3], 0]

            if np.any(np.isnan(pos)):
                to_del.append(t)

        trks = np.ma.compress_rows(np.ma.masked_invalid(trks))

        for t in reversed(to_del):
            self.trackers.pop(t)

        matched, unmatched_dets, unmatched_trks = associate_detections_to_trackers(dets, trks, self.iou_threshold)

        # Обновляем сопоставленные трекеры с назначенными обнаружениями
        for m in matched:
            self.trackers[m[1]].update(dets[m[0], :])

        # Создаем и инициализируем новые трекеры для несопоставленных обнаружений
        for i in unmatched_dets:
            trk = KalmanBoxTracker(dets[i,:])
            self.trackers.append(trk)

        i = len(self.trackers)
        for trk in reversed(self.trackers):
            d = trk.get_state()[0]

            if (trk.time_since_update < 1) and (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits):
                ret.append(np.concatenate((d, [trk.id+1])).reshape(1,-1)) # +1 потому что MOT Benchmark требует положительных ID

            i -= 1

            # Удаляем мертвые трекеры
            if trk.time_since_update > self.max_age:
                self.trackers.pop(i)

        if len(ret) > 0:
            return np.concatenate(ret)
        return np.empty((0,5))

# Загрузка предварительно обученной модели детекции (например, Faster R-CNN)
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
model.eval()
model.to(device)
# Загрузка COCO классов
COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana',
    'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut',
    'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table', 'N/A', 'N/A',
    'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

# Функция для детекции объектов
def detect_objects(img, model, threshold=0.5):
    """
    Обнаруживает объекты на изображении с помощью модели.
    Возвращает ограничивающие рамки и оценки.
    """
    # Преобразуем изображение в тензор
    img_tensor = torchvision.transforms.functional.to_tensor(img)
    img_tensor = img_tensor.unsqueeze(0) # Добавляем batch dimension
    img_tensor = img_tensor.to(device)

    # Детекция объектов
    with torch.no_grad():
        predictions = model(img_tensor)

    # Извлекаем рамки, оценки и метки классов
    boxes = predictions[0]['boxes'].cpu().numpy()
    scores = predictions[0]['scores'].cpu().numpy()
    labels = predictions[0]['labels'].cpu().numpy()

    # Фильтруем по порогу уверенности
    mask = scores >= threshold
    boxes = boxes[mask]
    scores = scores[mask]
    labels = labels[mask]

    # Объединяем рамки и оценки
    detections = np.hstack((boxes, scores[:, np.newaxis]))

    return detections, labels

# Основная функция для трекинга
def track_objects(video_path, output_path='output.mp4'):
    """
    Основная функция для трекинга объектов в видео.
    """
    # Инициализируем SORT трекер
    mot_tracker = Sort(max_age=5, min_hits=3, iou_threshold=0.3)

    # Открываем видео
    cap = cv2.VideoCapture(video_path)

    # Получаем параметры видео
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Создаем VideoWriter для сохранения результата
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Конвертируем BGR в RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Детектируем объекты
        detections, labels = detect_objects(rgb_frame, model)

        # Обновляем трекер
        tracked_objects = mot_tracker.update(detections)

        # Рисуем результаты
        for obj in tracked_objects:
            x1, y1, x2, y2, obj_id = obj
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])

            # Рисуем прямоугольник
            color = (int(255 * (obj_id % 3)/3), int(255 * (obj_id % 6)/6), int(255 * (obj_id % 9)/9))
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # Добавляем ID объекта
            cv2.putText(frame, f'ID: {int(obj_id)}', (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Сохраняем кадр
        out.write(frame)

        frame_count += 1
        if frame_count % 50 == 0:
            print(f'Обработано {frame_count} кадров')

    # Освобождаем ресурсы
    cap.release()
    out.release()
    cv2.destroyAllWindows()

    print(f'Трекинг завершен. Результат сохранен в {output_path}')
# Основная функция для трекинга
def track_objects(video_path, output_path='output.mp4'):
    """
    Основная функция для трекинга объектов в видео.
    """
    # Инициализируем SORT трекер
    mot_tracker = Sort(max_age=5, min_hits=3, iou_threshold=0.3)

    # Открываем видео
    cap = cv2.VideoCapture(video_path)

    # Получаем параметры видео
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Создаем VideoWriter для сохранения результата
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_count = 0

    # Для подсчета уникальных объектов и ложных срабатываний
    unique_object_ids = set()
    false_positives = 0
    track_lifespans = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Конвертируем BGR в RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Детектируем объекты и получаем метки классов
        detections, labels = detect_objects(rgb_frame, model)

        # Обновляем трекер
        tracked_objects = mot_tracker.update(detections)

        # Рисуем результаты
        for obj in tracked_objects:
            x1, y1, x2, y2, obj_id = obj
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
            obj_id = int(obj_id)

            # Находим соответствующую метку класса для детекции, с которой был сопоставлен этот трек
            # Это упрощенный подход, который предполагает, что если трек сопоставлен с детекцией,
            # то класс трека соответствует классу детекции.
            # В более сложных сценариях может потребоваться более сложная логика.
            obj_class = "Unknown"
            # Ищем детекцию, которая совпала с текущим треком по IoU
            for i, det in enumerate(detections):
                if iou(obj[:4], det[:4]) > mot_tracker.iou_threshold:
                     # Проверяем, есть ли соответствующая метка класса для этой детекции
                    if i < len(labels):
                        obj_class = COCO_INSTANCE_CATEGORY_NAMES[labels[i]]
                        break


            # Рисуем прямоугольник
            color = (int(255 * (obj_id % 3)/3), int(255 * (obj_id % 6)/6), int(255 * (obj_id % 9)/9))
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # Добавляем ID объекта и класс
            text = f'ID: {obj_id} Class: {obj_class}'
            cv2.putText(frame, text, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # Добавляем ID в набор уникальных
            unique_object_ids.add(obj_id)


        # Обработка треков после обновления для подсчета ложных срабатываний и времени жизни
        # Здесь мы проходим по трекерам после их обновления и удаления "мертвых"
        # В реальном приложении для точного подсчета времени жизни треков,
        # возможно, потребуется сохранить информацию о треках, которые были удалены.
        # В данном случае, мы можем оценить время жизни на основе активных треков
        # и треков, которые были недавно удалены.
        # Для простоты, подсчитаем ложные срабатывания как треки, которые были удалены
        # и не достигли минимального числа попаданий.

        # Сохраняем кадр
        out.write(frame)

        frame_count += 1
        if frame_count % 50 == 0:
            print(f'Обработано {frame_count} кадров')

    # После завершения обработки видео, проходим по всем трекерам, которые все еще активны,
    # и добавляем их время жизни в список track_lifespans.
    for trk in mot_tracker.trackers:
        track_lifespans.append(trk.age)
        # Ложные срабатывания - треки, которые не достигли min_hits
        if trk.hit_streak < mot_tracker.min_hits:
            false_positives += 1

    # Добавляем время жизни треков, которые были удалены в процессе
    # Это требует отслеживания удаленных треков, что не реализовано в базовом SORT.
    # Для более точного подсчета false_positives и среднего времени жизни,
    # нужно модифицировать класс SORT для хранения информации об удаленных треках.
    # В текущей реализации, false_positives будут считаться только для треков,
    # которые все еще активны в конце видео и не достигли min_hits.

    # Освобождаем ресурсы
    cap.release()
    out.release()
    cv2.destroyAllWindows()

    print(f'Трекинг завершен. Результат сохранен в {output_path}')

    # Вывод статистики
    print("\nСтатистика трекинга:")
    print(f"Общее число уникальных объектов: {len(unique_object_ids)}")

    # Расчет среднего времени жизни трека (для активных треков в конце видео)
    if len(track_lifespans) > 0:
        avg_lifespan = sum(track_lifespans) / len(track_lifespans)
        print(f"Среднее время жизни трека (для активных треков): {avg_lifespan:.2f} кадров")
    else:
        print("Нет активных треков для расчета среднего времени жизни.")

    # Число ложных срабатываний (треки, которые были активны в конце и не достигли min_hits)
    print(f"Число ложных срабатываний (оценка по активным трекам): {false_positives}")


#Замена на yolo
!pip install ultralytics
from ultralytics import YOLO
from collections import defaultdict
from scipy.optimize import linear_sum_assignment

model = YOLO('yolov8n.pt')
model.to(device)

# Получение названий классов из модели YOLOv8
# YOLOv8 хранит названия классов в атрибуте names
COCO_INSTANCE_CATEGORY_NAMES = model.names

def detect_objects(img, model, threshold=0.5):
    """
    Обнаруживает объекты на изображении с помощью модели YOLOv8.
    Возвращает ограничивающие рамки, оценки и метки классов.
    """
    # YOLOv8 ожидает изображения в формате BGR
    # img_rgb = cv2.cvtColor(img, cv2.COLOR_RGB2BGR) # В track_objects уже конвертируется в RGB, поэтому нет необходимости здесь

    # Выполнение детекции
    results = model(img, conf=threshold)

    # Извлекаем рамки, оценки и метки классов
    boxes = []
    scores = []
    labels = []

    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf = box.conf[0].item()
            cls = box.cls[0].item()

            boxes.append([x1, y1, x2, y2])
            scores.append(conf)
            labels.append(int(cls)) # Метки классов в YOLOv8 - это float, преобразуем в int

    detections = np.hstack((np.array(boxes), np.array(scores)[:, np.newaxis]))

    return detections, np.array(labels)'''