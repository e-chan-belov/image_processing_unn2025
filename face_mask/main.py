import telebot
from telebot import types
import numpy as np
from PIL import Image
import io
import os
import random
import joblib

import torch
import torch.nn as nn
from torchvision import transforms, models
from skimage.feature import hog


TOKEN = "8231811007:AAG856oZyu_wSKqcTk7dc5CPQNAAon3SPgA"

bot = telebot.TeleBot(TOKEN)

MENU_STATE = "menu"
STATE_1 = "1"
STATE_2 = "2"
STATE_3 = "3"
BEST_STATE = STATE_1

svm_model = None
cnn_model = None
resnet_model = None
best_model_type = None
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

CLASSES = ['В маске', 'Без маски']
IMG_SIZE = 128

user_states = {}

class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2, 2), nn.Dropout(0.25),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2, 2), nn.Dropout(0.25),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(2, 2), nn.Dropout(0.25),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 16 * 16, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(256, 1), nn.Sigmoid()
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def load_models():
    global svm_model, cnn_model, resnet_model, best_model_type
    
    if os.path.exists('model_1_svm.pkl'):
        svm_model = joblib.load('model_1_svm.pkl')
        print("Модель 1 (SVM) загружена")
    
    if os.path.exists('model_2_cnn.pth'):
        cnn_model = SimpleCNN().to(device)
        cnn_model.load_state_dict(torch.load('model_2_cnn.pth', map_location=device))
        cnn_model.eval()
        print("Модель 2 (CNN) загружена")
    
    if os.path.exists('model_3_resnet18.pth'):
        resnet_model = models.resnet18(weights=None)
        resnet_model.fc = nn.Sequential(
            nn.Linear(resnet_model.fc.in_features, 256),
            nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(256, 1), nn.Sigmoid()
        )
        resnet_model.load_state_dict(torch.load('model_3_resnet18.pth', map_location=device))
        resnet_model = resnet_model.to(device)
        resnet_model.eval()
        print("Модель 3 (ResNet18) загружена")
    
    best_model_type = STATE_3
    print(f"Лучшая модель: {best_model_type}")


transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


def extract_hog_features(img_pil, img_size=64):
    img = img_pil.convert('L').resize((img_size, img_size))
    img_array = np.array(img)
    features = hog(img_array, orientations=9, pixels_per_cell=(8, 8),
                   cells_per_block=(2, 2), visualize=False, feature_vector=True)
    return features


def predict_with_model(img_pil, model_state):
    
    if model_state == STATE_1:  # HOG + SVM
        if svm_model is None:
            return None, "Модель SVM не загружена"
        features = extract_hog_features(img_pil)
        pred = svm_model.predict([features])[0]
        prob = svm_model.predict_proba([features])[0]
        confidence = max(prob) * 100
        return pred, confidence
    
    elif model_state == STATE_2:  # CNN
        if cnn_model is None:
            return None, "Модель CNN не загружена"
        img_tensor = transform(img_pil.convert('RGB')).unsqueeze(0).to(device)
        with torch.no_grad():
            output = cnn_model(img_tensor).item()
        pred = 1 if output > 0.5 else 0
        confidence = output * 100 if pred == 1 else (1 - output) * 100
        return pred, confidence
    
    elif model_state == STATE_3:  # ResNet18
        if resnet_model is None:
            return None, "Модель ResNet18 не загружена"
        img_tensor = transform(img_pil.convert('RGB')).unsqueeze(0).to(device)
        with torch.no_grad():
            output = resnet_model(img_tensor).item()
        pred = 1 if output > 0.5 else 0
        confidence = output * 100 if pred == 1 else (1 - output) * 100
        return pred, confidence
    
    return None, "Неизвестная модель"


def get_model_name(state):
    names = {
        STATE_1: "HOG + SVM",
        STATE_2: "CNN",
        STATE_3: "ResNet18"
    }
    return names.get(state, "Неизвестная")



@bot.message_handler(commands=['start'])
def start(message):
    chat_id = message.from_user.id
    user_states[chat_id] = MENU_STATE
    markup = types.ReplyKeyboardMarkup(resize_keyboard=True)
    btn1 = types.KeyboardButton('Модель 1 (SVM)')
    btn2 = types.KeyboardButton('Модель 2 (CNN)')
    btn3 = types.KeyboardButton('Модель 3 (ResNet)')
    btn4 = types.KeyboardButton('Лучшая модель')
    markup.add(btn1, btn2, btn3, btn4)
    bot.send_message(chat_id, "Пожалуйста, выберите модель", reply_markup=markup)

@bot.message_handler(content_types=['text'])
def handle_text(message):
    chat_id = message.from_user.id
    text = message.text.strip().lower()

    if text == "меню выбора модели":
        start(message)
        return
    
    markup = types.ReplyKeyboardMarkup(resize_keyboard=True)
    btn1 = types.KeyboardButton('Выбрать случайную тестовую картинку')
    btn2 = types.KeyboardButton('Меню выбора модели')
    markup.add(btn1, btn2)

    if text == "выбрать случайную тестовую картинку" and user_states[chat_id] != MENU_STATE:
        if not images:
            bot.send_message(chat_id, "Тестовые изображения не найдены")
            return
        image_path = random.choice(images)
        img_pil = Image.open(image_path)
        
        img_bytes = io.BytesIO()
        img_pil.save(img_bytes, format='JPEG')
        img_bytes.seek(0)
        bot.send_photo(chat_id, photo=img_bytes)
        
        predict_and_send(chat_id, img_pil)
        return
    
    if text == "модель 1 (svm)":
        user_states[chat_id] = STATE_1
        bot.send_message(chat_id, "Выбрана модель 1", reply_markup=markup)
        return
    if text == "модель 2 (cnn)":
        user_states[chat_id] = STATE_2
        bot.send_message(chat_id, "Выбрана модель 2", reply_markup=markup)
        return
    if text == "модель 3 (resnet)":
        user_states[chat_id] = STATE_3
        bot.send_message(chat_id, "Выбрана модель 3", reply_markup=markup)
        return
    if text == "лучшая модель":
        user_states[chat_id] = STATE_3
        bot.send_message(chat_id, "Выбрана лучшая модель:\n (ResNet)", reply_markup=markup)
        return


def predict_and_send(chat_id, img_pil):
    """Делает предсказание и отправляет результат"""
    state = user_states.get(chat_id)
    if state is None or state == MENU_STATE:
        start_msg = type('obj', (object,), {'from_user': type('obj', (object,), {'id': chat_id})})()
        start(start_msg)
        return
    
    pred, confidence = predict_with_model(img_pil, state)
    
    if pred is None:
        bot.send_message(chat_id, f"❌ Ошибка: {confidence}")
        return
    
    result_class = CLASSES[pred]
    model_name = get_model_name(state)
    
    emoji = "😷" if pred == 0 else "😐"
    
    bot.send_message(
        chat_id,
        f"{emoji} **Результат классификации**\n\n"
        f"🤖 Модель: {model_name}\n"
        f"📊 Результат: {result_class}\n"
        f"💯 Уверенность: {confidence:.1f}%",
        parse_mode='Markdown'
    )


@bot.message_handler(content_types=['photo'])
def handle_photo(message):
    chat_id = message.from_user.id
    if (user_states[chat_id] == MENU_STATE): 
        start(message)
        return
    photo = message.photo[-1]

    file_info = bot.get_file(photo.file_id)
    data = bot.download_file(file_info.file_path)

    img_pil = Image.open(io.BytesIO(data))
    
    predict_and_send(chat_id, img_pil)


def collect_images(root_dir: str) -> list[str]:
    images = []
    for folder in os.listdir(root_dir):
        folder_path = os.path.join(root_dir, folder)

        if not os.path.isdir(folder_path):
            continue

        for file in os.listdir(folder_path):
            if file.lower().endswith((".png", ".jpg", ".jpeg", ".webp")):
                images.append(os.path.join(folder_path, file))

    return images

if __name__ == "__main__":
    print("Загрузка моделей...")
    load_models()

    global images
    images = collect_images("Test")

    print("BOT УСПЕШНО ЗАПУЩЕН")
    bot.infinity_polling()
