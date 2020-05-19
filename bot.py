import io
import os

import cv2
import requests
import telebot
import numpy as np

from classify import get_cancer_class_from

token = os.environ['TELEGRAM_ACCESS_TOKEN']
bot = telebot.TeleBot(token)


@bot.message_handler(commands=['start'])
def send_welcome(message):
    start_message = """Hi! I'm a Leskin Bot! I have been created to help you to diagnose 
                           a skin cancer. Please, be aware that I'm not 100% accurate and you
                           should consult with your doctor"""
    start_message = " ".join(start_message.split())

    bot.reply_to(message, start_message)
    bot.send_message(message.chat.id, "Send me a photo of a skin lesion and I will classify it")


@bot.message_handler(content_types=['photo'])
def send_image_class(message):
    bot.send_message(message.chat.id, "Received your image. Starting to analyze!")
    img = get_image_from(message)

    cancer_class = get_cancer_class_from(img)

    if cancer_class:
        bot.reply_to(message, f"Hm, it seems to me like {cancer_class}")
    else:
        bot.reply_to(message, "This lesion seems as a benign")


def get_image_from(message: telebot.types.Message) -> np.ndarray:
    photo = message.photo[-1]
    image_file_id = photo.file_id
    file_path = bot.get_file(image_file_id).file_path
    image_url = f"https://api.telegram.org/file/bot{token}/{file_path}"
    img_stream = io.BytesIO(requests.get(image_url).content)
    img = cv2.imdecode(np.fromstring(img_stream.read(), np.uint8), 1)
    return img


bot.set_webhook(f"https://leskin-bot.ew.r.appspot.com/{token}")
