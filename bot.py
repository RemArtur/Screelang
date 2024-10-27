import os
import numpy as np
from PIL import Image
from dotenv import find_dotenv, load_dotenv
from aiogram import Bot, Dispatcher, types
from aiogram.filters import Command
from aiogram.types import FSInputFile, InlineKeyboardButton, CallbackQuery
from aiogram.utils.keyboard import ReplyKeyboardBuilder, InlineKeyboardBuilder
from ultralytics import YOLO
from deep_translator import GoogleTranslator
import cv2

model = YOLO("yolo11n.pt")

load_dotenv(find_dotenv(filename=".env"))
API_TOKEN = str(os.getenv("BOT_TOKEN"))
bot = Bot(token=API_TOKEN)
dp = Dispatcher()


#------------------------------------- Работа с языком ---------------------------------------------

user_languages = {}
default_language = 'en'

LANGUAGE_OPTIONS = {
    "Русский": "ru",
    "Английский": "en",
    "Испанский": "es",
    "Французский": "fr",
    "Немецкий": "de",
}

def translate_text(text, lang):
    return GoogleTranslator(target=lang).translate(text)


#------------------------------------- Обработчики ---------------------------------------------


@dp.message(Command("start"))
async def start_command(message: types.Message):
    # Создаем клавиатуру с кнопкой для установки языка
    keyboard = ReplyKeyboardBuilder()
    keyboard.add(types.KeyboardButton(text="Выбрать другой язык"))

    await message.reply("Привет, я буду переводить предметы на фото!", reply_markup=keyboard.adjust(*[1,]).as_markup())

# Обработчик нажатий на inline-кнопки для изменения языка
@dp.callback_query(lambda callback_query: callback_query.data.startswith("set_lang_"))
async def handle_language_selection(callback_query: CallbackQuery):
    lang_code = callback_query.data.split("_")[-1]
    user_languages[callback_query.from_user.id] = lang_code
    
    # Сообщение пользователю об изменении языка
    await callback_query.message.edit_text(f"Язык перевода успешно изменен на {lang_code}.")

@dp.message()
async def handle_photo(message: types.Message):
    if "Выбрать другой язык" in message.text:
        keyboard = InlineKeyboardBuilder()
        for lang_name, lang_code in LANGUAGE_OPTIONS.items():
            keyboard.add(InlineKeyboardButton(text=lang_name, callback_data=f"set_lang_{lang_code}"))
        await message.reply("Выберите язык перевода:", reply_markup=keyboard.adjust(*(2,)).as_markup())
        return

    if not message.photo:
        await message.reply("Пришли, пожалуйста, фото")
        return

    # Загружаем фотографию, присланную пользователем
    photo_file = await bot.get_file(message.photo[-1].file_id)
    photo_data = await bot.download_file(photo_file.file_path)
    
    # Открываем изображение с помощью PIL
    image_cv = Image.open(photo_data).convert("RGB")
    image_cv = np.array(image_cv)
    
    # Преобразуем в формат OpenCV
    image_cv = cv2.cvtColor(image_cv, cv2.COLOR_RGB2BGR)


    # Детектируем объекты
    results = model(image_cv)
    names = results[0].names
    user_lang = user_languages.get(message.from_user.id, default_language)

    # Обрабатываем каждый детектированный объект
    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        class_name = names[int(box.cls[0].item())]

        # Переводим название объекта
        translated_label = translate_text(class_name, user_lang)

        # Рисуем прямоугольник и текст на изображении
        cv2.rectangle(image_cv, (x1, y1), (x2, y2), (255, 255, 255), 2)
        cv2.putText(image_cv, translated_label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2)

    cv2.imwrite('tmp.jpeg', image_cv)

    # Отправляем изображение обратно пользователю
    await message.reply_photo(FSInputFile('tmp.jpeg'))
    os.remove('tmp.jpeg')


# Запуск бота
if __name__ == '__main__':
    import asyncio
    async def main():
        await dp.start_polling(bot)
    asyncio.run(main())