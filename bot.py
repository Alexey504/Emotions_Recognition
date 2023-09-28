import logging
import os
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import numpy as np
from telegram import Update
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters, CallbackContext

# logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO
)

logger = logging.getLogger(__name__)
DICT_LABELS = {0: "злой", 1: "весёлый", 2: "грустный"}
model = 'https://storage.yandexcloud.net/model-er'


def start(update: Update, context: CallbackContext) -> None:
    """
    Отправляет сообщение при команде /start
    """
    update.message.reply_text('Моя главная задача – распознать какое у вас сейчас настроение. Отправьте фото.')


def help_command(update: Update, context: CallbackContext) -> None:
    """Отправляет сообщение при команде /help"""
    update.message.reply_text('Просто отправте фото...')


def load_my_model():
    """
    Загрузка модели локально
    """
    global model
    model = load_model('../my_model.h5', compile=False)
    print('Model loaded')


def get_prediction(img):
    """
    Функция классификации эмоцций

    :param img: изображение от пользователя
    :return: номер класса
    """
    img_path = './' + img
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    expanded_img_array = np.expand_dims(img_array, axis=0)
    preprocessed_img = expanded_img_array / 255.  # Preprocess the image
    prediction = model.predict(preprocessed_img)
    pred = np.argmax(prediction, axis=1)
    lbl = pred[0]
    return lbl


def photo(update: Update, context: CallbackContext):
    user = update.message.from_user
    photo_file = update.message.photo[-1].get_file()
    photo_file.download('user_photo.jpg')
    logger.info("Photo of %s: %s", user.first_name, 'user_photo.jpg')
    update.message.reply_text(
        'Одну секундочку'
    )
    # update.message.reply_text(get_prediction('user_photo.jpg'))
    label = get_prediction('user_photo.jpg')
    update.message.reply_text(f'Вы сейчас {DICT_LABELS[label]}')


def main():

    load_my_model()
    TOKEN = "6493159065:AAFwnFiZFhNOLlYLlCtnbfg9XOB8ufEYHe8"
    updater = Updater(TOKEN, use_context=True)
    # PORT = int(os.environ.get('PORT', '8443'))

    dispatcher = updater.dispatcher

    dispatcher.add_handler(CommandHandler("start", start))
    dispatcher.add_handler(CommandHandler("help", help_command))
    dispatcher.add_handler(MessageHandler(Filters.photo & ~Filters.command, photo))

    updater.start_polling()
    updater.idle()


if __name__ == '__main__':
    main()
