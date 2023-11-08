import os
from io import BytesIO

import cv2
import keras
import numpy as np
import requests
import tensorflow as tf
from keras.callbacks import ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator
from PIL import Image
from scipy.stats import entropy

MODEL_PATH = r"files\model"
IMAGE_SIZE = 150
VAL_PATH = r"files\val"
DB_IMAGES = r"files\temp"
MODEL_THRESHOLD = 0.92
EPOCHS = 2
BATCH_SIZE = 8

model = keras.models.load_model(MODEL_PATH)


def classify_image(image):
    img_array = np.array(image)
    image = cv2.cvtColor(img_array, cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE))
    image_array = np.array(image) / 255

    image_array = image_array[:, :, 0]
    image_array = image_array.reshape(-1, IMAGE_SIZE, IMAGE_SIZE, 1)
    prediction = model.predict(image_array)  # prediction[0][0]

    if prediction <= 0.5:
        prediction_text = "PNEUMONIA"
        # probability = 1 - prediction
    elif prediction > 0.5:
        prediction_text = "NORMAL"
        # probability = prediction

    # probability = abs(prediction - 0.5) * 2      # linear
    entropy_information = entropy([prediction, 1 - prediction], base=2)

    return prediction_text, (1 - entropy_information[0][0])


def retrain_model(image_list):
    x_val = []
    y_val = []
    for i in range(1, 8):
        p = os.path.join(VAL_PATH, r"NORMAL\NORMAL (" + str(i) + ").jpeg")
        img_arr = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        resized_arr = cv2.resize(img_arr, (IMAGE_SIZE, IMAGE_SIZE))
        x_val.append(resized_arr)
        y_val.append(1)  # 0 - PNEUMONIA, 1 - NORMAL

    for i in range(1, 8):
        p = os.path.join(VAL_PATH, r"PNEUMONIA\PNEUMONIA (" + str(i) + ").jpeg")
        img_arr = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        resized_arr = cv2.resize(img_arr, (IMAGE_SIZE, IMAGE_SIZE))
        x_val.append(resized_arr)
        y_val.append(0)  # 0 - PNEUMONIA, 1 - NORMAL

    x_val = np.array(x_val) / 255
    x_val = x_val.reshape(-1, IMAGE_SIZE, IMAGE_SIZE, 1)
    y_val = np.array(y_val)

    data = []
    labels = []

    for image_entry in image_list:
        image = Image.open(BytesIO(requests.get(image_entry[0]).content)).convert("L")
        image = cv2.resize(np.array(image), (IMAGE_SIZE, IMAGE_SIZE))
        image_array = np.array(image) / 255
        # image_array = image_array[:, :, 0]

        if image_entry[1] != "":
            if image_entry[2] != "":  # user label
                if image_entry[2] == "NORMAL":
                    labels.append(1)
                elif image_entry[2] == "PNEUMONIA":
                    labels.append(0)
                else:
                    print("Error in database labels.")
                    continue  # skip data entry
                data.append(image_array)
            else:  # model label
                if image_entry[1] == "NORMAL":
                    labels.append(1)
                elif image_entry[1] == "PNEUMONIA":
                    labels.append(0)
                else:
                    print("Error in database labels.")
                    continue  # skip data entry
                data.append(image_array)
        else:
            print("Error in database labels.")
            continue  # skip data entry

    data = np.array(data)
    data = data.reshape(-1, IMAGE_SIZE, IMAGE_SIZE, 1)

    datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=30,  # randomly rotate images in the range (degrees, 0 to 180)
        zoom_range=0.2,  # Randomly zoom image
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=False,  # randomly flip images
        vertical_flip=False,  # randomly flip images
    )
    datagen.fit(data)

    model = keras.models.load_model(MODEL_PATH)
    metrics = [
        tf.keras.metrics.Precision(name="accuracy"),
        tf.keras.metrics.Precision(name="precision"),
        tf.keras.metrics.Recall(name="recall"),
    ]
    model.compile(optimizer="rmsprop", loss="binary_crossentropy", metrics=metrics)
    learning_rate_reduction = ReduceLROnPlateau(
        monitor="val_accuracy", patience=2, verbose=1, factor=0.3, min_lr=0.000001
    )

    model.fit(
        datagen.flow(data, labels, batch_size=BATCH_SIZE),
        epochs=EPOCHS,
        validation_data=(x_val, y_val),
        callbacks=[learning_rate_reduction],
    )

    for temp_image in os.listdir(DB_IMAGES):
        os.remove(os.path.join(DB_IMAGES, temp_image))

    results = model.evaluate(x_val, y_val)
    if results[1] > MODEL_THRESHOLD:
        model.save(r"files\new_model")
        print("Model results:", results)
        return "Model successfully re-trained on new data."
    else:
        return "Re-trained model performed worse. Model discarded."
