import tensorflow as tf
from keras import models
import matplotlib.pyplot as plt
import pickle
import os
from settings import *
import numpy as np
import makeDataset as mdst

# функция для обучения модели
# datasetPath - путь к датасету
# pathToModel - путь к файлу модели
def teachModel(datasetPath, pathToModel):
    os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
    with open(datasetPath, 'rb') as data:
        imsTest, lblsTest, imsTrain, lblsTrain = pickle.load(data)

    model = models.Sequential()
    model.add(tf.keras.layers.Flatten(input_shape=(30, 18)))
    model.add(tf.keras.layers.Dense(128,activation=tf.nn.relu))
    model.add(tf.keras.layers.Dense(30,activation=tf.nn.relu))
    model.add(tf.keras.layers.Dense(10,activation=tf.nn.softmax))

    model.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy']
                )

    history = model.fit(imsTrain, lblsTrain, epochs=10, validation_data=(imsTest, lblsTest))

    plt.plot(history.history['accuracy'], label='accuracy')
    plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim([0.5, 1])
    plt.legend(loc='lower right')
    plt.show()

    test_loss, test_acc = model.evaluate(imsTest, lblsTest, verbose=2)
    print(test_loss)
    print(test_acc)

    model.save(pathToModel)

# Функция для использования капчи
# pathToPic - путь к разгадываемой картинке
def picToCaptcha(pathToPic):
    digitsModel = tf.keras.models.load_model(digitsModelPath)
    digits = np.array(mdst.cutterSimple(pathToPic))
    predictions = digitsModel.predict(digits)
    res = ''
    for p in predictions:
        res += str(np.argmax(p))    
    return res