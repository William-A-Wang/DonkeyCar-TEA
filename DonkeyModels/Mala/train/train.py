#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import glob
import os
import cv2
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Input, Dense, Dropout, Flatten
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt
import time
import re


def load_catalogs(folder: str):
    _img_name = []
    _angle = []
    _throttle = []

    for _file in sorted(glob.glob(f"{folder}/*.catalog"),
                        key=lambda x: [
                            int(c) if c.isdigit()
                            else c for c in re.split(r'(\d+)', x)]):
        with open(_file) as f:
            for _line in f:
                _img_name.append(_line.split()[7][1:-2])
                _angle.append(float(_line.split()[9][0:-1]))
                _throttle.append(float(_line.split()[13][0:-1]))

    print(f'Image count: {len(_img_name)}')
    return _img_name, _angle, _throttle


def load_images(_img_name: list, folder: str):
    _image = []
    for i in range(len(_img_name)):
        _img = cv2.imread(os.path.join(f"{folder}/images", _img_name[i]))
        assert _img.shape == (224, 224, 3),\
            "img %s has shape %r" % (_img_name[i], _img.shape)
        _image.append(_img)
    return _image


def data_preprocessing(_throttle, _angle, _image):
    _throttle = np.array(_throttle)
    _steering = np.array(_angle)
    _train_img = np.array(_image)
    _label = _steering
    _cut_height = 80
    _train_img_cut_orig = _train_img[:, _cut_height:224, :]
    _img_gray = np.dot(_train_img[..., :3], [0.299, 0.587, 0.114])
    _train_img_cut_gray = _img_gray[:, _cut_height:224, :]
    return _train_img_cut_orig, _train_img_cut_gray, _label


def train_split(_train_img_cut_orig, _train_img_cut_gray, _label):
    _X_train, _X_test, _y_train, _y_test = train_test_split(
        _train_img_cut_gray, _label,
        test_size=0.15, random_state=42)
    return _X_train, _X_test, _y_train, _y_test


def core_cnn_layers(img_in, drop, l4_stride=1):
    _ = img_in
    _ = tf.keras.layers.Conv1D(filters=24, kernel_size=5,
                               strides=2, activation='tanh',
                               name='conv1d_1')(_)
    _ = tf.keras.layers.Dropout(drop)(_)
    _ = tf.keras.layers.Conv1D(filters=32, kernel_size=3,
                               strides=l4_stride, activation='tanh',
                               name='conv1d_2')(_)
    _ = tf.keras.layers.Dropout(drop)(_)
    _ = tf.keras.layers.Conv1D(filters=32, kernel_size=3,
                               strides=1, activation='tanh',
                               name='conv1d_3')(_)
    _ = tf.keras.layers.Dropout(drop)(_)
    _ = Flatten(name='flattened')(_)
    return _


def default_n_linear(num_outputs=None, input_shape=(120, 160)):
    _drop = 0.5
    _img_in = Input(shape=input_shape, name='img_in')
    _x = core_cnn_layers(_img_in, _drop)
    _x = Dense(50, activation='tanh', name='dense_1')(_x)
    _x = Dropout(_drop)(_x)
    _outputs = Dense(1, activation='tanh', name='outputs')(_x)
    _model = Model(inputs=[_img_in], outputs=_outputs, name='regression_tanh')
    return _model


def train_start(_model, _X_train, _X_test, _y_train, _y_test):
    _optimizer = tf.optimizers.Adam(learning_rate=0.0001,
                                    beta_1=0.9, beta_2=0.999)
    _model.compile(optimizer=_optimizer, loss="mse", metrics=["accuracy"])
    _model.summary()
    _trained_model = _model.fit(_X_train, _y_train,
                                epochs=3000, batch_size=20,
                                validation_data=(_X_test, _y_test))
    return _trained_model


def plot_trained_model(_trained_model, show: bool = False,
                       save: bool = True):
    history = _trained_model.history

    plt.plot(history['loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    if save:
        plt.savefig(f'Loss_{time.ctime(time.time())}.png',
                    bbox_inches='tight')
    if show:
        plt.show()

    plt.plot(history['accuracy'], label='Train Accuracy')
    plt.plot(history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend()
    if save:
        plt.savefig(f'Accuracy_{time.ctime(time.time())}.png',
                    bbox_inches='tight')
    if show:
        plt.show()


if __name__ == "__main__":
    data_folder = "Your Data"
    img_name, angle, throttle = load_catalogs(data_folder)
    image = load_images(img_name, data_folder)
    image = np.array(image)
    train_img_cut_orig, train_img_cut_gray, label = data_preprocessing(
        throttle, angle, image)
    X_train, X_test, y_train, y_test = train_split(
        train_img_cut_orig, train_img_cut_gray, label)
    model = default_n_linear(1, input_shape=(144, 224))
    trained_model = train_start(model, X_train, X_test, y_train, y_test)
    plot_trained_model(trained_model, show=False, save=True)
    model.save(f"model_{time.ctime(time.time())}.h5")
