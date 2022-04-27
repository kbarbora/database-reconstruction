import numpy as np # linear algebra
import pandas as pd
import os
from time import process_time
import tensorflow as tf
import random

RWEIGHTS = 7960
X, Y = 0, 1
EPOCHS = 100  # Paper 100 epochs
LAYER_WIDTH = 1000
BATCH = 0  # Paper full-batch

def shadow_model(dataset, target):

    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(10000, 7960)),
        tf.keras.layers.Dense(LAYER_WIDTH, activation='relu'),
        tf.keras.layers.Dense(LAYER_WIDTH, activation='relu'),
        tf.keras.layers.Dense(10)
    ])
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    opt = tf.keras.optimizers.SGD(learning_rate=0.2, momentum=0.9)
    model.compile(optimizer=opt,
                  loss=loss_fn,
                  metrics=['accuracy'])
    x_train, y_train = dataset['train'][X], dataset['train'][Y]

    model.fit(x_train, y_train, batch_size=BATCH, epochs=EPOCHS)

    x_test, y_test = dataset['test'][X], dataset['test'][Y]
    model.evaluate(x_test, y_test, verbose=1)
    return model.get_weights()