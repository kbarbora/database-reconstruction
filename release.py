
import os
from time import process_time
import tensorflow as tf

X, Y = 0, 1
EPOCHS = 20  # Paper 100 epochs
BATCH = 60_000  # Paper full-batch

def get_dataset():
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train, x_test = x_train / 255.0, x_test / 255.0
    dataset = {'train': (x_train, y_train),
               'test': (x_test, y_test)}
    return dataset

def released_model():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(10, activation='elu'),
        tf.keras.layers.Dense(10)
    ])
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    opt = tf.keras.optimizers.SGD(learning_rate=0.2, momentum=0.9)
    model.compile(optimizer=opt,
                  loss=loss_fn,
                  metrics=['accuracy'])
    print(model.summary())
    return model

def train_released(predef_model=None):
    dataset = get_dataset()
    model = released_model() if predef_model is None else predef_model
    x_train, y_train = dataset['train'][X], dataset['train'][Y]
    model.fit(x_train, y_train, batch_size=BATCH, epochs=EPOCHS)

    x_test, y_test = dataset['test'][X], dataset['test'][Y]
    model.evaluate(x_test, y_test, verbose=1)
    return model

def test():
    times = []
    for i in range(1):
        timer = process_time()
        model = train_released()
        delta = process_time() - timer
        print(f"Elapsed time {delta}")
        times.append(delta)
    print(f"Average {sum(times) / len(times)}")



if __name__ == '__main__':
    test()