import os.path

import tensorflow as tf

import logger
from tumor_classification import ImageRecognizer

"""
This script is going to create models to evaluate a good configuration for the Tumor Classification. 

We will try: 1, 3, 5, 7 and 10 Neuron layers added to 2, 3 and 5 Convolutional layers. Each model will
be trained on 5, 10 and 20 epochs.
Amount of models: 
5 Dense layers * 4 Convolutional layer options * 3 different amounts of epochs each = 60 total models

Best overall model will be trained on different other specs, e. g. different batch size

Model tag structure: 
m_(model_number)_d_(n dense layers)_conv_(n convolutional layers)_e_(number of epochs trained on) 

e.g. m_1_d_1_conv_0_e_5
"""


def tag(model_number, dense_layers_number, convolutional_layers_number, epochs_numbers):
    return "m_" + str(model_number) + "_d_" + str(dense_layers_number) + "_conv_" + str(
        convolutional_layers_number) + "_e_" + str(epochs_numbers)


def get_current_epochs(i):
    assert 0 <= i <= 2
    if i == 0:
        return 5
    elif i == 1:
        return 10
    elif i == 2:
        return 20


def get_dense_layers(i):
    assert 0 <= i <= 4
    activation = tf.nn.relu
    if i == 0:
        return [
            tf.keras.layers.Dense(units=64, activation=activation, )
        ]
    elif i == 1:
        return [
            tf.keras.layers.Dense(units=64, activation=activation),
            tf.keras.layers.Dense(units=128, activation=activation),
            tf.keras.layers.Dense(units=256, activation=activation)
        ]
    elif i == 2:
        return [
            tf.keras.layers.Dense(units=64, activation=activation),
            tf.keras.layers.Dense(units=128, activation=activation),
            tf.keras.layers.Dense(units=256, activation=activation),
            tf.keras.layers.Dense(units=512, activation=activation),
            tf.keras.layers.Dense(units=1028, activation=activation)
        ]
    elif i == 3:
        return [
            tf.keras.layers.Dense(units=64, activation=activation),
            tf.keras.layers.Dense(units=128, activation=activation),
            tf.keras.layers.Dense(units=256, activation=activation),
            tf.keras.layers.Dense(units=512, activation=activation),
            tf.keras.layers.Dense(units=1028, activation=activation),
            tf.keras.layers.Dense(units=1028, activation=activation),
            tf.keras.layers.Dense(units=1028, activation=activation)
        ]
    elif i == 4:
        return [
            tf.keras.layers.Dense(units=64, activation=activation),
            tf.keras.layers.Dense(units=128, activation=activation),
            tf.keras.layers.Dense(units=256, activation=activation),
            tf.keras.layers.Dense(units=512, activation=activation),
            tf.keras.layers.Dense(units=1028, activation=activation),
            tf.keras.layers.Dense(units=1028, activation=activation),
            tf.keras.layers.Dense(units=1028, activation=activation),
            tf.keras.layers.Dense(units=1028, activation=activation),
            tf.keras.layers.Dense(units=1028, activation=activation),
            tf.keras.layers.Dense(units=1028, activation=activation)
        ]


def get_convolutional_layers(i):
    assert 0 <= i <= 3
    kernel_size = (3, 3)
    pool_size = (2, 2)
    activation = tf.nn.relu
    if i == 0:
        return []
    if i == 1:
        return [
            tf.keras.layers.Conv2D(16, kernel_size=kernel_size, activation=activation, padding="same"),
            tf.keras.layers.MaxPooling2D(pool_size=pool_size),
            tf.keras.layers.Conv2D(64, kernel_size=kernel_size, activation=activation, padding="same"),
            tf.keras.layers.MaxPooling2D(pool_size=pool_size)
        ]
    elif i == 2:
        return [
            tf.keras.layers.Conv2D(16, kernel_size=kernel_size, activation=activation, padding="same"),
            tf.keras.layers.MaxPooling2D(pool_size=pool_size),
            tf.keras.layers.Conv2D(32, kernel_size=kernel_size, activation=activation, padding="same"),
            tf.keras.layers.MaxPooling2D(pool_size=pool_size),
            tf.keras.layers.Conv2D(64, kernel_size=kernel_size, activation=activation, padding="same"),
            tf.keras.layers.MaxPooling2D(pool_size=pool_size)
        ]
    elif i == 3:
        return [
            tf.keras.layers.Conv2D(16, kernel_size=kernel_size, activation=activation, padding="same"),
            tf.keras.layers.MaxPooling2D(pool_size=pool_size),
            tf.keras.layers.Conv2D(32, kernel_size=kernel_size, activation=activation, padding="same"),
            tf.keras.layers.MaxPooling2D(pool_size=pool_size),
            tf.keras.layers.Conv2D(64, kernel_size=kernel_size, activation=activation, padding="same"),
            tf.keras.layers.MaxPooling2D(pool_size=pool_size),
            tf.keras.layers.Conv2D(128, kernel_size=kernel_size, activation=activation, padding="same"),
            tf.keras.layers.MaxPooling2D(pool_size=pool_size),
            tf.keras.layers.Conv2D(256, kernel_size=kernel_size, activation=activation, padding="same"),
            tf.keras.layers.MaxPooling2D(pool_size=pool_size)
        ]


default_batch_size = 16
model_n = 0
for epochs_index in range(0, 2):
    for dense_layers_index in range(0, 5):
        for conv_layers_index in range(0, 4):
            model_n += 1
            epochs = get_current_epochs(epochs_index)
            dense_layers = get_dense_layers(dense_layers_index)
            conv_layers = get_convolutional_layers(conv_layers_index)

            model_tag = tag(model_n, len(dense_layers), len(conv_layers), epochs)
            if os.path.exists("models_data/" + model_tag + ".png"):
                continue

            try:
                recognizer = ImageRecognizer(
                    model_tag=model_tag,
                    dataset_dir='dataset_1/dataset_sorted',
                    batch_size=default_batch_size,
                    model_save_folder='dataset_1/models/model_' + str(model_n),
                    layers=[
                        tf.keras.layers.Rescaling(1. / 255, input_shape=(240, 240, 3)),
                        *conv_layers,
                        tf.keras.layers.Flatten(),
                        *dense_layers,
                        tf.keras.layers.Dense(2, activation='softmax')
                    ],
                    dataset_tag="dataset_1"
                )
                recognizer.train(epochs=epochs, save_model=True, plot_training_data=True)
                recognizer.evaluate_on_unknown_dataset(silent=False)
            except:
                print("Error occurred, continuing. See logs for more information")
                logger.Logger(
                    "dataset_1/models_data/logs.txt").log("\nError in " + model_tag + "| Process continued, skipped "
                                                                                      "model.\n")
                logger.Logger("dataset_1/models_data/logs.txt").log(
                    "\n!!!!!!!!!!")
                continue
"""


ncnn_1_model = ImageRecognizer(
    dataset_dir='dataset_test',
    model_save_folder='models/v1',
    model_tag='v1_non_convolutional=1_hl=no_rescaling_model',
    batch_size=8,
    layers=[
        tf.keras.layers.Rescaling(1. / 255, input_shape=(240, 240, 3)),
        tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), activation=tf.nn.relu),
        tf.keras.layers.MaxPool2D(pool_size=(2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(units=128, activation="relu"),
        tf.keras.layers.Dense(units=128, activation="relu"),
        tf.keras.layers.Dense(units=128, activation="relu"),
        tf.keras.layers.Dense(units=2, activation="softmax")
    ]
)

ncnn_1_model.train(plot_training_data=False, epochs=3, save_model=True)
ncnn_1_model.evaluate_on_unknown_dataset(silent=True)

# Model 1: 3 hidden layers, no filters, no additional rescaling to preprocess images

ncnn_2_model = ImageRecognizer(
    dataset_dir='dataset_sorted',
    model_save_folder='models/v1',
    model_tag='v1 non-convolutional 3-hl no rescaling model',
    batch_size=8,
    layers=[
        tf.keras.layers.Flatten(input_shape=(240, 240, 3)),
        tf.keras.layers.Dense(units=128, activation="relu"),
        tf.keras.layers.Dense(units=256, activation="relu"),
        tf.keras.layers.Dense(units=512, activation="relu"),
        tf.keras.layers.Dense(units=2, activation="softmax")
    ]
)

ncnn_2_model.train(plot_training_data=True, epochs=10, save_model=True)
ncnn_2_model.evaluate_on_unknown_dataset(silent=False)

ncnn_3_model = ImageRecognizer(
    dataset_dir='dataset_sorted',
    model_save_folder='models/v1',
    model_tag='v1 non-convolutional 1-3 no rescaling model',
    batch_size=8,
    layers=[
        tf.keras.layers.Flatten(input_shape=(240, 240, 3)),
        tf.keras.layers.Dense(units=128, activation="relu"),
        tf.keras.layers.Dense(units=256, activation="relu"),
        tf.keras.layers.Dense(units=512, activation="relu"),
        tf.keras.layers.Dense(units=2, activation="softmax")
    ]
)

ncnn_3_model.train(plot_training_data=True, epochs=10, save_model=True)
ncnn_3_model.evaluate_on_unknown_dataset(silent=False)
model = tf.keras.models.Sequential([
    tf.keras.layers.RandomCrop(200, 200, input_shape=(240, 240, 3)),
    tf.keras.layers.Rescaling(1. / 255),
    tf.keras.layers.Conv2D(64, 3, activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(128, 3, activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(2000, activation='relu'),
    tf.keras.layers.Dense(5000, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(2, activation='softmax')
])
"""
