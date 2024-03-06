import tensorflow as tf

import tumor_classification


dataset_training: tf.data.Dataset = tf.keras.utils.image_dataset_from_directory(
    directory='dataset_2/Training',
    seed=123,
    image_size=(512, 512),
)
dataset_training = dataset_training.cache()
dataset_training = dataset_training.prefetch(buffer_size=tf.data.AUTOTUNE)

kernel_size = (3, 3)
pool_size = (2, 2)
epochs = 40


layers = [
    tf.keras.layers.Rescaling(1. / 255, input_shape=(512, 512, 3)),

    tf.keras.layers.Conv2D(16, kernel_size=kernel_size, activation=tf.keras.activations.relu, padding="same"),
    tf.keras.layers.MaxPooling2D(pool_size=pool_size),
    tf.keras.layers.Conv2D(32, kernel_size=kernel_size, activation=tf.keras.activations.relu, padding="same"),
    tf.keras.layers.MaxPooling2D(pool_size=pool_size),
    tf.keras.layers.Conv2D(64, kernel_size=kernel_size, activation=tf.keras.activations.relu, padding="same"),
    tf.keras.layers.MaxPooling2D(pool_size=pool_size),
    tf.keras.layers.Conv2D(128, kernel_size=kernel_size, activation=tf.keras.activations.relu, padding="same"),
    tf.keras.layers.MaxPooling2D(pool_size=pool_size),
    tf.keras.layers.Conv2D(256, kernel_size=kernel_size, activation=tf.keras.activations.relu, padding="same"),
    tf.keras.layers.MaxPooling2D(pool_size=pool_size),

    tf.keras.layers.Flatten(),

    tf.keras.layers.Dense(units=64, activation=tf.keras.activations.relu),
    tf.keras.layers.Dense(units=128, activation=tf.keras.activations.relu),
    tf.keras.layers.Dense(units=256, activation=tf.keras.activations.relu),

    tf.keras.layers.Dense(units=4, activation=tf.keras.activations.softmax)
]

model_n = str(2)

model = tumor_classification.ImageRecognizer(
    dataset_dir="dataset_2/Training",
    model_save_folder="dataset_2/models/model_" + model_n,
    layers=layers,
    model_tag="model_" + model_n + "_d_3_conv_10_ep_" + str(epochs),
    batch_size=32,
    compile_information=[tf.keras.optimizers.Adam(), tf.keras.losses.SparseCategoricalCrossentropy(),
                         [tf.keras.metrics.categorical_accuracy]],
    img_size=(512, 512),
    dataset_tag="dataset_2"
)

model.train(False, epochs, True)

model.evaluate_on_unknown_dataset("dataset_2/Testing")
