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
epochs = 10


layers = [
    tf.keras.layers.Rescaling(1. / 255, input_shape=(512, 512, 3)),

    tf.keras.layers.Conv2D(16, kernel_size=kernel_size, activation=tf.keras.activations.relu, padding="same"),
    tf.keras.layers.MaxPooling2D(pool_size=pool_size),
    tf.keras.layers.SpatialDropout2D(rate=0.2),
    tf.keras.layers.Conv2D(32, kernel_size=kernel_size, activation=tf.keras.activations.relu, padding="same"),
    tf.keras.layers.MaxPooling2D(pool_size=pool_size),
    tf.keras.layers.SpatialDropout2D(rate=0.2),
    tf.keras.layers.Conv2D(64, kernel_size=kernel_size, activation=tf.keras.activations.relu, padding="same"),
    tf.keras.layers.MaxPooling2D(pool_size=pool_size),
    tf.keras.layers.SpatialDropout2D(rate=0.1),
    tf.keras.layers.Conv2D(128, kernel_size=kernel_size, activation=tf.keras.activations.relu, padding="same"),
    tf.keras.layers.MaxPooling2D(pool_size=pool_size),
    tf.keras.layers.SpatialDropout2D(rate=0.05),
    tf.keras.layers.Conv2D(256, kernel_size=kernel_size, activation=tf.keras.activations.relu, padding="same"),
    tf.keras.layers.MaxPooling2D(pool_size=pool_size),
    tf.keras.layers.SpatialDropout2D(rate=0.05),

    tf.keras.layers.Flatten(),

    tf.keras.layers.Dense(units=64, activation=tf.keras.activations.relu),
    tf.keras.layers.Dense(units=128, activation=tf.keras.activations.relu),
    tf.keras.layers.Dense(units=256, activation=tf.keras.activations.relu),
    tf.keras.layers.Dense(units=256, activation=tf.keras.activations.relu),

    tf.keras.layers.Dense(units=4, activation=tf.keras.activations.softmax)
]

model = tumor_classification.ImageRecognizer(
    dataset_dir="dataset_2/Training",
    model_save_folder="dataset_2/models",
    layers=layers,
    model_tag="model_1_d_3_conv_10_ep_" + str(epochs),
    batch_size=32,
    compile_information=[tf.keras.optimizers.Adam(), tf.keras.losses.SparseCategoricalCrossentropy(),
                         [tf.keras.metrics.categorical_accuracy]],
    img_size=(512, 512),
    dataset_tag="dataset_2"
)

model.train(True, epochs, True)
model.evaluate_on_unknown_dataset("dataset_2/Testing")
