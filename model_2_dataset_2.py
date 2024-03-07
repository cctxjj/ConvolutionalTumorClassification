import tensorflow as tf

import tumor_classification

# difference: additional conv and dense layer

kernel_size = (3, 3)
pool_size = (2, 2)
epochs = 20

batch_size = 8


layers = [
    tf.keras.layers.Rescaling(1. / 255, input_shape=(512, 512, 3)),

    tf.keras.layers.Conv2D(16, kernel_size=kernel_size, activation=tf.keras.activations.relu, padding="same"),
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
    tf.keras.layers.Dense(units=256, activation=tf.keras.activations.relu),


    tf.keras.layers.Dense(units=4, activation=tf.keras.activations.softmax)
]

model_n = str(2)

model = tumor_classification.ImageRecognizer(
    dataset_dir="dataset_2/Training",
    model_save_folder="dataset_2/models/model_" + model_n,
    layers=layers,
    model_tag="model_" + model_n + "_dataset_2_ep_" + str(epochs),
    batch_size=batch_size,
    compile_information=["adam", tf.keras.losses.SparseCategoricalCrossentropy(),
                         [tf.keras.metrics.sparse_categorical_accuracy]],
    img_size=(512, 512),
    dataset_tag="dataset_2"
)

model.train(True, epochs, True)

model.evaluate_on_unknown_dataset("dataset_2/Testing")
