import tensorflow as tf

model: tf.keras.Sequential = tf.keras.models.load_model('dataset_2/models/model_11/model_11_dataset_2_ep_20')


ds_test = tf.keras.utils.image_dataset_from_directory(
        directory='dataset_2/Testing',
        seed=123,
        image_size=(512, 512),

    )

ds_test = ds_test.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)


print(model.evaluate(ds_test))


def fetch_data(direction: str):
    """
    Fetches the data from the given directory as a tf.data.Dataset object
    :return:
    """
    dataset = tf.keras.utils.image_dataset_from_directory(
        directory=direction,
        seed=123,
        image_size=(512, 512),

    )

    return dataset.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)

