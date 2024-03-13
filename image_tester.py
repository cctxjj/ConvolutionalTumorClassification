import tensorflow as tf
import numpy as np

model: tf.keras.Sequential = tf.keras.models.load_model('dataset_2/models/model_11/model_11_dataset_2_ep_20')

raw_image = tf.keras.utils.load_img('test_imgs/img_3.png', target_size=(512, 512))
image_array = tf.keras.utils.img_to_array(raw_image)
image = tf.expand_dims(image_array, 0)

raw_pred = model.predict(image, verbose=0)
print(raw_pred)
result = np.argmax(raw_pred)
print(result)





