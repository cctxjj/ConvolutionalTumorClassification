import tensorflow as tf
import numpy as np
"""
model: tf.keras.models.Sequential = tf.keras.models.load_model("dataset_1/models/model_28/m_28_d_3_conv_10_e_10")
raw_image = tf.keras.utils.load_img("img_6.png", target_size=(240, 240))
image_array = tf.keras.utils.img_to_array(raw_image)
image = tf.expand_dims(image_array, 0)
prediction = np.argmax(model.predict(image, verbose=0))
print(prediction)
"""
dataset_training: tf.data.Dataset = tf.keras.utils.image_dataset_from_directory(
    directory='dataset_2/Training',
    seed=123,
    image_size=(512, 512),
)

class_names = dataset_training.class_names

dataset_training = dataset_training.cache()

'''
plt.figure(figsize=(10, 10))
for images, labels in dataset_training.take(1):
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(class_names[labels[i]])
        plt.axis("off")

plt.show()
'''





