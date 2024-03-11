import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt

from logger import Logger


# testing:
# processor = ImageProcessor("dataset_raw/set_s", "dataset_raw/img_labels.csv")
# labels = processor.get_according_labels()
# print(labels)
# for index, image in enumerate(processor.get_translated_array()):
#    plt.imshow(image)
#    print(str(index) + " has label " + str(labels[index]))
#    plt.show()

class ImageRecognizer:
    def __init__(self,
                 dataset_dir: str,
                 model_save_folder: str,
                 layers: list,
                 model_tag: str,
                 batch_size: int = 16,
                 compile_information=None,
                 img_size: tuple = (240, 240),
                 dataset_tag: str = "dataset_1"
                 ):

        """
        Creates a new ImageRecognizer, which essentially is a keras Sequential with some additional steps to
        train and evaluate performance
        :param dataset_dir: list of strings with the size of 2, containing the directions for trainingimages
        :param model_save_folder: string with the name of the folder the model should be saved in after training,
         to be nulled in case the model should not be saved after training.
        :param layers: list of tensorflow layers to use in the model
        :param model_tag: unique name of the model
        :param batch_size: amount of images which are used together to train the model, defaults to 16 in one batch
        :param compile_information: additional data for compiling the Sequential, should be a list in the
         form [optimizer, loss function]
        """

        if compile_information is None:
            compile_information = ['adam', tf.keras.losses.SparseCategoricalCrossentropy(),
                                   [tf.keras.metrics.SparseCategoricalAccuracy()]]
        self.compile_info = compile_information
        self.dataset_dir = dataset_dir
        self.model_tag = model_tag
        self.img_size = img_size
        self.dataset_tag = dataset_tag
        self.batch_size = batch_size
        self.model_save_dir = model_save_folder + "/" + model_tag

        self.logger = Logger(dataset_tag + "/models_data/logs_2.txt")

        self.logger.log("\nLog for " + model_tag + "\n")

        self.__load_data__()
        self.__setup_model__(layers=layers, optimizer=compile_information[0], loss=compile_information[1],
                             metrics=compile_information[2])

    def __setup_model__(self,
                        layers,
                        optimizer,
                        loss,
                        metrics: list):

        """
        Sets up the model with the given layers and compiles afterward.
        For information about the parameters see __init__()
        :return:
        """

        self.model = tf.keras.models.Sequential(layers)
        self.model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
        self.logger.log("Successfully compiled model\n")
        print(self.model.summary())

    def __load_data__(self):
        """
        Loads the training data from the given directories in __init__()
        :return:
        """

        ds_training = tf.keras.utils.image_dataset_from_directory(
            directory=self.dataset_dir,
            seed=123,
            image_size=self.img_size,
            batch_size=self.batch_size
        )

        self.ds_training = ds_training.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)

    def train(self,
              plot_training_data: bool = True,
              epochs: int = 10,
              save_model: bool = True):

        """
        Trains the model on the given dataset_2
        :param plot_training_data: boolean whether to plot the loss-development during training
        :param epochs: int representing the amount of epochs to train the model
        :param save_model: boolean whether to safe the model after the training
        :return:
        """

        assert self.ds_training is not None

        training_history = self.model.fit(
            self.ds_training,
            epochs=epochs
        )
        self.logger.log("Training loss: ")
        for value in training_history.history['loss']:
            self.logger.log(str(value) + ";")
        self.logger.log("\n ")
        if plot_training_data:
            self.plot_loss_development(training_history.history['loss'])

        if save_model:
            self.model.save(self.model_save_dir)

    def plot_loss_development(self,
                              vals: list,
                              metric_name: str = 'loss'
                              ):

        """
        Plots the loss development based on the history returned from the model
        :param vals: list of values to plot
        :param metric_name: name of the metric, will most likely always be set to 'loss'
        :return:
        """

        ax = plt.subplot()

        x_axis = []
        for i in range(len(vals)):
            x_axis.append(i + 1)

        ax.plot(x_axis, vals)
        ax.set(xlabel="epoche", ylabel=metric_name, title=metric_name + "-development of " + self.model_tag)

        plt.savefig(self.dataset_tag + "/models_data/" + self.model_tag + ".png")
        plt.show()

    def evaluate_on_unknown_dataset(self,
                                    dataset_test_dir=None,
                                    silent: bool = False):

        """
        Evaluates the model on an unknown dataset,
        :param dataset_test_dir: list of directories containing the test images
        :param silent: boolean whether to print out loud the evaluation process
        :return:
        """

        if dataset_test_dir is None:
            dataset_test_dir = "dataset_1/dataset_test"

        evaluator = ModelPerformanceEvaluator(recognizer=self, test_dir=dataset_test_dir, img_size=self.img_size)
        return evaluator.evaluate(silent)


class ModelPerformanceEvaluator:

    def __init__(self,
                 recognizer: ImageRecognizer,
                 test_dir: list,
                 img_size: tuple = (240, 240)
                 ):
        """
        Instantiate a ModelPerformanceEvaluator for testing accuracy of a model on test images.
        :param recognizer: ImageRecognizer object containing the trained model, used mostly for its tag
        :param test_dir: Path to the test images directory
        """

        self.recognizer = recognizer
        self.model = recognizer.model
        self.img_size = img_size
        self.test_dir = test_dir
        self.__fetch_data__()

    def __fetch_data__(self):
        """
        Fetches the data from the given directory as a tf.data.Dataset object
        :return:
        """
        ds_test = tf.keras.utils.image_dataset_from_directory(
            directory=self.test_dir,
            seed=123,
            image_size=self.img_size,

        )

        self.ds_test = ds_test.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)

    def evaluate(self,
                 silent: bool = False):
        """
        Evaluates the models accuracy on new data given in the constructor
        :param silent: boolean whether to print out information about the evaluation process
        :return:
        """

        evaluation = self.model.evaluate(self.ds_test)
        self.recognizer.logger.log("Evaluation data: " + str(tuple(zip(self.model.metrics_names, evaluation))))
        if not silent:
            print(tuple(zip(self.model.metrics_names, evaluation)))
        self.recognizer.logger.log("\n-------\n")
        return evaluation

    def predict_specific_image(self,
                               image_path):
        """
        Predicts specific image
        :param image_path:
        :return:
        """

        raw_image = tf.keras.utils.load_img(image_path, target_size=self.img_size)
        image_array = tf.keras.utils.img_to_array(raw_image)
        image = tf.expand_dims(image_array, 0)
        prediction = np.argmax(self.model.predict(image, verbose=0))
        print(prediction)
        return prediction
