from tensorflow.keras.datasets import mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

from tensorflow import keras
from tensorflow.keras import layers
network = keras.Sequential([
                            layers.Dense(512, activation="relu"),
                            layers.Dense(10, activation="softmax")
                            ])
from tensorflow.keras import models
from tensorflow.keras import layers
network = models.Sequential()
network.add(layers.Dense(512, activation='relu', input_shape=(784,)))
network.add(layers.Dense(10, activation='softmax'))

network.compile(optimizer="rmsprop",loss="sparse_categorical_crossentropy",metrics=["accuracy"])

train_images = train_images.reshape((60000, 784))
train_images = train_images.astype("float32") / 255
test_images = test_images.reshape((10000, 784))
test_images = test_images.astype("float32") / 255

network.fit(train_images, train_labels, epochs=5, batch_size=128)

loss, acc = network.evaluate(test_images,test_labels)
network.save("mnist_model.h5")
