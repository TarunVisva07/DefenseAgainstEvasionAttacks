from adversarial_defense.simpleCNN import SimpleCNN
from adversarial_defense.datagen import generate_adversarial_batch
import matplotlib.pyplot as plt
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import mnist
import numpy as np

print("loading MNIST dataset...")
(trainX, trainY), (testX, testY) = mnist.load_data()
trainX = trainX / 255.0
testX = testX / 255.0
# add a channel dimension to the images
trainX = np.expand_dims(trainX, axis=-1)
testX = np.expand_dims(testX, axis=-1)
# one-hot encode our labels
trainY = to_categorical(trainY, 10)
testY = to_categorical(testY, 10)

print("compiling model...")
opt = Adam(lr=1e-3)
model = SimpleCNN.build(width=28, height=28, depth=1, classes=10)
model.compile(loss="categorical_crossentropy", optimizer=opt,
	metrics=["accuracy"])
# train the simple CNN on MNIST
print("training network...")
model.fit(trainX, trainY,
	validation_data=(testX, testY),
	batch_size=64,
	epochs=10,
	verbose=1)

(loss, acc) = model.evaluate(x=testX, y=testY, verbose=0)
print("Normal testing images:")
print("Loss: {:.4f}, Acc: {:.4f}\n".format(loss, acc))  
# generate a set of adversarial from our test set
xpoints = []
ypoints = []
for i in range(1, 11):
    eps = 0.01 * i
    print("Generating adversarial examples with FGSM (eps =", (10**i), ")...\n")
    (advX, advY) = next(generate_adversarial_batch(model, len(testX),
        testX, testY, (28, 28, 1), eps=eps))
    # re-evaluate the model on the adversarial images

    (loss, acc) = model.evaluate(x=advX, y=advY, verbose=0)
    print("Adversarial testing images (eps =", eps, "):")
    xpoints.append(eps)
    print("Loss: {:.4f}, Acc: {:.4f}\n".format(loss, acc))
    ypoints.append(acc)
print(list(zip(xpoints,ypoints)))
xpoints = np.array(xpoints)
ypoints = np.array(ypoints)
plt.xlabel("epsilon (amount of noise)")
plt.ylabel("Test Accuarcy")
plt.plot(xpoints,ypoints)
plt.show()



