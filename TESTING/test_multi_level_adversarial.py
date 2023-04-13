from adversarial_defense.simpleCNN import SimpleCNN
from adversarial_defense.datagen import generate_adversarial_batch
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt
import numpy as np
import copy

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

# train the simpleCNN on mnist
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
simple_model = copy.deepcopy(model)
for i in range(1, 11):
    eps = 0.01 * i
    print("Generating adversarial examples with FGSM...(", eps, ")\n")
    (advX, advY) = next(generate_adversarial_batch(simple_model, len(testX),
        testX, testY, (28, 28, 1), eps=eps))
    # re-evaluate the model on the adversarial images
    (loss, acc) = model.evaluate(x=advX, y=advY, verbose=0)
    print("Adversarial testing images:")
    print("Loss: {:.4f}, Acc: {:.4f}\n".format(loss, acc))

    print("Re-compiling model...")
    opt = Adam(lr=1e-4)
    model.compile(loss="categorical_crossentropy", optimizer=opt,
        metrics=["accuracy"])
    print("Fine-tuning network on adversarial examples...")
    model.fit(advX, advY,
        batch_size=64,
        epochs=10,
        verbose=1)

for i in range(10, 0, -1):
    eps = 0.01 * i
    print("Generating adversarial examples with FGSM...\n")
    (advX, advY) = next(generate_adversarial_batch(simple_model, len(testX),
        testX, testY, (28, 28, 1), eps=eps))
    # re-evaluate the model on the adversarial images
    (loss, acc) = model.evaluate(x=advX, y=advY, verbose=0)
    print("Adversarial testing images:")
    print("Loss: {:.4f}, Acc: {:.4f}\n".format(loss, acc))

    print("Re-compiling model...")
    opt = Adam(lr=1e-4)
    model.compile(loss="categorical_crossentropy", optimizer=opt,
        metrics=["accuracy"])
    # fine-tune our CNN on the adversarial images
    print("Fine-tuning network on adversarial examples...")
    model.fit(advX, advY,
        batch_size=64,
        epochs=10,
        verbose=1)

(loss, acc) = model.evaluate(x=testX, y=testY, verbose=0)
print()
print("Normal testing images after fine-tuning:")
print("Loss: {:.4f}, Acc: {:.4f}\n".format(loss, acc))

# do a final evaluation of the model on the adversarial images
xpoints = []
ypoints = []
for i in range(1, 15, 2):
    eps = 0.01 * i
    print("Generating adversarial examples with FGSM (eps =", (eps), ")...\n")
    (advX, advY) = next(generate_adversarial_batch(simple_model, len(testX),
        testX, testY, (28, 28, 1), eps=(eps)))
    # re-evaluate the model on the adversarial images
    (loss, acc) = model.evaluate(x=advX, y=advY, verbose=0)
    print("Adversarial testing images (eps =", (eps), "):")
    print("Loss: {:.4f}, Acc: {:.4f}\n".format(loss, acc))
    xpoints.append(eps)
    ypoints.append(acc)
print(list(zip(xpoints,ypoints)))
plt.xlabel("epsilon (amount of noise)")
plt.ylabel("Test Accuarcy")
plt.plot(np.array(xpoints),np.array(ypoints))
plt.show()