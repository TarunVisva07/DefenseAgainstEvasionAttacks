from adversarial_defense.simpleCNN import SimpleCNN
from adversarial_defense.datagen import generate_adversarial_batch
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

for i in range(1, 11):
	print("\n\nFor epsilon =", 0.01*i)
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
	print("Generating adversarial examples with FGSM  (eps =", (0.01*i), ")...\n")
	(advX, advY) = next(generate_adversarial_batch(model, len(testX),
		testX, testY, (28, 28, 1), eps=(0.01*i)))
	# re-evaluate the model on the adversarial images
	(loss, acc) = model.evaluate(x=advX, y=advY, verbose=0)
	print("Adversarial testing images  (eps =", (0.01*i), "):")
	print("Loss: {:.4f}, Acc: {:.4f}\n".format(loss, acc))


	print("Re-compiling model...")
	opt = Adam(lr=1e-4)
	model.compile(loss="categorical_crossentropy", optimizer=opt,
		metrics=["accuracy"])
	# fine-tune our CNN on the adversarial images
	print("Fine-tuning network on adversarial examples  (eps =", (0.01*i), ")...")
	model.fit(advX, advY,
		batch_size=64,
		epochs=10,
		verbose=1)

	(loss, acc) = model.evaluate(x=testX, y=testY, verbose=0)
	print()
	print("Normal testing images after fine-tuning:")
	print("Loss: {:.4f}, Acc: {:.4f}\n".format(loss, acc))

	for j in range(i-3, i+3):
		print("Generating adversarial test samples with FGSM  (eps =", (0.01*j), ")...\n")
		(advX, advY) = next(generate_adversarial_batch(model, len(testX),
													   testX, testY, (28, 28, 1), eps=(0.01*j)))
		# do a final evaluation of the model on the adversarial images
		(loss, acc) = model.evaluate(x=advX, y=advY, verbose=0)
		print("Adversarial images after fine-tuning:")
		print("Loss: {:.4f}, Acc: {:.4f}".format(loss, acc))