from adversarial_defense.simpleCNN import SimpleCNN
from adversarial_defense.datagen import generate_adversarial_batch
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt
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

result = dict()
for i in range(1,11,2):
	eps = 0.01 * i
	print("\n\nFor epsilon =", eps)
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

	simple_model = copy.deepcopy(model)
	# generate a set of adversarial from our test set
	print("Generating adversarial examples with FGSM  (eps =", (eps), ")...\n")
	(advX, advY) = next(generate_adversarial_batch(model, len(testX),
		testX, testY, (28, 28, 1), eps=(eps)))
	# re-evaluate the model on the adversarial images
	(loss, acc) = model.evaluate(x=advX, y=advY, verbose=0)
	print("Adversarial testing images  (eps =", (eps), "):")
	print("Loss: {:.4f}, Acc: {:.4f}\n".format(loss, acc))

	
	print("Re-compiling model...")
	opt = Adam(lr=1e-4)
	model.compile(loss="categorical_crossentropy", optimizer=opt,
		metrics=["accuracy"])
	# fine-tune our CNN on the adversarial images
	print("Fine-tuning network on adversarial examples  (eps =", (eps), ")...")
	model.fit(advX, advY,
		batch_size=64,
		epochs=10,
		verbose=1)

	(loss, acc) = model.evaluate(x=testX, y=testY, verbose=0)
	print()
	print("Normal testing images after fine-tuning:")
	print("Loss: {:.4f}, Acc: {:.4f}\n".format(loss, acc))

	xpoints = []
	ypoints = []
	for j in range(0,15,2):
		eps_2 = 0.01 * j
		print("Generating adversarial test samples with FGSM  (eps =", (eps_2), ")...\n")
		(advX, advY) = next(generate_adversarial_batch(simple_model, len(testX),
													   testX, testY, (28, 28, 1), eps=(eps_2)))
		# do a final evaluation of the model on the adversarial images
		(loss, acc) = model.evaluate(x=advX, y=advY, verbose=0)

		xpoints.append(eps_2)
		ypoints.append(acc)
		print("Adversarial images after fine-tuning:")
		print("Loss: {:.4f}, Acc: {:.4f}".format(loss, acc))
	result[eps] = list(zip(xpoints,ypoints))
	xpoints = np.array(xpoints)
	ypoints = np.array(ypoints)
	plt.plot(xpoints,ypoints,label = eps)
plt.xlabel("test epsilon (amount of noise)")
plt.ylabel("Test Accuarcy")
plt.legend()
plt.show()
print(result)
