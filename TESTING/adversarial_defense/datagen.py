from .fgsm import generate_image_adversary
from sklearn.utils import shuffle
import numpy as np
def generate_adversarial_batch(model, total, images, labels, dims,
	eps=0.01):
	# unpack the image dimensions into convenience variables
	(h, w, c) = dims
    	# we're constructing a data generator here so we need to loop
	# indefinitely
	while True:
		# initialize our perturbed images and labels
		perturbImages = []
		perturbLabels = []
		# randomly sample indexes (without replacement) from the
		# input data
		idxs = np.random.choice(range(0, len(images)), size=total,
			replace=False)
            		# loop over the indexes
		for i in idxs:
			# grab the current image and label
			image = images[i]
			label = labels[i]
			# generate an adversarial image
			adversary = generate_image_adversary(model,
				image.reshape(1, h, w, c), label, eps=eps)
			# update our perturbed images and labels lists
			perturbImages.append(adversary.reshape(h, w, c))
			perturbLabels.append(label)
		# yield the perturbed images and labels
		yield (np.array(perturbImages), np.array(perturbLabels))

def generate_mixed_adverserial_batch(model, total, images, labels,
	dims, eps=0.01, split=0.5):
	# unpack the image dimensions into convenience variables
	(h, w, c) = dims
	# compute the total number of training images to keep along with
	# the number of adversarial images to generate
	totalNormal = int(total * split)
	totalAdv = int(total * (1 - split))
		# we're constructing a data generator so we need to loop
	# indefinitely
	while True:
		# randomly sample indexes (without replacement) from the
		# input data and then use those indexes to sample our normal
		# images and labels
		idxs = np.random.choice(range(0, len(images)),
			size=totalNormal, replace=False)
		mixedImages = images[idxs]
		mixedLabels = labels[idxs]
		# again, randomly sample indexes from the input data, this
		# time to construct our adversarial images
		idxs = np.random.choice(range(0, len(images)), size=totalAdv,
			replace=False)
				# loop over the indexes
		for i in idxs:
			# grab the current image and label, then use that data to
			# generate the adversarial example
			image = images[i]
			label = labels[i]
			adversary = generate_image_adversary(model,
				image.reshape(1, h, w, c), label, eps=eps)
			# update the mixed images and labels lists
			mixedImages = np.vstack([mixedImages, adversary])
			mixedLabels = np.vstack([mixedLabels, label])
		# shuffle the images and labels together
		(mixedImages, mixedLabels) = shuffle(mixedImages, mixedLabels)
		# yield the mixed images and labels to the calling function
		yield (mixedImages, mixedLabels)