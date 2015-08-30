mnist:
	scala -d target/scala-2.10/classes/ classification.MNIST train-images.idx3-ubyte train-labels.idx1-ubyte t10k-images.idx3-ubyte t10k-labels.idx1-ubyte

.PHONY: mnist