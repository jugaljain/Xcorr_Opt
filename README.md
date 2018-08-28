# Xcorr_Opt

Project based on [this paper](http://papers.nips.cc/paper/6367-deep-neural-networks-with-inexact-matching-for-person-re-identification.pdf).

Specifically, implementing the normalized cross-correlation (normxcorr) layer described in the paper as a TensorFlow custom GPU op (code found in subdirectory `custom_op`), which also contains instructions for compilation in its own README.

This project was initially developed in PyCuda, which was chosen for ease of development. It was later migrated to a TF custom op to implement and test out the gradient calculation, which is currently under development. (PyCuda code is in `Xcorr_gpu_opt.py`) It performs the normxcorr operation on two feature map sets of size 25x37x12, harnessing the massive parallelization power of a GPU to perform 49.95 million operations in about 0.9 milliseconds.  

Note: this custom op is made only for the GPU, and requires the relevant NVidia drivers be installed (CUDA 9.0 required). A concurrent CPU program operating on randomly generated numpy arrays can be found in `test/Xcorr_np.py`, however a tensorflow version of it was deemed far too slow to be of any practical use, thus was not developed further. A failed prototype of it is found in `Xcorr_tf_opt.py`.  