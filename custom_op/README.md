Run `compile.sh` to create the shared library.

The Makefile does not compile the op correctly for some reason, so only use it for `make clean`, which removes `normxcorr.so` and `unroll_op.cu.o`. 

test.py contains some basic boileplate code to test the op's functionality - its output should be identical to running `python Xcorr_gpu_opt.py` from the main folder. 