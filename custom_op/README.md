`compile.sh` creates the shared library and supporting object files, stores them in a newly created `build` folder.

The Makefile does not compile the op correctly for some reason, only used for `make clean`, which removes the `build` folder created by `compile.sh`

`test/test.py` contains some basic boileplate code to test the op's functionality - its output should be identical to running `python Xcorr_gpu_opt.py` from the repository's root folder. 