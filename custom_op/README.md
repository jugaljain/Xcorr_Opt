To compile op:

```
$ chmod +x compile.sh
$ ./compile.sh
```
Will create a `build` folder and put in the compiled shared object library, as well as supporting object files.

`Makefile` does not compile the op correctly. Use only for `make clean`, which removes the `build` folder created by `compile.sh`

`test/test.py` contains some basic boileplate code to test the op's functionality - its output should be identical to running `python Xcorr_gpu_opt.py` from the repository's root folder. 