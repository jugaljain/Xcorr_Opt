To compile op:

```
$ chmod +x compile.sh
$ ./compile.sh
```
Will create a `build` folder and put in the compiled shared object library, as well as supporting object files.

`Makefile` does not compile the op correctly. Use only for `make clean`, which removes the `build` folder created by `compile.sh`

`test/test.py` contains some basic boileplate code to test the op's functionality - its output should be identical to running `python Xcorr_gpu_opt.py` from the repository's root folder. 



Next Steps:

 1. Define the ReRollA & ReRollB functions in unroll_op_gradient.cu.cc
 2. Modify compile.sh to add compilations unroll_op_gradient.cc and unroll_op_gradient.cu.cc into the compiled normxcorr.so file
 3. Register the gradient of unroll_op to be unroll_op_gradient.
 4. Test the gradient implementation This might help: https://timvieira.github.io/blog/post/2017/04/21/how-to-test-gradient-implementations/ 
 5. Re-integrate the final custom-op into Manideep's person re-id project, further test for improvements in loss and accuracy.
 		a. Try different datasets, compare results.

