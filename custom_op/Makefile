CC=g++
NVC=nvcc
#ABI changes in GCC5.1 will result in undefined references to symbols
#(see https://gcc.gnu.org/onlinedocs/libstdc++/manual/using_dual_abi.html)
#When compiling with gcc>=5 using pip installed tensorflow 
CFLAGS=-Wall -std=c++11 -shared -fPIC -D GOOGLE_CUDA=1
NVCFLAGS= -std=c++11 -c -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC

#For tensorflow compiled with newer ABI or compiling using gcc<=4.9
#CFLAGS=-Wall -std=c++11 -shared -fPIC

#Output library name
TFOP=unroll_op
SOURCES=$(wildcard *.cc)

#Get location of Tensorflow headers and library files
TF_INC=$(shell python -c 'import tensorflow as tf; print(tf.sysconfig.get_include())')
TF_LIB=$(shell python -c 'import tensorflow as tf; print(tf.sysconfig.get_lib())')

#Use this if building for Tensorflow 1.2 or 1.3
#TF_CFLAGS=-I$(TF_INC) -I$(TF_INC)/external/nsync/public
#Tensorflor 1.4 added libtensorflow_framework.so
TF_CFLAGS=-I$(TF_INC) -I$(TF_INC)/external/nsync/public -L$(TF_LIB) -ltensorflow_framework

all: unroll_op.so

unroll_op.so: unroll_op.cu.o unroll_op.cc
	$(CC) $(CFLAGS) $^ $(TF_CFLAGS) -o $@

unroll_op.cu.o: unroll_op.cu.cc
	$(NVC) $(NVCFLAGS) $^ $(TF_CFLAGS) -o $@

clean:
	rm -f unroll_op.so unroll_op.cu.o