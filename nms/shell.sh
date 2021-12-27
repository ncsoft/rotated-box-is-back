TF_CFLAGS=( $(python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))') )
TF_LFLAGS=( $(python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))') )
nvcc -std=c++11 -c -o nms_kernel.cu.o nms_kernel.cu \
  ${TF_CFLAGS[@]} -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC

nvcc -std=c++11 -c -o rbox_nms_kernel.cu.o rbox_nms_kernel.cu\
  ${TF_CFLAGS[@]} -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC
g++ -std=c++11 -shared -o ./nms_kernel_op.so nms_kernel.cc nms_kernel_ops.cc \
  nms_kernel.cu.o rbox_nms_kernel.cu.o ${TF_CFLAGS[@]} -L/usr/local/cuda/lib64 -fPIC -lcudart ${TF_LFLAGS[@]}
