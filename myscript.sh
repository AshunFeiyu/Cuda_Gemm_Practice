nvcc -o gemm gemm.cu -lcublas
ncu -k gemm -o profile -f ./gemm 1024 1024 1024
ncu -i profile.ncu-rep
ncu --set detailed -k gemm -o profile -f ./gemm 1024 1024 1024
# ./gemm 4096 4096 4096