.PHONY: clean

# src=$(wildcard *.cu)
src=gemm_ALL.cu

obj=$(patsubst %.cu,%.out,$(src))

prof=$(patsubst %.out,%.ncu-rep,$(obj))

ALL:$(prof)

$(obj):%.out:%.cu
	nvcc -o $@ $< -lineinfo -lcublas

$(prof):%.ncu-rep:%.out
	ncu --set full --import-source=1 -o $@ -f ./$< 1024 1024 1024

clean:
	-rm -rf $(obj) $(prof)