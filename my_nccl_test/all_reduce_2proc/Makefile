# Makefile — 编译多进程 (一卡一进程) all_reduce 测试
#
# 使用:
#   make                 # 构建可执行 all_reduce_2proc
#   make clean           # 删除产物
#   make help            # 显示这段说明
#
# 可覆盖变量:
#   NCCL_HOME  默认 ../../nccl/build    (build_nccl.sh 的产物)
#   CUDA_HOME  默认 /usr/local/cuda-12.9
#   GENCODE    默认 sm_80 (A100)

REPO_ROOT := $(abspath $(dir $(lastword $(MAKEFILE_LIST)))/../..)
NCCL_HOME ?= $(REPO_ROOT)/nccl/build
CUDA_HOME ?= /usr/local/cuda-12.9
NVCC      := $(CUDA_HOME)/bin/nvcc
GENCODE   ?= -gencode=arch=compute_80,code=sm_80

TARGET := all_reduce_2proc

.PHONY: all clean help

all: $(TARGET)

$(TARGET): $(TARGET).cu
	$(NVCC) $(GENCODE) -O2 -std=c++14 \
	    -I$(NCCL_HOME)/include \
	    $< -o $@ \
	    -L$(NCCL_HOME)/lib -lnccl \
	    -L$(CUDA_HOME)/lib64 -lcudart

clean:
	rm -f $(TARGET)

help:
	@awk 'NR>1 && /^#/ {print; next} NR>1 {exit}' $(lastword $(MAKEFILE_LIST))
