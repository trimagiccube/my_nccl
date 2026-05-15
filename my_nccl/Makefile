# Makefile — my_nccl bootstrap 最小子集
#
# 使用:
#   make                 构建 my_nccl_bootstrap (默认带 ASan + UBSan + 调试)
#   make SANITIZE=       关 sanitizer (release-ish)
#   make clean
#   make help

CUDA_HOME ?= /usr/local/cuda-12.9

CXX       := g++
SANITIZE  ?= -fsanitize=address,undefined -fno-omit-frame-pointer
CXXFLAGS  := -O0 -g -std=c++17 -Iinclude -I$(CUDA_HOME)/include $(SANITIZE) \
             -Wno-format-truncation -Wno-stringop-overflow
LDFLAGS   := -L$(CUDA_HOME)/lib64 -lcudart -lpthread -ldl $(SANITIZE)

VENDOR_SRCS := $(wildcard src/vendor/*.cc)
STUB_SRCS   := $(wildcard src/stubs/*.cc)
ALL_SRCS    := src/main.cc $(VENDOR_SRCS) $(STUB_SRCS)
ALL_OBJS    := $(ALL_SRCS:.cc=.o)

TARGET := my_nccl_bootstrap

.PHONY: all clean help

all: $(TARGET)

$(TARGET): $(ALL_OBJS)
	$(CXX) $^ -o $@ $(LDFLAGS)

%.o: %.cc
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -f $(ALL_OBJS) $(TARGET)

help:
	@awk 'NR>1 && /^#/ {print; next} NR>1 {exit}' $(lastword $(MAKEFILE_LIST))
