# recamera-lite edge daemon — cross-compile for the SG2002/CV181x (riscv64 musl).
#
# You need the Seeed reCamera SG2002 SDK + the musl cross toolchain on PATH. These are
# NOT vendored here (multi-GB vendor trees). Point the vars below at your SDK checkout,
# or set them in the environment, e.g.:
#
#   export SG200X_SDK_PATH=/path/to/sg2002_recamera_emmc
#   export RECAM_OS_BUILD=/path/to/reCamera-OS        # for SensorSupportList headers
#   export PATH=/path/to/host-tools/.../bin:$PATH     # riscv64-unknown-linux-musl-g++
#   make
#
# See https://github.com/Seeed-Studio/reCamera-OS for the SDK/toolchain. The result is a
# single static-ish `vision` ELF you scp to the device (see ../install.sh).

CROSS  ?= riscv64-unknown-linux-musl-
CXX    := $(CROSS)g++

# --- point these at your SDK (override on the make cmdline or via env) ---
SDK         ?= $(SG200X_SDK_PATH)
MPI         ?= $(SDK)/cvi_mpi
RTSP        ?= $(SDK)/cvi_rtsp/install
REF         ?= $(SDK)/cvi_mpi
RECAM_OS_BUILD ?= $(SDK)/../reCamera-OS
TPU         ?= $(SDK)/install/soc_sg2002_recamera_emmc/tpu_musl_riscv64/cvitek_tpu_sdk

INCS := -I$(MPI)/include -I$(MPI)/include/isp/cv181x \
        -I$(REF)/sample/common -I$(RECAM_OS_BUILD)/SensorSupportList/common \
        -I$(RTSP)/include -I$(TPU)/include
CXXFLAGS := -O2 -std=c++17 -Wall -fPIC -DARCH_CV181X $(INCS)

PIPE_STATIC := $(MPI)/lib/libsample.a $(MPI)/lib/3rd/libini.a
PIPE_DYN := -L$(MPI)/lib -L$(MPI)/lib/3rd -L$(RTSP)/lib \
            -lsys -lvi -lvpss -lvo -lvenc -lvdec -lgdc -lmisc \
            -lcvi_bin -lcvi_bin_isp -lisp -lisp_algo -lae -laf -lawb \
            -lsns_full -lcvi_rtsp
TPU_DYN := -L$(TPU)/lib -lcviruntime -lcvikernel -lcvimath
COMMON  := -lpthread -latomic -lm

all: vision

# vision — the production daemon (RTSP + NPU salience + threat-gated JPEG + UDP events)
# --wrap lets vision.cpp intercept the SDK sensor->Bayer resolvers to fix the OV5647 R/B swap
WRAP := -Wl,--wrap=SAMPLE_COMM_VI_GetDevAttrBySns -Wl,--wrap=SAMPLE_COMM_ISP_GetIspAttrBySns
vision: vision.cpp yolo.cpp yolo.h
	$(CXX) $(CXXFLAGS) -o $@ vision.cpp yolo.cpp $(PIPE_STATIC) $(PIPE_DYN) $(TPU_DYN) $(COMMON) $(WRAP)

clean:
	rm -f vision

.PHONY: all clean
