# Desktop build of the ESP32 Needle runtime (same sources as the firmware).
RT := ../firmware/needle_runtime
SRC := main_host.cpp $(wildcard $(RT)/*.cpp)
CXXFLAGS ?= -O2 -g -std=c++17 -Wall -Wextra -Wno-unused-parameter -ffp-contract=off
CHUNK ?= 1

needle_host: $(SRC) $(wildcard $(RT)/*.h)
	$(CXX) $(CXXFLAGS) -DNR_MAX_CHUNK=$(CHUNK) -o $@ $(SRC)

clean:
	rm -f needle_host
.PHONY: clean
