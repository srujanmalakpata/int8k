# Cross-compile for 64-bit ARM Linux with the Debian/Ubuntu cross toolchain:
#   sudo apt-get install g++-aarch64-linux-gnu qemu-user libgtest-dev
# The x86 AVX2 kernels compile out (INT8K_X86 is 0), so this checks the portable path on a real
# ARM target. If qemu-aarch64 is installed, CTest runs every test binary through it.
set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR aarch64)
set(CMAKE_C_COMPILER aarch64-linux-gnu-gcc)
set(CMAKE_CXX_COMPILER aarch64-linux-gnu-g++)

set(CMAKE_FIND_ROOT_PATH /usr/aarch64-linux-gnu)
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
# The host's GoogleTest is x86, so never pick it up; the arm64-cross preset builds GoogleTest
# from source instead (FetchContent pointed at the apt-installed /usr/src/googletest).
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)

find_program(INT8K_QEMU_AARCH64 qemu-aarch64)
if(INT8K_QEMU_AARCH64)
  set(CMAKE_CROSSCOMPILING_EMULATOR "${INT8K_QEMU_AARCH64};-L;/usr/aarch64-linux-gnu")
endif()
