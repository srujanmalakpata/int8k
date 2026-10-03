# Prefer a system GoogleTest (apt: libgtest-dev, brew: googletest). FetchContent is only a
# fallback for CI runners that lack the package; it needs network access to github.com.
find_package(GTest QUIET)

if(NOT GTest_FOUND)
  if(NOT INT8K_FETCH_GTEST)
    message(FATAL_ERROR
      "GoogleTest not found. Install it (e.g. `sudo apt-get install libgtest-dev`) or "
      "configure with -DINT8K_FETCH_GTEST=ON to download it.")
  endif()
  message(STATUS "GoogleTest not found; fetching v1.14.0 with FetchContent")
  include(FetchContent)
  FetchContent_Declare(googletest
    URL https://github.com/google/googletest/archive/refs/tags/v1.14.0.tar.gz
    URL_HASH SHA256=8ad598c73ad796e0d8280b082cebd82a630d73e73cd3c70057938a6501bba5d7
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE)
  set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
  set(BUILD_GMOCK OFF CACHE BOOL "" FORCE)
  FetchContent_MakeAvailable(googletest)
  if(NOT TARGET GTest::gtest_main)
    add_library(GTest::gtest_main ALIAS gtest_main)
  endif()
endif()

include(GoogleTest)
