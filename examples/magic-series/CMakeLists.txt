if(NOT BUILD_TESTING)
  return()
endif()

find_package(GTest REQUIRED)

include_directories(../../include .)

add_executable(test-magic-series test-magic-series.cpp)
target_compile_options(test-magic-series PRIVATE -fopenmp)
target_link_libraries(test-magic-series GTest::gtest_main -fopenmp)
gtest_discover_tests(test-magic-series)
