if(NOT BUILD_TESTING)
  return()
endif()

find_package(GTest REQUIRED)

include_directories(../../include .)

add_executable(test-magic-series-kinnaes test-magic-series-kinnaes.cpp)
target_link_libraries(test-magic-series-kinnaes GTest::gtest_main)
gtest_discover_tests(test-magic-series-kinnaes)
