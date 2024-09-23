if(NOT BUILD_TESTING)
  return()
endif()

find_package(GTest REQUIRED)
find_package(benchmark REQUIRED)

# Not using FindOpenMP.cmake to support static linking.
set(SVENTT_CXX_FLAGS -fopenmp)
set(SVENTT_LINK_LIBRARIES Boost::boost -fopenmp)

include_directories(../include .)

add_executable(test-ntt-reference test-ntt-reference.cpp)
target_link_libraries(test-ntt-reference Boost::boost GTest::gtest_main)
gtest_discover_tests(test-ntt-reference)

add_executable(test-modulus test-modulus.cpp)
target_link_libraries(test-modulus Boost::boost GTest::gtest_main)
gtest_discover_tests(test-modulus)

add_executable(bench-stream-cmg bench-stream-cmg.cpp)
target_compile_options(bench-stream-cmg PRIVATE -fopenmp)
target_link_libraries(bench-stream-cmg benchmark::benchmark_main -fopenmp)
add_test(bench-stream-cmg bench-stream-cmg)

add_executable(bench-transpose bench-transpose.cpp)
target_compile_options(bench-transpose PRIVATE ${SVENTT_CXX_FLAGS})
target_link_libraries(bench-transpose ${SVENTT_LINK_LIBRARIES} benchmark::benchmark_main)
add_test(bench-transpose bench-transpose)

set(ntt_test_list
    # iterative-scalar-radix2-two5
    # iterative-scalar-radix2-two10
    # iterative-scalar-radix4-two10
    # iterative-scalar-radix24-two11
    # iterative-scalar-radix8-two12
    # iterative-scalar-radix248-two13
    # recursive-scalar-radix2-two5
    # recursive-scalar-radix24-two11
    # recursive-scalar-radix248-two13
    # recursive-scalar-fourstep-two13
    iterative-sve-radix2-two10
    iterative-sve-radix4-two12
    iterative-sve-radix8-two12
    recursive-sve-radix248-two13
    recursive-sve-fourstep-two13)
foreach(ntt_test IN LISTS ntt_test_list)
  add_executable(test-ntt-${ntt_test} bench-ntt.cpp)
  target_compile_definitions(
    test-ntt-${ntt_test}
    PRIVATE
      NTT_TEST_CASE_FILE="${CMAKE_CURRENT_SOURCE_DIR}/ntt-tests/${ntt_test}.hpp"
  )
  target_compile_options(test-ntt-${ntt_test} PRIVATE ${SVENTT_CXX_FLAGS})
  target_link_libraries(test-ntt-${ntt_test} ${SVENTT_LINK_LIBRARIES}
                        benchmark::benchmark)
  add_test(test-ntt-${ntt_test} test-ntt-${ntt_test})
endforeach()
