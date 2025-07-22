cmake_minimum_required(VERSION 3.5)

project(sventt LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)

add_compile_options(-W -Wall -Wextra)
set(CMAKE_CXX_FLAGS_DEBUG "-g -fno-omit-frame-pointer -fsanitize=address")

include(CTest)
include(GNUInstallDirs)

find_package(Boost REQUIRED)

add_subdirectory(include)
add_subdirectory(tests)
add_subdirectory(examples)
