cmake_minimum_required(VERSION 3.20)
project(mfsi_iprojection_native LANGUAGES CXX)

find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
execute_process(
    COMMAND "${Python_EXECUTABLE}" -m pybind11 --cmakedir
    OUTPUT_VARIABLE pybind11_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
    COMMAND_ERROR_IS_FATAL ANY
)
find_package(pybind11 CONFIG REQUIRED)
find_package(OpenMP REQUIRED COMPONENTS CXX)

pybind11_add_module(_iprojection_native src/bindings.cpp)
target_compile_features(_iprojection_native PRIVATE cxx_std_17)
target_compile_options(_iprojection_native PRIVATE
    $<$<CXX_COMPILER_ID:GNU,Clang>:-O3;-march=native;-Wall;-Wextra;-Wpedantic>
)
target_link_libraries(_iprojection_native PRIVATE OpenMP::OpenMP_CXX)
