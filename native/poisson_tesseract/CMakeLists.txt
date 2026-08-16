cmake_minimum_required(VERSION 3.20)
project(mfsi_poisson_native LANGUAGES CXX)

find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
execute_process(
    COMMAND "${Python_EXECUTABLE}" -m pybind11 --cmakedir
    OUTPUT_VARIABLE pybind11_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
    COMMAND_ERROR_IS_FATAL ANY
)
find_package(pybind11 CONFIG REQUIRED)
find_package(OpenMP REQUIRED COMPONENTS CXX)

pybind11_add_module(_poisson_native
    src/bindings.cpp
    src/poisson_solver.cpp
)
target_compile_features(_poisson_native PRIVATE cxx_std_17)
target_compile_options(_poisson_native PRIVATE
    $<$<CXX_COMPILER_ID:GNU,Clang>:-O3;-march=native;-Wall;-Wextra;-Wpedantic>
)
target_link_libraries(_poisson_native PRIVATE OpenMP::OpenMP_CXX)

