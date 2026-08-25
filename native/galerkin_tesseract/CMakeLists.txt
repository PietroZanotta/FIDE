cmake_minimum_required(VERSION 3.20)
project(mfsi_galerkin_native LANGUAGES CXX)

find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
execute_process(
    COMMAND "${Python_EXECUTABLE}" -m pybind11 --cmakedir
    OUTPUT_VARIABLE pybind11_DIR OUTPUT_STRIP_TRAILING_WHITESPACE
    COMMAND_ERROR_IS_FATAL ANY)
find_package(pybind11 CONFIG REQUIRED)
find_package(OpenMP REQUIRED COMPONENTS CXX)
find_package(BLAS REQUIRED)
execute_process(
    COMMAND "${Python_EXECUTABLE}" -c
            "import pathlib, scipy; print(next((pathlib.Path(scipy.__file__).parent.parent / 'scipy.libs').glob('libscipy_openblas-*.so')))"
    OUTPUT_VARIABLE SCIPY_OPENBLAS_LIBRARY OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE SCIPY_OPENBLAS_RESULT)

pybind11_add_module(_galerkin_native src/bindings.cpp)
target_compile_features(_galerkin_native PRIVATE cxx_std_17)
target_compile_options(_galerkin_native PRIVATE
    $<$<CXX_COMPILER_ID:GNU,Clang>:-O3;-march=native;-Wall;-Wextra;-Wpedantic>)
target_link_libraries(_galerkin_native PRIVATE OpenMP::OpenMP_CXX BLAS::BLAS)
if(SCIPY_OPENBLAS_RESULT EQUAL 0 AND EXISTS "${SCIPY_OPENBLAS_LIBRARY}")
    target_link_libraries(_galerkin_native PRIVATE "${SCIPY_OPENBLAS_LIBRARY}")
    target_compile_definitions(_galerkin_native PRIVATE MFSI_HAVE_SCIPY_OPENBLAS=1)
    get_filename_component(SCIPY_OPENBLAS_DIR "${SCIPY_OPENBLAS_LIBRARY}" DIRECTORY)
    set_property(TARGET _galerkin_native APPEND PROPERTY BUILD_RPATH "${SCIPY_OPENBLAS_DIR}")
endif()

