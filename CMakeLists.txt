cmake_minimum_required(VERSION 3.20)
project(PyLite LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(UNIX AND NOT APPLE)
  add_link_options(-Wl,--no-as-needed)
endif()

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

find_package(LLVM REQUIRED CONFIG HINTS /usr/lib/llvm-22/lib/cmake/llvm)
message(STATUS "找到 LLVM ${LLVM_PACKAGE_VERSION} (${LLVM_DIR})")

separate_arguments(LLVM_DEFINITIONS_LIST NATIVE_COMMAND "${LLVM_DEFINITIONS}")
add_compile_options(${LLVM_DEFINITIONS_LIST})

add_library(pylite_runtime SHARED
  runtime/error.cpp
  runtime/gc.cpp
  runtime/arith.cpp
  runtime/string.cpp
  runtime/tuple.cpp
  runtime/list.cpp
  runtime/dict.cpp
  runtime/index.cpp
  runtime/method.cpp
  runtime/io.cpp
  runtime/docker.cpp
  runtime/llm.cpp
  runtime/cuda.cpp
  runtime/fusion.cpp
  runtime/inference.cpp
  runtime/tokenizer.cpp
  runtime/server.cpp
  runtime/quantize.cpp
)
target_include_directories(pylite_runtime PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)
target_compile_options(pylite_runtime PRIVATE -fvisibility=default)
target_link_libraries(pylite_runtime PRIVATE pthread dl)

add_library(pylite_frontend STATIC
  frontend/lexer.cpp
  frontend/ast.cpp
  frontend/parser.cpp
)
target_include_directories(pylite_frontend PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/frontend>
)

add_executable(lex_dump tests/lex_dump.cpp)
target_link_libraries(lex_dump PRIVATE pylite_frontend)

add_executable(ast_dump tests/ast_dump.cpp)
target_link_libraries(ast_dump PRIVATE pylite_frontend)

add_executable(frontend_test tests/frontend_test.cpp)
target_link_libraries(frontend_test PRIVATE pylite_frontend)

add_library(pylite_irgen STATIC
  irgen/abi.cpp
  irgen/irgen.cpp
  aot/compiler.cpp
)
target_include_directories(pylite_irgen PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  ${LLVM_INCLUDE_DIRS}
)
target_link_libraries(pylite_irgen PUBLIC pylite_frontend LLVM)

add_executable(pylitec tools/pylitec.cpp)
target_link_libraries(pylitec PRIVATE pylite_irgen)

add_executable(abi_test tests/abi_test.cpp)
target_link_libraries(abi_test PRIVATE pylite_runtime)

add_executable(runtime_test tests/runtime_test.cpp)
target_link_libraries(runtime_test PRIVATE pylite_runtime)

add_executable(gc_test tests/gc_test.cpp)
target_link_libraries(gc_test PRIVATE pylite_runtime)

add_executable(jit_spike tests/jit_spike.cpp)
target_include_directories(jit_spike PRIVATE ${LLVM_INCLUDE_DIRS})
target_link_libraries(jit_spike PRIVATE LLVM pylite_runtime)

add_executable(target_probe tests/target_probe.cpp)
target_include_directories(target_probe PRIVATE ${LLVM_INCLUDE_DIRS})
target_link_libraries(target_probe PRIVATE LLVM)

enable_testing()
add_test(NAME abi_test COMMAND abi_test)
add_test(NAME jit_spike COMMAND jit_spike)
add_test(NAME frontend_test COMMAND frontend_test)
add_test(NAME runtime_test COMMAND runtime_test)
add_test(NAME gc_test COMMAND gc_test)

foreach(example range_print fib containers)
  add_test(NAME parse_${example}
           COMMAND ast_dump ${CMAKE_CURRENT_SOURCE_DIR}/examples/${example}.pys)
endforeach()

set(E2E_CASES
  arith_ops containers dicts fib for_step gc_stress if_else
  inplace_add input_types lists logic long_loop loop_sum
  multiassign range_print slices strings unpack_call while_ctl
)

set(LINKTEST_OBJ ${CMAKE_CURRENT_BINARY_DIR}/linktest/arith.o)
add_custom_command(
  OUTPUT ${LINKTEST_OBJ}
  COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_CURRENT_BINARY_DIR}/linktest
  COMMAND pylitec ${CMAKE_CURRENT_SOURCE_DIR}/tests/linktest/arith.pys -o ${LINKTEST_OBJ}
  DEPENDS pylitec ${CMAKE_CURRENT_SOURCE_DIR}/tests/linktest/arith.pys
  COMMENT "pylitec linktest/arith.pys")

add_executable(linktest tests/linktest/main.cpp ${LINKTEST_OBJ})
target_link_libraries(linktest PRIVATE pylite_runtime)

add_test(NAME linktest
         COMMAND ${CMAKE_COMMAND}
                 -DEXE=$<TARGET_FILE:linktest>
                 -DEXPECTED=${CMAKE_CURRENT_SOURCE_DIR}/tests/linktest/arith.expected
                 -P ${CMAKE_CURRENT_SOURCE_DIR}/tests/compare_output.cmake)

set(MIXED_OBJ ${CMAKE_CURRENT_BINARY_DIR}/mixed/scores.o)
add_custom_command(
  OUTPUT ${MIXED_OBJ}
  COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_CURRENT_BINARY_DIR}/mixed
  COMMAND pylitec ${CMAKE_CURRENT_SOURCE_DIR}/examples/mixed/scores.pys -o ${MIXED_OBJ}
  DEPENDS pylitec ${CMAKE_CURRENT_SOURCE_DIR}/examples/mixed/scores.pys
  COMMENT "pylitec mixed/scores.pys")

add_executable(mixed_demo examples/mixed/main.cpp ${MIXED_OBJ})
target_link_libraries(mixed_demo PRIVATE pylite_runtime)

add_test(NAME mixed_demo COMMAND mixed_demo)

foreach(t ${E2E_CASES})
  set(pys ${CMAKE_CURRENT_SOURCE_DIR}/tests/e2e/${t}.pys)
  set(exp ${CMAKE_CURRENT_SOURCE_DIR}/tests/e2e/${t}.expected)
  set(inp ${CMAKE_CURRENT_SOURCE_DIR}/tests/e2e/${t}.in)
  set(obj ${CMAKE_CURRENT_BINARY_DIR}/e2e/${t}.o)
  set(drv ${CMAKE_CURRENT_BINARY_DIR}/e2e/${t}_main.cpp)

  file(WRITE ${drv}
    "extern \"C\" void pylite_${t}_main();\n"
    "int main() { pylite_${t}_main(); return 0; }\n")

  add_custom_command(
    OUTPUT ${obj}
    COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_CURRENT_BINARY_DIR}/e2e
    COMMAND pylitec ${pys} -o ${obj}
    DEPENDS pylitec ${pys}
    COMMENT "pylitec ${t}.pys")

  add_executable(e2e_${t} ${drv} ${obj})
  target_link_libraries(e2e_${t} PRIVATE pylite_runtime)

  set(inputArg)
  if(EXISTS ${inp})
    list(APPEND inputArg -DINPUT=${inp})
  endif()

  add_test(NAME e2e_${t}
           COMMAND ${CMAKE_COMMAND}
                   -DEXE=$<TARGET_FILE:e2e_${t}>
                   -DEXPECTED=${exp}
                   ${inputArg}
                   -P ${CMAKE_CURRENT_SOURCE_DIR}/tests/compare_output.cmake)

  add_test(NAME e2e_${t}_gc
           COMMAND ${CMAKE_COMMAND}
                   -DEXE=$<TARGET_FILE:e2e_${t}>
                   -DEXPECTED=${exp}
                   ${inputArg}
                   -P ${CMAKE_CURRENT_SOURCE_DIR}/tests/compare_output.cmake)
  set_tests_properties(e2e_${t}_gc PROPERTIES
                       ENVIRONMENT "PYLITE_GC_STRESS=1")
endforeach()
