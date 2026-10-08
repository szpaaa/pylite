// PyValue —— PyLite 的通用运行时值表示。
//
// 这是整个项目的 ABI 基石:运行时库(由 GCC 编译)、IRGen 生成的代码
// (由 LLVM 降低)、以及用户手写的 C++ 三方都按这份布局交换数据。
// 改动这里的任何字段都会破坏已编译的目标文件,务必谨慎。
//
// 布局必须与 LLVM 侧的 `StructType::create(ctx, {i32, i64}, "PyValue")` 逐字节一致:
//   offset 0: tag (i32)
//   offset 4: 填充
//   offset 8: payload (i64)
// 总计 16 字节,对齐 8。
#pragma once

#include <cstdint>
#include <cstddef>

enum PyTag : int32_t {
  PY_NULL   = 0,
  PY_BOOL   = 1,
  PY_INT    = 2,
  PY_FLOAT  = 3,
  PY_STR    = 4,
  PY_LIST   = 5,
  PY_DICT   = 6,
  PY_TUPLE  = 7,
};

struct PyStr;
struct PyList;
struct PyDict;
struct PyTuple;

struct PyValue {
  int32_t tag;
  union {
    int64_t i;
    double  f;
    void   *ptr;
  } as;
};

static_assert(sizeof(PyValue) == 16, "PyValue layout is part of the ABI");
static_assert(alignof(PyValue) == 8, "PyValue layout is part of the ABI");

inline PyValue py_none() {
  PyValue r;
  r.tag = PY_NULL;
  r.as.i = 0;
  return r;
}

inline PyValue py_int(int64_t v) {
  PyValue r;
  r.tag = PY_INT;
  r.as.i = v;
  return r;
}

inline PyValue py_float(double v) {
  PyValue r;
  r.tag = PY_FLOAT;
  r.as.f = v;
  return r;
}

inline PyValue py_bool(bool v) {
  PyValue r;
  r.tag = PY_BOOL;
  r.as.i = v ? 1 : 0;
  return r;
}

inline PyValue py_ptr(int32_t tag, void *p) {
  PyValue r;
  r.tag = tag;
  r.as.ptr = p;
  return r;
}

inline int64_t py_as_int(PyValue v)   { return v.as.i; }
inline double  py_as_float(PyValue v) { return v.as.f; }
inline bool    py_as_bool(PyValue v)  { return v.as.i != 0; }
inline void   *py_as_ptr(PyValue v)   { return v.as.ptr; }
