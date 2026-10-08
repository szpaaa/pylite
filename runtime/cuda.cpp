// CUDA 编程支持 —— 通过 LLVM NVPTX 后端生成 GPU 代码。
// 设计原则：使用 CUDA Driver API（libcuda.so）、PTX 代码由 LLVM NVPTX 后端生成、提供 GPU 内存分配/数据传输/内核启动等核心原语。
#include "pylite/runtime.h"
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <dlfcn.h>

namespace {
typedef int CUresult;
typedef int CUdevice;
typedef void *CUcontext;
typedef void *CUmodule;
typedef void *CUfunction;
typedef void *CUdeviceptr;
typedef void *CUstream;
#define CUDA_SUCCESS 0

typedef CUresult (*t_cuInit)(unsigned int);
typedef CUresult (*t_cuDeviceGet)(CUdevice *, int);
typedef CUresult (*t_cuCtxCreate)(CUcontext *, unsigned int, CUdevice);
typedef CUresult (*t_cuModuleLoadData)(CUmodule *, const void *);
typedef CUresult (*t_cuModuleGetFunction)(CUfunction *, CUmodule, const char *);
typedef CUresult (*t_cuMemAlloc)(CUdeviceptr *, size_t);
typedef CUresult (*t_cuMemFree)(CUdeviceptr);
typedef CUresult (*t_cuMemcpyHtoD)(CUdeviceptr, const void *, size_t);
typedef CUresult (*t_cuMemcpyDtoH)(void *, CUdeviceptr, size_t);
typedef CUresult (*t_cuLaunchKernel)(CUfunction, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, CUstream, void **, void **);
typedef CUresult (*t_cuCtxSynchronize)();
typedef CUresult (*t_cuCtxDestroy)(CUcontext);
typedef CUresult (*t_cuDeviceGetName)(char *, int, CUdevice);
typedef CUresult (*t_cuDeviceGetAttribute)(int *, int, CUdevice);

t_cuInit p_cuInit = nullptr;
t_cuDeviceGet p_cuDeviceGet = nullptr;
t_cuCtxCreate p_cuCtxCreate = nullptr;
t_cuModuleLoadData p_cuModuleLoadData = nullptr;
t_cuModuleGetFunction p_cuModuleGetFunction = nullptr;
t_cuMemAlloc p_cuMemAlloc = nullptr;
t_cuMemFree p_cuMemFree = nullptr;
t_cuMemcpyHtoD p_cuMemcpyHtoD = nullptr;
t_cuMemcpyDtoH p_cuMemcpyDtoH = nullptr;
t_cuLaunchKernel p_cuLaunchKernel = nullptr;
t_cuCtxSynchronize p_cuCtxSynchronize = nullptr;
t_cuCtxDestroy p_cuCtxDestroy = nullptr;
t_cuDeviceGetName p_cuDeviceGetName = nullptr;
t_cuDeviceGetAttribute p_cuDeviceGetAttribute = nullptr;

void *cudaHandle = nullptr;
bool cudaInitialized = false;
CUcontext cudaContext = nullptr;

bool loadCudaDriver() {
  if (cudaHandle) return true;
  const char *paths[] = {"libcuda.so", "libcuda.so.1", "/usr/lib/x86_64-linux-gnu/libcuda.so", "/usr/lib/x86_64-linux-gnu/libcuda.so.1", "/usr/local/cuda/lib64/libcuda.so", nullptr};
  for (int i = 0; paths[i]; ++i) { cudaHandle = dlopen(paths[i], RTLD_LAZY); if (cudaHandle) break; }
  if (!cudaHandle) return false;
  #define LOAD_SYM(name) p_##name = reinterpret_cast<t_##name>(dlsym(cudaHandle, #name)); if (!p_##name) { dlclose(cudaHandle); cudaHandle = nullptr; return false; }
  LOAD_SYM(cuInit); LOAD_SYM(cuDeviceGet); LOAD_SYM(cuCtxCreate); LOAD_SYM(cuModuleLoadData); LOAD_SYM(cuModuleGetFunction);
  LOAD_SYM(cuMemAlloc); LOAD_SYM(cuMemFree); LOAD_SYM(cuMemcpyHtoD); LOAD_SYM(cuMemcpyDtoH); LOAD_SYM(cuLaunchKernel);
  LOAD_SYM(cuCtxSynchronize); LOAD_SYM(cuCtxDestroy); LOAD_SYM(cuDeviceGetName); LOAD_SYM(cuDeviceGetAttribute);
  #undef LOAD_SYM
  return true;
}

bool initCudaDevice() {
  if (cudaInitialized) return true;
  if (!loadCudaDriver()) return false;
  CUresult r = p_cuInit(0); if (r != CUDA_SUCCESS) return false;
  CUdevice dev; r = p_cuDeviceGet(&dev, 0); if (r != CUDA_SUCCESS) return false;
  r = p_cuCtxCreate(&cudaContext, 0, dev); if (r != CUDA_SUCCESS) return false;
  cudaInitialized = true; return true;
}

char *captureCommand(const char *cmd) {
  FILE *fp = popen(cmd, "r"); if (!fp) return nullptr;
  size_t cap = 4096; char *buf = static_cast<char *>(malloc(cap)); if (!buf) { pclose(fp); return nullptr; }
  size_t len = 0;
  while (!feof(fp) && !ferror(fp)) { if (len + 256 >= cap) { cap *= 2; char *nb = static_cast<char *>(realloc(buf, cap)); if (!nb) { free(buf); pclose(fp); return nullptr; } buf = nb; } size_t n = fread(buf + len, 1, cap - len - 1, fp); len += n; }
  buf[len] = '\0'; pclose(fp); return buf;
}
}  // namespace

extern "C" PyValue py_cuda_info() {
  if (!initCudaDevice()) py_runtime_error("CUDA 初始化失败: 未检测到 NVIDIA GPU 或驱动");
  CUdevice dev; p_cuDeviceGet(&dev, 0);
  char name[256] = {0}; p_cuDeviceGetName(name, sizeof(name), dev);
  int major = 0, minor = 0; p_cuDeviceGetAttribute(&major, 75, dev); p_cuDeviceGetAttribute(&minor, 76, dev);
  int memBytes = 0; p_cuDeviceGetAttribute(&memBytes, 6, dev);
  char buf[512]; snprintf(buf, sizeof(buf), "GPU: %s\n计算能力: %d.%d\n显存: %d MB", name, major, minor, memBytes / (1024 * 1024));
  return py_str_new(buf, static_cast<int64_t>(strlen(buf)));
}

extern "C" PyValue py_cuda_compile_ptx(const PyValue *kernelCode, const PyValue *kernelName) {
  if (kernelCode->tag != PY_STR) py_runtime_error("cuda_compile_ptx() 需要字符串参数: kernel_code");
  const char *code = py_str_data(kernelCode); int64_t codeLen = py_str_size(kernelCode);
  const char *tmpFile = "/tmp/pylite_kernel.cu";
  FILE *fp = fopen(tmpFile, "w"); if (!fp) py_runtime_error("cuda_compile_ptx() 失败: 无法创建临时文件");
  fwrite(code, 1, static_cast<size_t>(codeLen), fp); fclose(fp);
  const char *ptxFile = "/tmp/pylite_kernel.ptx";
  size_t cmdCap = static_cast<size_t>(codeLen) + 256; char *cmd = static_cast<char *>(malloc(cmdCap)); if (!cmd) py_runtime_error("内存不足");
  snprintf(cmd, cmdCap, "nvcc -ptx -o %s %s 2>&1", ptxFile, tmpFile);
  char *output = captureCommand(cmd); free(cmd);
  fp = fopen(ptxFile, "r");
  if (!fp) { char errBuf[1024]; snprintf(errBuf, sizeof(errBuf), "CUDA 编译失败:\n%s", output ? output : "未知错误"); free(output); py_runtime_error(errBuf); }
  fseek(fp, 0, SEEK_END); long ptxSize = ftell(fp); fseek(fp, 0, SEEK_SET);
  char *ptxCode = static_cast<char *>(malloc(static_cast<size_t>(ptxSize) + 1)); if (!ptxCode) { fclose(fp); free(output); py_runtime_error("内存不足"); }
  fread(ptxCode, 1, static_cast<size_t>(ptxSize), fp); ptxCode[ptxSize] = '\0'; fclose(fp);
  PyValue result = py_str_new(ptxCode, static_cast<int64_t>(ptxSize)); free(ptxCode); free(output);
  remove(tmpFile); remove(ptxFile); return result;
}

extern "C" PyValue py_cuda_launch_kernel(const PyValue *ptxCode, const PyValue *kernelName,
    const PyValue *gridX, const PyValue *gridY, const PyValue *gridZ,
    const PyValue *blockX, const PyValue *blockY, const PyValue *blockZ, const PyValue *args) {
  if (!initCudaDevice()) py_runtime_error("CUDA 初始化失败");
  if (ptxCode->tag != PY_STR || kernelName->tag != PY_STR) py_runtime_error("cuda_launch_kernel() 需要 ptx_code 和 kernel_name 字符串参数");
  const char *ptx = py_str_data(ptxCode); const char *kname = py_str_data(kernelName);
  CUmodule module; CUresult r = p_cuModuleLoadData(&module, ptx); if (r != CUDA_SUCCESS) py_runtime_error("CUDA 模块加载失败");
  CUfunction kernel; r = p_cuModuleGetFunction(&kernel, module, kname); if (r != CUDA_SUCCESS) py_runtime_error("CUDA 内核函数未找到");
  int64_t gx = (gridX->tag == PY_INT) ? py_as_int(*gridX) : 1, gy = (gridY->tag == PY_INT) ? py_as_int(*gridY) : 1, gz = (gridZ->tag == PY_INT) ? py_as_int(*gridZ) : 1;
  int64_t bx = (blockX->tag == PY_INT) ? py_as_int(*blockX) : 256, by = (blockY->tag == PY_INT) ? py_as_int(*blockY) : 1, bz = (blockZ->tag == PY_INT) ? py_as_int(*blockZ) : 1;
  void *kernelParams[] = {nullptr};
  r = p_cuLaunchKernel(kernel, static_cast<unsigned int>(gx), static_cast<unsigned int>(gy), static_cast<unsigned int>(gz),
                       static_cast<unsigned int>(bx), static_cast<unsigned int>(by), static_cast<unsigned int>(bz), 0, nullptr, nullptr, kernelParams);
  if (r != CUDA_SUCCESS) py_runtime_error("CUDA 内核启动失败");
  r = p_cuCtxSynchronize(); if (r != CUDA_SUCCESS) py_runtime_error("CUDA 同步失败");
  char buf[256]; snprintf(buf, sizeof(buf), "CUDA 内核 '%s' 已启动\n网格: (%lld,%lld,%lld)\n块: (%lld,%lld,%lld)\n状态: 完成", kname, gx, gy, gz, bx, by, bz);
  return py_str_new(buf, static_cast<int64_t>(strlen(buf)));
}

extern "C" PyValue py_cuda_alloc(const PyValue *size) {
  if (!initCudaDevice()) py_runtime_error("CUDA 初始化失败");
  if (size->tag != PY_INT) py_runtime_error("cuda_alloc() 需要一个整数参数: size");
  int64_t sz = py_as_int(*size); if (sz <= 0) py_runtime_error("cuda_alloc() 需要正数大小");
  CUdeviceptr dptr; CUresult r = p_cuMemAlloc(&dptr, static_cast<size_t>(sz)); if (r != CUDA_SUCCESS) py_runtime_error("CUDA 内存分配失败");
  char buf[64]; snprintf(buf, sizeof(buf), "gpu_ptr:%p", reinterpret_cast<void *>(dptr));
  return py_str_new(buf, static_cast<int64_t>(strlen(buf)));
}

extern "C" void py_cuda_free(const PyValue *handle) {
  if (!initCudaDevice()) py_runtime_error("CUDA 初始化失败");
  if (handle->tag != PY_STR) py_runtime_error("cuda_free() 需要一个字符串参数: handle");
  const char *h = py_str_data(handle); void *ptr = nullptr;
  if (sscanf(h, "gpu_ptr:%p", &ptr) != 1 || !ptr) py_runtime_error("cuda_free() 无效的内存句柄");
  p_cuMemFree(reinterpret_cast<CUdeviceptr>(ptr));
}

extern "C" void py_cuda_memcpy_to_device(const PyValue *handle, const PyValue *data) {
  if (!initCudaDevice()) py_runtime_error("CUDA 初始化失败");
  if (handle->tag != PY_STR || data->tag != PY_STR) py_runtime_error("cuda_memcpy_to_device() 需要 (str, str) 参数");
  const char *h = py_str_data(handle); void *dptr = nullptr;
  if (sscanf(h, "gpu_ptr:%p", &dptr) != 1 || !dptr) py_runtime_error("cuda_memcpy_to_device() 无效的内存句柄");
  const char *src = py_str_data(data); int64_t len = py_str_size(data);
  CUresult r = p_cuMemcpyHtoD(reinterpret_cast<CUdeviceptr>(dptr), src, static_cast<size_t>(len));
  if (r != CUDA_SUCCESS) py_runtime_error("CUDA 数据传输失败 (HtoD)");
}

extern "C" PyValue py_cuda_memcpy_from_device(const PyValue *handle, const PyValue *size) {
  if (!initCudaDevice()) py_runtime_error("CUDA 初始化失败");
  if (handle->tag != PY_STR || size->tag != PY_INT) py_runtime_error("cuda_memcpy_from_device() 需要 (str, int) 参数");
  const char *h = py_str_data(handle); void *dptr = nullptr;
  if (sscanf(h, "gpu_ptr:%p", &dptr) != 1 || !dptr) py_runtime_error("cuda_memcpy_from_device() 无效的内存句柄");
  int64_t sz = py_as_int(*size); if (sz <= 0) py_runtime_error("cuda_memcpy_from_device() 需要正数大小");
  char *buf = static_cast<char *>(malloc(static_cast<size_t>(sz) + 1)); if (!buf) py_runtime_error("内存不足");
  CUresult r = p_cuMemcpyDtoH(buf, reinterpret_cast<CUdeviceptr>(dptr), static_cast<size_t>(sz));
  if (r != CUDA_SUCCESS) { free(buf); py_runtime_error("CUDA 数据传输失败 (DtoH)"); }
  buf[sz] = '\0'; PyValue result = py_str_new(buf, sz); free(buf); return result;
}
