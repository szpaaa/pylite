// PyLite 跨平台兼容层
// 统一 Linux/macOS/Windows 的 API 差异，让运行时模块无需关心平台细节。
#pragma once

#include <cstdio>
#include <cstdlib>
#include <cstring>

#if defined(_WIN32) || defined(_WIN64)
  #define PYLITE_WINDOWS 1
  #define PYLITE_PLATFORM "Windows"
#elif defined(__APPLE__)
  #define PYLITE_APPLE 1
  #define PYLITE_PLATFORM "macOS"
#elif defined(__linux__)
  #define PYLITE_LINUX 1
  #define PYLITE_PLATFORM "Linux"
#else
  #error "Unsupported platform"
#endif

// 动态库加载
#ifdef PYLITE_WINDOWS
  #include <windows.h>
  #define PYLITE_DLOPEN(path)   (void*)LoadLibraryA(path)
  #define PYLITE_DLSYM(handle, name) (void*)GetProcAddress((HMODULE)(handle), name)
  #define PYLITE_DLCLOSE(handle) FreeLibrary((HMODULE)(handle))
  #define PYLITE_DLERROR()      ""
#else
  #include <dlfcn.h>
  #define PYLITE_DLOPEN(path)   dlopen(path, RTLD_LAZY)
  #define PYLITE_DLSYM(handle, name) dlsym(handle, name)
  #define PYLITE_DLCLOSE(handle) dlclose(handle)
  #define PYLITE_DLERROR()      dlerror()
#endif

// 进程管道
#ifdef PYLITE_WINDOWS
  #define PYLITE_POPEN(cmd, mode)  _popen(cmd, mode)
  #define PYLITE_PCLOSE(fp)        _pclose(fp)
#else
  #define PYLITE_POPEN(cmd, mode)  popen(cmd, mode)
  #define PYLITE_PCLOSE(fp)        pclose(fp)
#endif

// 路径分隔符
#ifdef PYLITE_WINDOWS
  #define PYLITE_PATH_SEP '\\'
  #define PYLITE_PATH_SEP_STR "\\"
#else
  #define PYLITE_PATH_SEP '/'
  #define PYLITE_PATH_SEP_STR "/"
#endif

// 临时文件路径
#ifdef PYLITE_WINDOWS
  #include <windows.h>
  inline const char* pylite_tmp_dir() {
    static char buf[MAX_PATH];
    if (buf[0] == '\0') { GetTempPathA(MAX_PATH, buf); size_t len = strlen(buf); if (len > 0 && buf[len - 1] == '\\') buf[len - 1] = '\0'; }
    return buf;
  }
#else
  inline const char* pylite_tmp_dir() { return "/tmp"; }
#endif

// 文件操作
#ifdef PYLITE_WINDOWS
  #define PYLITE_FOPEN_READ(path)  fopen(path, "rb")
  #define PYLITE_FOPEN_WRITE(path) fopen(path, "wb")
#else
  #define PYLITE_FOPEN_READ(path)  fopen(path, "r")
  #define PYLITE_FOPEN_WRITE(path) fopen(path, "w")
#endif

// 线程
#ifdef PYLITE_WINDOWS
  #include <windows.h>
  #include <process.h>
  #define PYLITE_THREAD_TYPE HANDLE
  #define PYLITE_THREAD_CREATE(thread, func, arg) ((*(thread) = (HANDLE)_beginthreadex(NULL, 0, (unsigned(__stdcall*)(void*))(func), arg, 0, NULL)) != NULL)
  #define PYLITE_THREAD_JOIN(thread) WaitForSingleObject(thread, INFINITE)
  #define PYLITE_THREAD_DETACH(thread) CloseHandle(thread)
#else
  #include <thread>
  #define PYLITE_THREAD_TYPE std::thread
  #define PYLITE_THREAD_CREATE(thread, func, arg) (*(thread) = std::thread(func, arg), true)
  #define PYLITE_THREAD_JOIN(thread) (thread)->join()
  #define PYLITE_THREAD_DETACH(thread) (thread)->detach()
#endif

// 网络 socket
#ifdef PYLITE_WINDOWS
  #include <winsock2.h>
  #include <ws2tcpip.h>
  #define PYLITE_SOCKET_INIT() do { WSADATA wsa; WSAStartup(MAKEWORD(2,2), &wsa); } while(0)
  #define PYLITE_SOCKET_CLEANUP() WSACleanup()
  #define PYLITE_CLOSE_SOCKET(s) closesocket(s)
  typedef SOCKET pylite_socket_t;
#else
  #include <sys/socket.h>
  #include <netinet/in.h>
  #include <arpa/inet.h>
  #include <unistd.h>
  #define PYLITE_SOCKET_INIT()
  #define PYLITE_SOCKET_CLEANUP()
  #define PYLITE_CLOSE_SOCKET(s) close(s)
  typedef int pylite_socket_t;
#endif

// 原子操作
#ifdef PYLITE_WINDOWS
  #include <windows.h>
  #define PYLITE_ATOMIC_BOOL volatile LONG
  #define PYLITE_ATOMIC_LOAD(p) (InterlockedCompareExchange(p, 0, 0) != 0)
  #define PYLITE_ATOMIC_STORE(p, v) InterlockedExchange(p, (v) ? 1 : 0)
#else
  #include <atomic>
  #define PYLITE_ATOMIC_BOOL std::atomic<bool>
  #define PYLITE_ATOMIC_LOAD(p) (p)->load()
  #define PYLITE_ATOMIC_STORE(p, v) (p)->store(v)
#endif

// 栈基址获取（GC 保守扫描）
#ifdef PYLITE_WINDOWS
  inline uintptr_t pylite_stack_base() {
    uintptr_t v;
    #ifdef _MSC_VER
      v = (uintptr_t)__readgsqword(0x08);
    #else
      asm volatile("movq %%gs:0x08, %0" : "=r"(v));
    #endif
    return v;
  }
#elif defined(PYLITE_LINUX)
  #include <pthread.h>
  inline uintptr_t pylite_stack_base() {
    pthread_attr_t attr; void *stackaddr = nullptr; size_t stacksize = 0;
    pthread_getattr_np(pthread_self(), &attr);
    pthread_attr_getstack(&attr, &stackaddr, &stacksize);
    pthread_attr_destroy(&attr);
    return reinterpret_cast<uintptr_t>(stackaddr) + stacksize;
  }
#elif defined(PYLITE_APPLE)
  #include <pthread.h>
  inline uintptr_t pylite_stack_base() {
    return reinterpret_cast<uintptr_t>(pthread_get_stackaddr_np(pthread_self()));
  }
#endif

// 可执行文件扩展名
#ifdef PYLITE_WINDOWS
  #define PYLITE_EXE_EXT ".exe"
#else
  #define PYLITE_EXE_EXT ""
#endif

// 共享库扩展名
#ifdef PYLITE_WINDOWS
  #define PYLITE_SO_EXT ".dll"
#elif defined(PYLITE_APPLE)
  #define PYLITE_SO_EXT ".dylib"
#else
  #define PYLITE_SO_EXT ".so"
#endif
