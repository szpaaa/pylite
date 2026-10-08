// PyLite 运行时库的 C ABI。
//
// IRGen 会对这里每个函数生成 call;用户手写的 C++ 也可以直接调用它们。
// 所有函数都从 libpylite_runtime.so 导出。
//
// ⚠️ 跨边界约定 —— 改动前务必先读 docs/llvm-notes.md 第 2 节
//
// 下面这些函数在 C++ 源码里是按值的 `PyValue f(PyValue, PyValue)`,
// 但 IRGen 在 IR 里必须把它们声明成:
//
//     void @f(ptr sret(%PyValue), ptr, ptr)
//
// 即参数是"指向 PyValue 的指针"、返回值写进 sret 指针。
// 两者在机器层面一致,因为 Win64 ABI 规定 16 字节聚合体按引用传递、
// 返回值走隐藏 sret 寄存器(RCX = sret,RDX/R8/... = 各参数)。
//
// 若在 IR 里按值声明(`%PyValue @f(%PyValue, %PyValue)`),LLVM 会选另一套
// 降低方式,与 GCC 不匹配 —— 链接通过、lookup 成功,但一执行就段错误。
//
// 其它约定:
//   * 谓词一律返回 int32_t 而非 bool(i1 的 ABI 标注 GCC 与 Clang 不一致)
//   * 数组参数(如 py_print 的 PyValue*)是普通指针,不受上面限制
#pragma once

#include "pylite/value.h"

extern "C" {

// --- 错误 ---------------------------------------------------------------
[[noreturn]] void py_runtime_error(const char *msg);
const char *py_tag_name(int32_t tag);

// --- 算术 ---------------------------------------------------------------
PyValue py_add(PyValue a, PyValue b);
PyValue py_sub(PyValue a, PyValue b);
PyValue py_mul(PyValue a, PyValue b);
PyValue py_div(PyValue a, PyValue b);
PyValue py_floordiv(PyValue a, PyValue b);
PyValue py_mod(PyValue a, PyValue b);
PyValue py_pow(PyValue a, PyValue b);
PyValue py_neg(PyValue a);
PyValue py_iadd(PyValue *a, const PyValue *b);

// --- 比较 ---------------------------------------------------------------
PyValue py_eq(PyValue a, PyValue b);
PyValue py_ne(PyValue a, PyValue b);
PyValue py_lt(PyValue a, PyValue b);
PyValue py_le(PyValue a, PyValue b);
PyValue py_gt(PyValue a, PyValue b);
PyValue py_ge(PyValue a, PyValue b);
int32_t py_compare(const PyValue *a, const PyValue *b);

// --- 逻辑 ---------------------------------------------------------------
PyValue py_not(PyValue a);
int32_t py_truthy(PyValue v);

// --- 转换 ---------------------------------------------------------------
int64_t py_to_int(PyValue v);

// --- 输出 ---------------------------------------------------------------
void py_print(PyValue *vals, int64_t n);
PyValue py_repr(const PyValue *v);

// --- 输入 ---------------------------------------------------------------
PyValue py_read_int();
PyValue py_read_float();
PyValue py_read_bool();
PyValue py_read_str();

// --- 内存与垃圾回收 ------------------------------------------------------
typedef enum {
  PY_GC_SCAN_NONE = 0,
  PY_GC_SCAN_WORDS = 1,
  PY_GC_SCAN_VALUES = 2,
} PyGcScan;

typedef struct {
  int64_t alloc_bytes_total;
  int64_t freed_bytes_total;
  int64_t live_bytes;
  int64_t live_blocks;
  int64_t collections;
  int32_t dry_run;
  int32_t disabled;
} PyGcStats;

void *py_gc_alloc(size_t n, PyGcScan kind);
void *py_gc_alloc_values(size_t n, size_t firstCellOffset);
void py_gc_collect();
void py_gc_stats(PyGcStats *out);
void py_gc_init();
void py_gc_root_push(void **slot);
void py_gc_root_pop(size_t n);
void *py_alloc(size_t n);
void py_arena_reset();

// --- 元组 ---------------------------------------------------------------
PyValue py_tuple_new(PyValue *elems, int64_t n);
int64_t py_tuple_len(const PyValue *t);
PyValue py_tuple_get(const PyValue *t, int64_t i);
void py_unpack(const PyValue *t, PyValue *out, int64_t n);

// --- 字符串 -------------------------------------------------------------
PyValue py_str_new(const char *bytes, int64_t len);
int64_t py_str_size(const PyValue *s);
const char *py_str_data(const PyValue *s);
PyValue py_str_concat(const PyValue *a, const PyValue *b);
PyValue py_str_get(const PyValue *s, int64_t i);
PyValue py_str_repeat(const PyValue *s, int64_t times);
int32_t py_str_compare(const PyValue *a, const PyValue *b);
int32_t py_str_find(const PyValue *s, const PyValue *needle, int64_t from);
int32_t py_str_starts_with(const PyValue *s, const PyValue *prefix);
int32_t py_str_ends_with(const PyValue *s, const PyValue *suffix);

// --- 列表 ---------------------------------------------------------------
PyValue py_list_new(PyValue *elems, int64_t n);
int64_t py_list_len(const PyValue *v);
PyValue py_list_get(const PyValue *v, int64_t i);
void py_list_set(const PyValue *v, int64_t i, const PyValue *val);
PyValue py_list_concat(const PyValue *a, const PyValue *b);
PyValue py_list_repeat(const PyValue *a, int64_t times);
void py_list_append(const PyValue *v, const PyValue *item);
void py_list_extend(const PyValue *v, const PyValue *other);
void py_list_insert(const PyValue *v, int64_t i, const PyValue *item);
void py_list_remove(const PyValue *v, const PyValue *item);
PyValue py_list_pop(const PyValue *v, const PyValue *idxOrNone);
int64_t py_list_index(const PyValue *v, const PyValue *item);
int64_t py_list_count(const PyValue *v, const PyValue *item);
void py_list_reverse(const PyValue *v);
void py_list_clear(const PyValue *v);
void py_list_sort(const PyValue *v);

// --- 字典 ---------------------------------------------------------------
PyValue py_dict_new(PyValue *keys, PyValue *vals, int64_t n);
int64_t py_dict_len(const PyValue *d);
PyValue py_dict_get(const PyValue *d, const PyValue *key);
PyValue py_dict_get_default(const PyValue *d, const PyValue *key, const PyValue *def);
int32_t py_dict_has(const PyValue *d, const PyValue *key);
void py_dict_set(const PyValue *d, const PyValue *key, const PyValue *val);
PyValue py_dict_key_at(const PyValue *d, int64_t i);
PyValue py_dict_val_at(const PyValue *d, int64_t i);
PyValue py_dict_pop(const PyValue *d, const PyValue *key);
void py_dict_clear(const PyValue *d);

// --- 下标 / 切片 / 长度 --------------------------------------------------
int64_t py_len(const PyValue *v);
PyValue py_index(const PyValue *obj, const PyValue *idx);
void py_setindex(const PyValue *obj, const PyValue *idx, const PyValue *val);
PyValue py_slice(const PyValue *obj, const PyValue *lo, const PyValue *hi, const PyValue *step);
PyValue py_iter_at(const PyValue *obj, int64_t i);

// --- 方法调用 -----------------------------------------------------------
PyValue py_call_method(const PyValue *obj, const char *name, int64_t nameLen,
                       PyValue *args, int64_t nargs);

// --- Docker 容器调用 -----------------------------------------------------
PyValue py_docker_run(const PyValue *image, const PyValue *cmd);
PyValue py_docker_ps();
void py_docker_stop(const PyValue *containerId);
PyValue py_docker_logs(const PyValue *containerId);
PyValue py_docker_pull(const PyValue *image);
PyValue py_docker_generate_dockerfile(const PyValue *baseImage, const PyValue *projectName,
                                       const PyValue *setupCommands, const PyValue *entrypoint,
                                       const PyValue *exposePort);
PyValue py_docker_build(const PyValue *dockerfile, const PyValue *imageName,
                         const PyValue *tag, const PyValue *contextDir);
PyValue py_docker_push(const PyValue *imageName, const PyValue *tag, const PyValue *registry);
PyValue py_docker_project_package(const PyValue *projectDir, const PyValue *imageName,
                                   const PyValue *tag, const PyValue *baseImage,
                                   const PyValue *setupCommands, const PyValue *entrypoint,
                                   const PyValue *exposePort);

// --- LLM 神经网络接口 -----------------------------------------------------
PyValue py_llm_chat(const PyValue *prompt, const PyValue *systemPrompt,
                    const PyValue *model, const PyValue *apiKey, const PyValue *endpoint);
PyValue py_llm_create_network(const PyValue *name, const PyValue *layers);
PyValue py_llm_add_layer(const PyValue *network, const PyValue *layer);
PyValue py_llm_remove_layer(const PyValue *network, const PyValue *index);
PyValue py_llm_network_summary(const PyValue *network);
PyValue py_llm_create_dataset(const PyValue *examples, const PyValue *systemPrompt);
PyValue py_llm_upload_file(const PyValue *data, const PyValue *filename,
                           const PyValue *apiKey, const PyValue *endpoint);
PyValue py_llm_create_finetune(const PyValue *fileId, const PyValue *model,
                                const PyValue *suffix, const PyValue *apiKey,
                                const PyValue *endpoint);
PyValue py_llm_finetune_status(const PyValue *jobId, const PyValue *apiKey,
                                const PyValue *endpoint);
PyValue py_llm_list_finetunes(const PyValue *apiKey, const PyValue *endpoint);

// --- CUDA 编程支持 --------------------------------------------------------
PyValue py_cuda_info();
PyValue py_cuda_compile_ptx(const PyValue *kernelCode, const PyValue *kernelName);
PyValue py_cuda_launch_kernel(const PyValue *ptxCode, const PyValue *kernelName,
                               const PyValue *gridX, const PyValue *gridY,
                               const PyValue *gridZ, const PyValue *blockX,
                               const PyValue *blockY, const PyValue *blockZ,
                               const PyValue *args);
PyValue py_cuda_alloc(const PyValue *size);
void py_cuda_free(const PyValue *handle);
void py_cuda_memcpy_to_device(const PyValue *handle, const PyValue *data);
PyValue py_cuda_memcpy_from_device(const PyValue *handle, const PyValue *size);

// --- 算子融合优化 ---------------------------------------------------------
PyValue py_fusion_create_graph(const PyValue *name);
PyValue py_fusion_add_op(const PyValue *graph, const PyValue *opType,
                          const PyValue *opName, const PyValue *params);
PyValue py_fusion_apply_rules(const PyValue *graph);
PyValue py_fusion_generate_kernel(const PyValue *fusedOp, const PyValue *config);
PyValue py_fusion_autotune(const PyValue *opType, const PyValue *M,
                            const PyValue *N, const PyValue *K);
PyValue py_fusion_compile_and_run(const PyValue *kernelCode, const PyValue *kernelName,
                                   const PyValue *M, const PyValue *N, const PyValue *K);
PyValue py_fusion_benchmark(const PyValue *kernelCode, const PyValue *kernelName,
                             const PyValue *M, const PyValue *N,
                             const PyValue *K, const PyValue *iterations);

// --- LLM 推理运行时 -------------------------------------------------------
PyValue py_inference_load_model(const PyValue *modelPath, const PyValue *configPath);
PyValue py_inference_generate(const PyValue *prompt, const PyValue *maxTokens,
                               const PyValue *temperature, const PyValue *topP,
                               const PyValue *topK);
PyValue py_inference_kv_cache_stats();
PyValue py_inference_flash_attention_kernel();
PyValue py_inference_multi_gpu_info();
PyValue py_inference_profile();
void py_inference_unload();

// --- Tokenizer 分词器 -----------------------------------------------------
PyValue py_tokenizer_load(const PyValue *path);
PyValue py_tokenizer_encode(const PyValue *text);
PyValue py_tokenizer_decode(const PyValue *tokenIds);
PyValue py_tokenizer_info();

// --- REST API 服务 --------------------------------------------------------
PyValue py_server_start(const PyValue *port);
void py_server_stop();
PyValue py_server_status();

// --- 模型量化工具与多架构支持 ---------------------------------------------
PyValue py_quantize_fp16_to_awq(const PyValue *weightsJson, const PyValue *bits,
                                 const PyValue *groupSize);
PyValue py_quantize_arch_template(const PyValue *archName);
PyValue py_quantize_verify_precision(const PyValue *fp16Values, const PyValue *awqValues);
PyValue py_quantize_model_info(const PyValue *modelPath);

}  // extern "C"
