// LLM 神经网络接口 —— 通过 curl 调用 OpenAI 兼容 API。
// 设计原则：零外部依赖、支持 OpenAI 兼容 API、提供网络结构描述和修改能力。
#include "pylite/runtime.h"
#include <cstdio>
#include <cstring>
#include <cstdlib>

namespace {
char *captureCommand(const char *cmd) {
  FILE *fp = popen(cmd, "r");
  if (!fp) return nullptr;
  size_t cap = 4096;
  char *buf = static_cast<char *>(malloc(cap));
  if (!buf) { pclose(fp); return nullptr; }
  size_t len = 0;
  while (!feof(fp) && !ferror(fp)) {
    if (len + 256 >= cap) { cap *= 2; char *nb = static_cast<char *>(realloc(buf, cap)); if (!nb) { free(buf); pclose(fp); return nullptr; } buf = nb; }
    size_t n = fread(buf + len, 1, cap - len - 1, fp); len += n;
  }
  buf[len] = '\0'; pclose(fp); return buf;
}

char *jsonEscape(const char *s, int64_t len) {
  size_t cap = static_cast<size_t>(len) * 2 + 4;
  char *out = static_cast<char *>(malloc(cap));
  if (!out) return nullptr;
  size_t pos = 0;
  for (int64_t i = 0; i < len; ++i) {
    if (pos + 4 >= cap) { cap *= 2; char *nb = static_cast<char *>(realloc(out, cap)); if (!nb) { free(out); return nullptr; } out = nb; }
    switch (s[i]) {
      case '"': out[pos++] = '\\'; out[pos++] = '"'; break;
      case '\\': out[pos++] = '\\'; out[pos++] = '\\'; break;
      case '\n': out[pos++] = '\\'; out[pos++] = 'n'; break;
      case '\r': out[pos++] = '\\'; out[pos++] = 'r'; break;
      case '\t': out[pos++] = '\\'; out[pos++] = 't'; break;
      default: out[pos++] = s[i]; break;
    }
  }
  out[pos] = '\0'; return out;
}

char *extractContent(const char *json) {
  const char *key = "\"content\":\"";
  const char *start = strstr(json, key);
  if (!start) { key = "\"content\": \""; start = strstr(json, key); }
  if (!start) return nullptr;
  start += strlen(key);
  size_t cap = 256;
  char *out = static_cast<char *>(malloc(cap));
  if (!out) return nullptr;
  size_t pos = 0;
  while (*start && *start != '"') {
    if (pos + 2 >= cap) { cap *= 2; char *nb = static_cast<char *>(realloc(out, cap)); if (!nb) { free(out); return nullptr; } out = nb; }
    if (*start == '\\' && *(start + 1)) {
      start++;
      switch (*start) { case 'n': out[pos++] = '\n'; break; case 'r': out[pos++] = '\r'; break; case 't': out[pos++] = '\t'; break; case '"': out[pos++] = '"'; break; case '\\': out[pos++] = '\\'; break; default: out[pos++] = *start; break; }
    } else { out[pos++] = *start; }
    start++;
  }
  out[pos] = '\0'; return out;
}
}  // namespace

extern "C" PyValue py_llm_chat(const PyValue *prompt, const PyValue *systemPrompt,
    const PyValue *model, const PyValue *apiKey, const PyValue *endpoint) {
  if (prompt->tag != PY_STR || model->tag != PY_STR || apiKey->tag != PY_STR || endpoint->tag != PY_STR)
    py_runtime_error("llm_chat() 需要字符串参数: prompt, model, api_key, endpoint");

  const char *p = py_str_data(prompt);
  const char *sp = (systemPrompt->tag == PY_STR) ? py_str_data(systemPrompt) : "";
  const char *m = py_str_data(model);
  const char *key = py_str_data(apiKey);
  const char *ep = py_str_data(endpoint);
  int64_t pLen = py_str_size(prompt);
  int64_t spLen = (systemPrompt->tag == PY_STR) ? py_str_size(systemPrompt) : 0;
  int64_t mLen = py_str_size(model);
  int64_t keyLen = py_str_size(apiKey);
  int64_t epLen = py_str_size(endpoint);

  char *escPrompt = jsonEscape(p, pLen);
  char *escSystem = spLen > 0 ? jsonEscape(sp, spLen) : nullptr;
  if (!escPrompt) py_runtime_error("内存不足");

  size_t jsonCap = static_cast<size_t>(pLen + spLen + mLen + keyLen + epLen) + 1024;
  char *jsonBody = static_cast<char *>(malloc(jsonCap));
  if (!jsonBody) { free(escPrompt); free(escSystem); py_runtime_error("内存不足"); }

  int n = snprintf(jsonBody, jsonCap, "{\"model\":\"%.*s\",\"messages\":[", static_cast<int>(mLen), m);
  if (escSystem && spLen > 0)
    n += snprintf(jsonBody + n, jsonCap - static_cast<size_t>(n), "{\"role\":\"system\",\"content\":\"%s\"},", escSystem);
  n += snprintf(jsonBody + n, jsonCap - static_cast<size_t>(n), "{\"role\":\"user\",\"content\":\"%s\"}],\"temperature\":0.7,\"max_tokens\":2048}", escPrompt);
  free(escPrompt); free(escSystem);

  size_t cmdCap = jsonCap + keyLen + epLen + 512;
  char *cmd = static_cast<char *>(malloc(cmdCap));
  if (!cmd) { free(jsonBody); py_runtime_error("内存不足"); }
  snprintf(cmd, cmdCap, "curl -s -X POST '%.*s' -H 'Content-Type: application/json' -H 'Authorization: Bearer %.*s' --data-raw '%s' 2>&1",
           static_cast<int>(epLen), ep, static_cast<int>(keyLen), key, jsonBody);
  free(jsonBody);

  char *response = captureCommand(cmd);
  free(cmd);
  if (!response || strlen(response) == 0) { free(response); py_runtime_error("LLM API 调用失败: 无响应"); }
  if (strstr(response, "\"error\"")) {
    PyValue err = py_str_new(response, static_cast<int64_t>(strlen(response)));
    free(response);
    char errBuf[1024]; snprintf(errBuf, sizeof(errBuf), "LLM API 返回错误: %s", py_str_data(&err));
    py_runtime_error(errBuf);
  }

  char *content = extractContent(response);
  if (!content) {
    int64_t respLen = static_cast<int64_t>(strlen(response));
    if (respLen > 2000) respLen = 2000;
    PyValue result = py_str_new(response, respLen);
    free(response); return result;
  }
  PyValue result = py_str_new(content, static_cast<int64_t>(strlen(content)));
  free(response); free(content);
  return result;
}

extern "C" PyValue py_llm_create_network(const PyValue *name, const PyValue *layers) {
  if (name->tag != PY_STR || layers->tag != PY_STR)
    py_runtime_error("llm_create_network() 需要两个字符串参数: name, layers");
  const char *n = py_str_data(name);
  const char *l = py_str_data(layers);
  int64_t nLen = py_str_size(name);
  int64_t lLen = py_str_size(layers);
  size_t cap = static_cast<size_t>(nLen + lLen) + 128;
  char *json = static_cast<char *>(malloc(cap));
  if (!json) py_runtime_error("内存不足");
  snprintf(json, cap, "{\"name\":\"%.*s\",\"layers\":%.*s}", static_cast<int>(nLen), n, static_cast<int>(lLen), l);
  PyValue result = py_str_new(json, static_cast<int64_t>(strlen(json)));
  free(json); return result;
}

extern "C" PyValue py_llm_add_layer(const PyValue *network, const PyValue *layer) {
  if (network->tag != PY_STR || layer->tag != PY_STR)
    py_runtime_error("llm_add_layer() 需要两个字符串参数: network, layer");
  const char *net = py_str_data(network);
  const char *lay = py_str_data(layer);
  int64_t netLen = py_str_size(network);
  int64_t layLen = py_str_size(layer);
  const char *layersStart = strstr(net, "\"layers\":[");
  if (!layersStart) py_runtime_error("llm_add_layer() 失败: 无效的网络描述格式");
  const char *arrStart = strchr(layersStart, '[');
  if (!arrStart) py_runtime_error("llm_add_layer() 失败: 无效的网络描述格式");
  const char *end = net + netLen - 1;
  while (end > arrStart && *end != ']') end--;
  if (end <= arrStart) py_runtime_error("llm_add_layer() 失败: 无效的网络描述格式");
  size_t prefixLen = static_cast<size_t>(end - net);
  size_t suffixLen = static_cast<size_t>(netLen - (end - net));
  bool isEmpty = (end == arrStart + 1);
  size_t cap = prefixLen + static_cast<size_t>(layLen) + suffixLen + 4;
  char *newJson = static_cast<char *>(malloc(cap));
  if (!newJson) py_runtime_error("内存不足");
  if (isEmpty)
    snprintf(newJson, cap, "%.*s%.*s%.*s", static_cast<int>(prefixLen), net, static_cast<int>(layLen), lay, static_cast<int>(suffixLen), end);
  else
    snprintf(newJson, cap, "%.*s,%.*s%.*s", static_cast<int>(prefixLen), net, static_cast<int>(layLen), lay, static_cast<int>(suffixLen), end);
  PyValue result = py_str_new(newJson, static_cast<int64_t>(strlen(newJson)));
  free(newJson); return result;
}

extern "C" PyValue py_llm_remove_layer(const PyValue *network, const PyValue *index) {
  py_runtime_error("llm_remove_layer() 暂未实现。建议使用 llm_create_network() 重新构建网络描述。");
}

extern "C" PyValue py_llm_network_summary(const PyValue *network) {
  if (network->tag != PY_STR) py_runtime_error("llm_network_summary() 需要一个字符串参数: network");
  const char *net = py_str_data(network);
  int64_t netLen = py_str_size(network);
  size_t cap = static_cast<size_t>(netLen) + 256;
  char *summary = static_cast<char *>(malloc(cap));
  if (!summary) py_runtime_error("内存不足");
  snprintf(summary, cap, "[PyLite 神经网络描述]\n%.*s\n\n提示: 使用 llm_add_layer() 添加层，使用 llm_chat() 调用推理 API。", static_cast<int>(netLen), net);
  PyValue result = py_str_new(summary, static_cast<int64_t>(strlen(summary)));
  free(summary); return result;
}

extern "C" PyValue py_llm_create_dataset(const PyValue *examples, const PyValue *systemPrompt) {
  if (examples->tag != PY_STR) py_runtime_error("llm_create_dataset() 需要字符串参数: examples (JSON 数组)");
  const char *ex = py_str_data(examples);
  int64_t exLen = py_str_size(examples);
  const char *sp = (systemPrompt->tag == PY_STR) ? py_str_data(systemPrompt) : "";
  int64_t spLen = (systemPrompt->tag == PY_STR) ? py_str_size(systemPrompt) : 0;
  size_t cap = static_cast<size_t>(exLen + spLen) * 3 + 4096;
  char *jsonl = static_cast<char *>(malloc(cap));
  if (!jsonl) py_runtime_error("内存不足");
  size_t pos = 0;
  const char *p = ex;
  const char *end = ex + exLen;
  while (p < end) {
    while (p < end && (*p == ' ' || *p == '\n' || *p == '\r' || *p == '\t' || *p == ',')) p++;
    if (p >= end || *p == ']') break;
    if (*p != '{') { p++; continue; }
    const char *objStart = p;
    int braceDepth = 0;
    while (p < end) { if (*p == '{') braceDepth++; else if (*p == '}') { braceDepth--; if (braceDepth == 0) { p++; break; } } p++; }
    const char *inputStart = nullptr, *inputEnd = nullptr, *outputStart = nullptr, *outputEnd = nullptr;
    const char *scan = objStart;
    while (scan < p) {
      const char *key = strstr(scan, "\"input\"");
      if (key && key < p) { const char *val = strchr(key + 7, '"'); if (val) { val = strchr(val + 1, '"'); if (val) { inputStart = val + 1; val = strchr(inputStart, '"'); if (val) inputEnd = val; } } }
      key = strstr(scan, "\"output\"");
      if (key && key < p) { const char *val = strchr(key + 8, '"'); if (val) { val = strchr(val + 1, '"'); if (val) { outputStart = val + 1; val = strchr(outputStart, '"'); if (val) outputEnd = val; } } }
      if (inputStart && outputStart) break;
      scan++;
    }
    if (!inputStart || !outputStart) continue;
    int64_t inLen = inputEnd - inputStart, outLen = outputEnd - outputStart;
    char *escInput = jsonEscape(inputStart, inLen), *escOutput = jsonEscape(outputStart, outLen);
    if (!escInput || !escOutput) { free(escInput); free(escOutput); continue; }
    int n = 0;
    if (spLen > 0) {
      char *escSystem = jsonEscape(sp, spLen);
      n = snprintf(jsonl + pos, cap - pos, "{\"messages\":[{\"role\":\"system\",\"content\":\"%s\"},{\"role\":\"user\",\"content\":\"%s\"},{\"role\":\"assistant\",\"content\":\"%s\"}]}\n", escSystem, escInput, escOutput);
      free(escSystem);
    } else {
      n = snprintf(jsonl + pos, cap - pos, "{\"messages\":[{\"role\":\"user\",\"content\":\"%s\"},{\"role\":\"assistant\",\"content\":\"%s\"}]}\n", escInput, escOutput);
    }
    free(escInput); free(escOutput);
    if (n > 0) pos += static_cast<size_t>(n);
  }
  jsonl[pos] = '\0';
  PyValue result = py_str_new(jsonl, static_cast<int64_t>(pos));
  free(jsonl); return result;
}

extern "C" PyValue py_llm_upload_file(const PyValue *data, const PyValue *filename,
    const PyValue *apiKey, const PyValue *endpoint) {
  if (data->tag != PY_STR || filename->tag != PY_STR || apiKey->tag != PY_STR || endpoint->tag != PY_STR)
    py_runtime_error("llm_upload_file() 需要四个字符串参数");
  const char *d = py_str_data(data), *fn = py_str_data(filename), *key = py_str_data(apiKey), *ep = py_str_data(endpoint);
  int64_t dLen = py_str_size(data), fnLen = py_str_size(filename), keyLen = py_str_size(apiKey), epLen = py_str_size(endpoint);
  char tmpPath[256];
  snprintf(tmpPath, sizeof(tmpPath), "/tmp/pylite_upload_%.*s", static_cast<int>(fnLen), fn);
  FILE *fp = fopen(tmpPath, "w");
  if (!fp) py_runtime_error("llm_upload_file() 失败: 无法创建临时文件");
  fwrite(d, 1, static_cast<size_t>(dLen), fp); fclose(fp);
  size_t cmdCap = static_cast<size_t>(epLen + keyLen + fnLen) + 512;
  char *cmd = static_cast<char *>(malloc(cmdCap));
  if (!cmd) py_runtime_error("内存不足");
  snprintf(cmd, cmdCap, "curl -s -X POST '%.*s' -H 'Authorization: Bearer %.*s' -F 'purpose=fine-tune' -F 'file=@%s' 2>&1",
           static_cast<int>(epLen), ep, static_cast<int>(keyLen), key, tmpPath);
  char *response = captureCommand(cmd); free(cmd); remove(tmpPath);
  if (!response || strlen(response) == 0) { free(response); py_runtime_error("文件上传失败: 无响应"); }
  const char *idKey = "\"id\":\"";
  const char *idStart = strstr(response, idKey);
  char *fileId = nullptr;
  if (idStart) { idStart += strlen(idKey); const char *idEnd = strchr(idStart, '"'); if (idEnd) { int64_t idLen = idEnd - idStart; fileId = static_cast<char *>(malloc(static_cast<size_t>(idLen) + 1)); if (fileId) { memcpy(fileId, idStart, static_cast<size_t>(idLen)); fileId[idLen] = '\0'; } } }
  if (!fileId) { PyValue result = py_str_new(response, static_cast<int64_t>(strlen(response))); free(response); return result; }
  PyValue result = py_str_new(fileId, static_cast<int64_t>(strlen(fileId)));
  free(response); free(fileId); return result;
}

extern "C" PyValue py_llm_create_finetune(const PyValue *fileId, const PyValue *model,
    const PyValue *suffix, const PyValue *apiKey, const PyValue *endpoint) {
  if (fileId->tag != PY_STR || model->tag != PY_STR || apiKey->tag != PY_STR || endpoint->tag != PY_STR)
    py_runtime_error("llm_create_finetune() 需要字符串参数");
  const char *fid = py_str_data(fileId), *m = py_str_data(model), *suf = (suffix->tag == PY_STR) ? py_str_data(suffix) : "pylite-model";
  const char *key = py_str_data(apiKey), *ep = py_str_data(endpoint);
  int64_t fidLen = py_str_size(fileId), mLen = py_str_size(model), sufLen = (suffix->tag == PY_STR) ? py_str_size(suffix) : 11;
  int64_t keyLen = py_str_size(apiKey), epLen = py_str_size(endpoint);
  size_t jsonCap = static_cast<size_t>(fidLen + mLen + sufLen) + 256;
  char *jsonBody = static_cast<char *>(malloc(jsonCap));
  if (!jsonBody) py_runtime_error("内存不足");
  snprintf(jsonBody, jsonCap, "{\"training_file\":\"%.*s\",\"model\":\"%.*s\",\"suffix\":\"%.*s\"}",
           static_cast<int>(fidLen), fid, static_cast<int>(mLen), m, static_cast<int>(sufLen), suf);
  size_t cmdCap = jsonCap + static_cast<size_t>(epLen + keyLen) + 256;
  char *cmd = static_cast<char *>(malloc(cmdCap));
  if (!cmd) { free(jsonBody); py_runtime_error("内存不足"); }
  snprintf(cmd, cmdCap, "curl -s -X POST '%.*s' -H 'Content-Type: application/json' -H 'Authorization: Bearer %.*s' --data-raw '%s' 2>&1",
           static_cast<int>(epLen), ep, static_cast<int>(keyLen), key, jsonBody);
  free(jsonBody);
  char *response = captureCommand(cmd); free(cmd);
  if (!response || strlen(response) == 0) { free(response); py_runtime_error("微调任务创建失败: 无响应"); }
  const char *idKey = "\"id\":\"";
  const char *idStart = strstr(response, idKey);
  char *jobId = nullptr;
  if (idStart) { idStart += strlen(idKey); const char *idEnd = strchr(idStart, '"'); if (idEnd) { int64_t idLen = idEnd - idStart; jobId = static_cast<char *>(malloc(static_cast<size_t>(idLen) + 1)); if (jobId) { memcpy(jobId, idStart, static_cast<size_t>(idLen)); jobId[idLen] = '\0'; } } }
  if (!jobId) { PyValue result = py_str_new(response, static_cast<int64_t>(strlen(response))); free(response); return result; }
  PyValue result = py_str_new(jobId, static_cast<int64_t>(strlen(jobId)));
  free(response); free(jobId); return result;
}

extern "C" PyValue py_llm_finetune_status(const PyValue *jobId, const PyValue *apiKey, const PyValue *endpoint) {
  if (jobId->tag != PY_STR || apiKey->tag != PY_STR || endpoint->tag != PY_STR)
    py_runtime_error("llm_finetune_status() 需要三个字符串参数");
  const char *jid = py_str_data(jobId), *key = py_str_data(apiKey), *ep = py_str_data(endpoint);
  int64_t jidLen = py_str_size(jobId), keyLen = py_str_size(apiKey), epLen = py_str_size(endpoint);
  size_t cmdCap = static_cast<size_t>(epLen + jidLen + keyLen) + 256;
  char *cmd = static_cast<char *>(malloc(cmdCap));
  if (!cmd) py_runtime_error("内存不足");
  snprintf(cmd, cmdCap, "curl -s -X GET '%.*s/%.*s' -H 'Authorization: Bearer %.*s' 2>&1",
           static_cast<int>(epLen), ep, static_cast<int>(jidLen), jid, static_cast<int>(keyLen), key);
  char *response = captureCommand(cmd); free(cmd);
  if (!response || strlen(response) == 0) { free(response); py_runtime_error("微调状态查询失败: 无响应"); }
  PyValue result = py_str_new(response, static_cast<int64_t>(strlen(response)));
  free(response); return result;
}

extern "C" PyValue py_llm_list_finetunes(const PyValue *apiKey, const PyValue *endpoint) {
  if (apiKey->tag != PY_STR || endpoint->tag != PY_STR)
    py_runtime_error("llm_list_finetunes() 需要两个字符串参数");
  const char *key = py_str_data(apiKey), *ep = py_str_data(endpoint);
  int64_t keyLen = py_str_size(apiKey), epLen = py_str_size(endpoint);
  size_t cmdCap = static_cast<size_t>(epLen + keyLen) + 256;
  char *cmd = static_cast<char *>(malloc(cmdCap));
  if (!cmd) py_runtime_error("内存不足");
  snprintf(cmd, cmdCap, "curl -s -X GET '%.*s' -H 'Authorization: Bearer %.*s' 2>&1",
           static_cast<int>(epLen), ep, static_cast<int>(keyLen), key);
  char *response = captureCommand(cmd); free(cmd);
  if (!response || strlen(response) == 0) { free(response); py_runtime_error("微调列表查询失败: 无响应"); }
  PyValue result = py_str_new(response, static_cast<int64_t>(strlen(response)));
  free(response); return result;
}
