// Docker 容器调用 —— 通过 popen 调用 docker CLI。
//
// 设计原则：
//   1. 不引入 Docker SDK 依赖，只用系统自带的 docker 命令；
//   2. 所有操作返回 PyValue，与 PyLite 类型系统无缝对接；
//   3. 错误通过 py_runtime_error 抛出，调用方可以用 try/except 捕获。
#include "pylite/runtime.h"

#include <cstdio>
#include <cstring>
#include <cstdlib>

namespace {

char *captureCommand(const char *cmd) {
  FILE *fp = popen(cmd, "r");
  if (!fp) return nullptr;

  size_t cap = 256;
  char *buf = static_cast<char *>(malloc(cap));
  if (!buf) { pclose(fp); return nullptr; }

  size_t len = 0;
  while (!feof(fp) && !ferror(fp)) {
    if (len + 128 >= cap) {
      cap *= 2;
      char *nb = static_cast<char *>(realloc(buf, cap));
      if (!nb) { free(buf); pclose(fp); return nullptr; }
      buf = nb;
    }
    size_t n = fread(buf + len, 1, cap - len - 1, fp);
    len += n;
  }
  buf[len] = '\0';

  int rc = pclose(fp);
  if (rc != 0) return buf;
  return buf;
}

void trimNewline(char *s) {
  size_t len = strlen(s);
  while (len > 0 && (s[len - 1] == '\n' || s[len - 1] == '\r')) s[--len] = '\0';
}

}  // namespace

extern "C" PyValue py_docker_run(const PyValue *image, const PyValue *cmd) {
  if (image->tag != PY_STR || cmd->tag != PY_STR)
    py_runtime_error("docker_run() 需要两个字符串参数: image, command");

  const char *img = py_str_data(image);
  const char *command = py_str_data(cmd);
  int64_t imgLen = py_str_size(image);
  int64_t cmdLen = py_str_size(cmd);

  size_t bufSize = static_cast<size_t>(imgLen + cmdLen + 64);
  char *fullCmd = static_cast<char *>(malloc(bufSize));
  if (!fullCmd) py_runtime_error("内存不足");

  snprintf(fullCmd, bufSize, "docker run -d --rm %.*s %.*s 2>&1",
           static_cast<int>(imgLen), img, static_cast<int>(cmdLen), command);

  char *output = captureCommand(fullCmd);
  free(fullCmd);

  if (!output) py_runtime_error("docker run 失败: 无法执行 docker 命令");

  trimNewline(output);

  if (strstr(output, "Error") || strstr(output, "Cannot connect") ||
      strstr(output, "permission denied")) {
    PyValue err = py_str_new(output, static_cast<int64_t>(strlen(output)));
    free(output);
    char errBuf[512];
    snprintf(errBuf, sizeof(errBuf), "docker run 失败: %s", py_str_data(&err));
    py_runtime_error(errBuf);
  }

  PyValue result = py_str_new(output, static_cast<int64_t>(strlen(output)));
  free(output);
  return result;
}

extern "C" PyValue py_docker_ps() {
  char *output = captureCommand("docker ps 2>&1");
  if (!output) py_runtime_error("docker ps 失败: 无法执行 docker 命令");
  trimNewline(output);
  PyValue result = py_str_new(output, static_cast<int64_t>(strlen(output)));
  free(output);
  return result;
}

extern "C" void py_docker_stop(const PyValue *containerId) {
  if (containerId->tag != PY_STR)
    py_runtime_error("docker_stop() 需要一个字符串参数: container_id");

  const char *cid = py_str_data(containerId);
  int64_t cidLen = py_str_size(containerId);

  size_t bufSize = static_cast<size_t>(cidLen + 32);
  char *fullCmd = static_cast<char *>(malloc(bufSize));
  if (!fullCmd) py_runtime_error("内存不足");

  snprintf(fullCmd, bufSize, "docker stop %.*s 2>&1", static_cast<int>(cidLen), cid);

  char *output = captureCommand(fullCmd);
  free(fullCmd);

  if (output) {
    if (strstr(output, "Error") || strstr(output, "No such container")) {
      char errBuf[512];
      snprintf(errBuf, sizeof(errBuf), "docker stop 失败: %s", output);
      free(output);
      py_runtime_error(errBuf);
    }
    free(output);
  }
}

extern "C" PyValue py_docker_logs(const PyValue *containerId) {
  if (containerId->tag != PY_STR)
    py_runtime_error("docker_logs() 需要一个字符串参数: container_id");

  const char *cid = py_str_data(containerId);
  int64_t cidLen = py_str_size(containerId);

  size_t bufSize = static_cast<size_t>(cidLen + 32);
  char *fullCmd = static_cast<char *>(malloc(bufSize));
  if (!fullCmd) py_runtime_error("内存不足");

  snprintf(fullCmd, bufSize, "docker logs %.*s 2>&1", static_cast<int>(cidLen), cid);

  char *output = captureCommand(fullCmd);
  free(fullCmd);

  if (!output) py_runtime_error("docker logs 失败: 无法执行 docker 命令");

  trimNewline(output);
  PyValue result = py_str_new(output, static_cast<int64_t>(strlen(output)));
  free(output);
  return result;
}

extern "C" PyValue py_docker_pull(const PyValue *image) {
  if (image->tag != PY_STR)
    py_runtime_error("docker_pull() 需要一个字符串参数: image");

  const char *img = py_str_data(image);
  int64_t imgLen = py_str_size(image);

  size_t bufSize = static_cast<size_t>(imgLen + 32);
  char *fullCmd = static_cast<char *>(malloc(bufSize));
  if (!fullCmd) py_runtime_error("内存不足");

  snprintf(fullCmd, bufSize, "docker pull %.*s 2>&1", static_cast<int>(imgLen), img);

  char *output = captureCommand(fullCmd);
  free(fullCmd);

  if (!output) py_runtime_error("docker pull 失败: 无法执行 docker 命令");

  trimNewline(output);
  PyValue result = py_str_new(output, static_cast<int64_t>(strlen(output)));
  free(output);
  return result;
}

extern "C" PyValue py_docker_generate_dockerfile(const PyValue *baseImage,
    const PyValue *projectName, const PyValue *setupCommands,
    const PyValue *entrypoint, const PyValue *exposePort) {
  if (baseImage->tag != PY_STR || projectName->tag != PY_STR || entrypoint->tag != PY_STR)
    py_runtime_error("docker_generate_dockerfile() 需要字符串参数");

  const char *img = py_str_data(baseImage);
  const char *name = py_str_data(projectName);
  const char *cmds = (setupCommands->tag == PY_STR) ? py_str_data(setupCommands) : "";
  const char *entry = py_str_data(entrypoint);
  int64_t port = (exposePort->tag == PY_INT) ? py_as_int(*exposePort) : 0;

  int64_t imgLen = py_str_size(baseImage);
  int64_t nameLen = py_str_size(projectName);
  int64_t cmdsLen = (setupCommands->tag == PY_STR) ? py_str_size(setupCommands) : 0;
  int64_t entryLen = py_str_size(entrypoint);

  size_t cap = static_cast<size_t>(imgLen + nameLen + cmdsLen + entryLen) + 1024;
  char *dockerfile = static_cast<char *>(malloc(cap));
  if (!dockerfile) py_runtime_error("内存不足");

  int n = snprintf(dockerfile, cap,
    "# Dockerfile generated by PyLite for project: %.*s\nFROM %.*s\n\nWORKDIR /app\n\n",
    static_cast<int>(nameLen), name, static_cast<int>(imgLen), img);

  if (cmdsLen > 0) {
    const char *start = cmds;
    const char *end = cmds + cmdsLen;
    while (start < end) {
      const char *semi = static_cast<const char *>(memchr(start, ';', static_cast<size_t>(end - start)));
      if (!semi) semi = end;
      while (start < semi && (*start == ' ' || *start == '\t')) start++;
      int64_t cmdLen = semi - start;
      if (cmdLen > 0)
        n += snprintf(dockerfile + n, cap - static_cast<size_t>(n), "RUN %.*s\n", static_cast<int>(cmdLen), start);
      start = semi + 1;
    }
    n += snprintf(dockerfile + n, cap - static_cast<size_t>(n), "\n");
  }

  n += snprintf(dockerfile + n, cap - static_cast<size_t>(n), "COPY . /app\n\n");

  if (port > 0)
    n += snprintf(dockerfile + n, cap - static_cast<size_t>(n), "EXPOSE %lld\n\n", port);

  n += snprintf(dockerfile + n, cap - static_cast<size_t>(n), "CMD %.*s\n", static_cast<int>(entryLen), entry);

  PyValue result = py_str_new(dockerfile, static_cast<int64_t>(n));
  free(dockerfile);
  return result;
}

extern "C" PyValue py_docker_build(const PyValue *dockerfile, const PyValue *imageName,
    const PyValue *tag, const PyValue *contextDir) {
  if (imageName->tag != PY_STR || tag->tag != PY_STR || contextDir->tag != PY_STR)
    py_runtime_error("docker_build() 需要字符串参数");

  const char *df = (dockerfile->tag == PY_STR) ? py_str_data(dockerfile) : "";
  const char *name = py_str_data(imageName);
  const char *t = py_str_data(tag);
  const char *ctx = py_str_data(contextDir);

  int64_t dfLen = (dockerfile->tag == PY_STR) ? py_str_size(dockerfile) : 0;
  int64_t nameLen = py_str_size(imageName);
  int64_t tagLen = py_str_size(tag);
  int64_t ctxLen = py_str_size(contextDir);

  char tmpDf[256] = {0};
  bool useTempFile = false;
  if (dfLen > 0 && df[0] != '/') {
    snprintf(tmpDf, sizeof(tmpDf), "/tmp/pylite_dockerfile_%s", t);
    FILE *fp = fopen(tmpDf, "w");
    if (!fp) py_runtime_error("docker_build() 失败: 无法创建临时 Dockerfile");
    fwrite(df, 1, static_cast<size_t>(dfLen), fp);
    fclose(fp);
    useTempFile = true;
  }

  const char *dfPath = useTempFile ? tmpDf : df;

  size_t cmdCap = static_cast<size_t>(nameLen + tagLen + ctxLen) + 512;
  char *cmd = static_cast<char *>(malloc(cmdCap));
  if (!cmd) py_runtime_error("内存不足");

  if (dfLen > 0)
    snprintf(cmd, cmdCap, "docker build -f %s -t %.*s:%.*s %.*s 2>&1",
             dfPath, static_cast<int>(nameLen), name, static_cast<int>(tagLen), t, static_cast<int>(ctxLen), ctx);
  else
    snprintf(cmd, cmdCap, "docker build -t %.*s:%.*s %.*s 2>&1",
             static_cast<int>(nameLen), name, static_cast<int>(tagLen), t, static_cast<int>(ctxLen), ctx);

  char *output = captureCommand(cmd);
  free(cmd);
  if (useTempFile) remove(tmpDf);

  if (!output) py_runtime_error("docker build 失败: 无法执行 docker 命令");

  trimNewline(output);
  PyValue result = py_str_new(output, static_cast<int64_t>(strlen(output)));
  free(output);
  return result;
}

extern "C" PyValue py_docker_push(const PyValue *imageName, const PyValue *tag, const PyValue *registry) {
  if (imageName->tag != PY_STR || tag->tag != PY_STR)
    py_runtime_error("docker_push() 需要字符串参数: image_name, tag");

  const char *name = py_str_data(imageName);
  const char *t = py_str_data(tag);
  const char *reg = (registry->tag == PY_STR) ? py_str_data(registry) : "";

  int64_t nameLen = py_str_size(imageName);
  int64_t tagLen = py_str_size(tag);
  int64_t regLen = (registry->tag == PY_STR) ? py_str_size(registry) : 0;

  size_t fullCap = static_cast<size_t>(regLen + nameLen + tagLen) + 32;
  char *fullName = static_cast<char *>(malloc(fullCap));
  if (!fullName) py_runtime_error("内存不足");

  if (regLen > 0)
    snprintf(fullName, fullCap, "%.*s/%.*s:%.*s", static_cast<int>(regLen), reg,
             static_cast<int>(nameLen), name, static_cast<int>(tagLen), t);
  else
    snprintf(fullName, fullCap, "%.*s:%.*s", static_cast<int>(nameLen), name, static_cast<int>(tagLen), t);

  if (regLen > 0) {
    size_t tagCmdCap = fullCap + static_cast<size_t>(nameLen + tagLen) + 64;
    char *tagCmd = static_cast<char *>(malloc(tagCmdCap));
    if (tagCmd) {
      snprintf(tagCmd, tagCmdCap, "docker tag %.*s:%.*s %s 2>&1",
               static_cast<int>(nameLen), name, static_cast<int>(tagLen), t, fullName);
      char *tagOut = captureCommand(tagCmd);
      free(tagOut);
      free(tagCmd);
    }
  }

  size_t cmdCap = fullCap + 64;
  char *cmd = static_cast<char *>(malloc(cmdCap));
  if (!cmd) { free(fullName); py_runtime_error("内存不足"); }

  snprintf(cmd, cmdCap, "docker push %s 2>&1", fullName);

  char *output = captureCommand(cmd);
  free(cmd);
  free(fullName);

  if (!output) py_runtime_error("docker push 失败: 无法执行 docker 命令");

  trimNewline(output);
  PyValue result = py_str_new(output, static_cast<int64_t>(strlen(output)));
  free(output);
  return result;
}

extern "C" PyValue py_docker_project_package(const PyValue *projectDir, const PyValue *imageName,
    const PyValue *tag, const PyValue *baseImage, const PyValue *setupCommands,
    const PyValue *entrypoint, const PyValue *exposePort) {
  if (projectDir->tag != PY_STR || imageName->tag != PY_STR || tag->tag != PY_STR ||
      baseImage->tag != PY_STR || entrypoint->tag != PY_STR)
    py_runtime_error("docker_project_package() 需要字符串参数");

  PyValue df = py_docker_generate_dockerfile(baseImage, imageName, setupCommands, entrypoint, exposePort);

  const char *dir = py_str_data(projectDir);
  int64_t dirLen = py_str_size(projectDir);

  size_t dfPathCap = static_cast<size_t>(dirLen) + 32;
  char *dfPath = static_cast<char *>(malloc(dfPathCap));
  if (!dfPath) py_runtime_error("内存不足");

  snprintf(dfPath, dfPathCap, "%.*s/Dockerfile", static_cast<int>(dirLen), dir);

  const char *dfContent = py_str_data(&df);
  int64_t dfContentLen = py_str_size(&df);

  FILE *fp = fopen(dfPath, "w");
  if (!fp) { free(dfPath); py_runtime_error("docker_project_package() 失败: 无法写入 Dockerfile"); }
  fwrite(dfContent, 1, static_cast<size_t>(dfContentLen), fp);
  fclose(fp);

  PyValue buildResult = py_docker_build(&df, imageName, tag, projectDir);

  const char *buildOut = py_str_data(&buildResult);
  int64_t buildOutLen = py_str_size(&buildResult);
  const char *imgName = py_str_data(imageName);
  int64_t imgNameLen = py_str_size(imageName);
  const char *t = py_str_data(tag);
  int64_t tLen = py_str_size(tag);

  size_t summaryCap = static_cast<size_t>(buildOutLen + imgNameLen + tLen + dirLen) + 512;
  char *summary = static_cast<char *>(malloc(summaryCap));
  if (!summary) { free(dfPath); py_runtime_error("内存不足"); }

  snprintf(summary, summaryCap,
    "[PyLite Docker 封装完成]\n项目目录: %.*s\nDockerfile: %s\n镜像: %.*s:%.*s\n\n--- 构建输出 ---\n%.*s",
    static_cast<int>(dirLen), dir, dfPath,
    static_cast<int>(imgNameLen), imgName, static_cast<int>(tLen), t,
    static_cast<int>(buildOutLen), buildOut);

  free(dfPath);

  PyValue result = py_str_new(summary, static_cast<int64_t>(strlen(summary)));
  free(summary);
  return result;
}
