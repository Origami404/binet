// LD_PRELOAD shim that saves CUDA 12.9 nvdisasm instruction and latency descriptions.
#undef _FORTIFY_SOURCE  // glibc's fortified inline memcpy would clash with ours

#include <dlfcn.h>
#include <fcntl.h>
#include <malloc.h>
#include <unistd.h>

#include <cstdlib>
#include <cstring>

namespace {

using MemcpyFn = void *(*)(void *, const void *, size_t);
using FreeFn = void (*)(void *);

constexpr char kInstructions[] = "ARCHITECTURE \"";
constexpr char kLatencies[] = "OPERATION SETS";

MemcpyFn real_memcpy;
FreeFn real_free;
__thread int busy __attribute__((tls_model("initial-exec")));  // inside a hook already
volatile int lock_word;
int out_dir = -1;
int latencies_fd = -1;
const void *latency_src;  // staging buffer the latency chunks are copied out of
size_t instructions_len;  // longest instruction text saved so far

void lock() {
  while (__sync_lock_test_and_set(&lock_word, 1)) {
  }
}

void unlock() { __sync_lock_release(&lock_word); }

// Only used while dlsym() itself copies memory; volatile stops the compiler
// from turning the loop back into a memcpy call.
void *slow_copy(void *dst, const void *src, size_t n) {
  auto *d = static_cast<volatile unsigned char *>(dst);
  auto *s = static_cast<const unsigned char *>(src);
  for (size_t i = 0; i < n; i++) d[i] = s[i];
  return dst;
}

// Start of `key` if it begins within the first `slack` bytes of p[0, n).
const char *find_near(const void *p, size_t n, const char *key, size_t slack) {
  size_t k = strlen(key);
  size_t window = n < k + slack ? n : k + slack;
  return window < k ? nullptr : static_cast<const char *>(memmem(p, window, key, k));
}

int open_out(const char *name) {
  if (out_dir < 0) {
    const char *dir = getenv("INTERCEPT_SASS_DIR");
    out_dir = open(dir ? dir : ".", O_RDONLY | O_DIRECTORY);
  }
  return out_dir < 0 ? -1 : openat(out_dir, name, O_WRONLY | O_CREAT | O_TRUNC, 0644);
}

void write_all(int fd, const char *p, size_t n) {
  while (n > 0) {
    ssize_t k = write(fd, p, n);
    if (k <= 0) return;
    p += k;
    n -= static_cast<size_t>(k);
  }
}

void on_memcpy(const void *src, size_t n) {
  if (latency_src == nullptr && find_near(src, n, kLatencies, 8) != nullptr) {
    latency_src = src;
    latencies_fd = open_out("latencies.raw");
  }
  if (src == latency_src && latencies_fd >= 0) write_all(latencies_fd, static_cast<const char *>(src), n);
}

void on_free(void *p) {
  size_t cap = malloc_usable_size(p);
  const char *text = find_near(p, cap, kInstructions, 16);
  if (text == nullptr) return;
  size_t room = cap - static_cast<size_t>(text - static_cast<const char *>(p));
  const char *end = static_cast<const char *>(memchr(text, 0, room));
  size_t len = end != nullptr ? static_cast<size_t>(end - text) : room;
  if (len <= instructions_len) return;  // keep the full text, not an LZ4 fragment
  int fd = open_out("instructions.raw");
  if (fd < 0) return;
  write_all(fd, text, len);
  close(fd);
  instructions_len = len;
}

}  // namespace

extern "C" void *memcpy(void *dst, const void *src, size_t n) noexcept {
  if (real_memcpy == nullptr) {
    if (busy) return slow_copy(dst, src, n);
    busy = 1;
    real_memcpy = reinterpret_cast<MemcpyFn>(dlsym(RTLD_NEXT, "memcpy"));
    busy = 0;
  }
  if (!busy && n > 0) {
    busy = 1;
    lock();
    on_memcpy(src, n);
    unlock();
    busy = 0;
  }
  return real_memcpy(dst, src, n);
}

extern "C" void free(void *p) noexcept {
  if (real_free == nullptr) {
    if (busy) return;  // dlsym() bootstrap: leak rather than recurse
    busy = 1;
    real_free = reinterpret_cast<FreeFn>(dlsym(RTLD_NEXT, "free"));
    busy = 0;
  }
  if (p != nullptr && !busy) {
    busy = 1;
    lock();
    on_free(p);
    unlock();
    busy = 0;
  }
  real_free(p);
}
