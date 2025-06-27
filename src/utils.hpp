#pragma once

#if defined(__APPLE__) && defined(__aarch64__)
#  define EIGEN_USE_BLAS
#  define EIGEN_USE_LAPACKE
#else
#  define EIGEN_USE_MKL_ALL
#endif

#include <cstdint>
#include <string>
#include <chrono>

#if defined(__GNUC__) || defined(__clang__)
#  define LIKELY(x)    __builtin_expect(!!(x), 1)
#  define UNLIKELY(x)  __builtin_expect(!!(x), 0)
#else
#  define LIKELY(x)    (x)
#  define UNLIKELY(x)  (x)
#endif

class PerfTimer {
public:
  explicit PerfTimer(const std::string &name = "");
  ~PerfTimer();

  void elapsed(const std::string &elapsed_name);
  void stop();

private:
  std::pair<std::string, std::chrono::high_resolution_clock::time_point>
  elapsed_to(const std::chrono::high_resolution_clock::time_point &time_point);

private:
  std::string name_;
  const std::chrono::high_resolution_clock::time_point start_time_;
  std::chrono::high_resolution_clock::time_point elapsed_time_;
  bool stopped_;
};

/// Guass summation, sum of an arithmetic series.
template <typename T> T gsum(T x) { return (1 + x) * x / 2; }

bool compare_float(float a, float b, float epsilon = 1e-5f);

void read_grmAB_forrt_parallel(const std::string& file, float var);
void read_grmA_oneCPU_forrt_withmiss(const std::string& grm_path, int64_t start, int64_t end, float var);
void read_grmAB_oneCPU_forrt_withmiss(const std::string& grm_path, int64_t start, int64_t end, float var);

void read_grmA_oneCPU_withmiss_batch(const std::string& grm_path, int64_t start, int64_t end);
void read_grmAB_oneCPU_withmiss_batch(const std::string& grm_path, int64_t start, int64_t end);
void read_grmAB_faster_parallel(const std::string& grm_path);
