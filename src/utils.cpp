#include "utils.hpp"

#include <iostream>
#include <vector>
#include <Eigen/Eigen>
#include <fstream>

#include <spdlog/spdlog.h>

extern int64_t n;
extern std::vector<int> nomissgrmid;
extern Eigen::MatrixXf _A;
extern Eigen::MatrixXf _B;
extern Eigen::VectorXf _diag;

/// PerfTimer
PerfTimer::PerfTimer(const std::string &name)
    : name_(name), start_time_(std::chrono::high_resolution_clock::now()),
      elapsed_time_(start_time_), stopped_(false) {}

PerfTimer::~PerfTimer() {
  if (!stopped_) {
    stop();
  }
}

void PerfTimer::elapsed(const std::string &elapsed_name) {
  if (stopped_)
    return; // Avoid double stop
  auto result = elapsed_to(elapsed_time_);
  std::string time_str = result.first;
  elapsed_time_ = result.second;
  spdlog::info("[perf] ===> elapsed {}, cost {}", elapsed_name, time_str);
}

void PerfTimer::stop() {
  if (stopped_)
    return; // Avoid double stop
  auto result = elapsed_to(start_time_);
  std::string time_str = result.first;
  spdlog::info("[perf] ===> {}, cost {}", name_, time_str);
  stopped_ = true;
}

std::pair<std::string, std::chrono::high_resolution_clock::time_point>
PerfTimer::elapsed_to(
    const std::chrono::high_resolution_clock::time_point &time_point) {
  auto end_time = std::chrono::high_resolution_clock::now();
  auto duration = end_time - time_point;
  std::chrono::duration<double, std::milli> milliseconds = duration;
  std::chrono::duration<double> seconds = duration;
  std::chrono::duration<double, std::ratio<60>> minutes = duration;
  std::chrono::duration<double, std::ratio<3600>> hours = duration;

  std::string time_str;
  if (milliseconds.count() < 1000.0) {
    time_str = fmt::format("{}ms", milliseconds.count());
  } else if (seconds.count() < 60.0) {
    time_str = fmt::format("{:02}:{:02}:{:02.3f}s", 0, 0, seconds.count());
  } else if (minutes.count() < 60) {
    auto min = std::trunc(minutes.count());
    auto sec = seconds.count() - 60.0 * min;
    time_str = fmt::format("{:02}:{:02}:{:02.3f}s", 0, min, sec);
  } else {
    auto h = std::trunc(hours.count());
    auto min = std::trunc(minutes.count()) - 60 * h;
    auto sec = seconds.count() - 60 * std::trunc(minutes.count());
    time_str = fmt::format("{:02}:{:02}:{:02.3f}s", h, min, sec);
  }

  return {time_str, end_time};
}


bool compare_float(float a, float b, float epsilon) {
  return std::fabs(a - b) < epsilon;
}

void read_grmA_oneCPU_forrt_withmiss(const std::string& grm_path, int64_t start, int64_t end, float var) {
    std::ifstream fin(grm_path, std::ios::in | std::ios::binary);
    if (!fin.is_open()) std::cout << "can not open the file\n";

    // Increase internal buffer
    std::vector<char> stream_buffer(8 * 1024 * 1024);
    fin.rdbuf()->pubsetbuf(stream_buffer.data(), stream_buffer.size());

    std::vector<char> buffer((nomissgrmid[end - 1] + 1) * sizeof(float));
    for (int i = start; i < end; i++) {
        if(i % 10000 == 0) spdlog::info("Reading id: {}", i);
        int64_t nomiss_i = nomissgrmid[i];
        int64_t offset = nomiss_i * (nomiss_i + 1) / 2;
        fin.seekg(offset * sizeof(float), std::ios::beg);
        fin.read(buffer.data(), sizeof(float) * (nomiss_i + 1));
        float* values = reinterpret_cast<float*>(buffer.data());
        for (int j = 0; j <= i; j++) {
            float val = values[nomissgrmid[j]] * var;
            if(UNLIKELY(i == j)) _diag(i) += val;
            else _A(i, j) += val;
        }
    }
    fin.close();
}

void read_grmAB_oneCPU_forrt_withmiss(const std::string& grm_path, int64_t start, int64_t end, float var) {
    std::ifstream fin(grm_path, std::ios::in | std::ios::binary);
    if (!fin.is_open()) std::cout << "can not open the file\n";

    // Increase internal buffer
    std::vector<char> stream_buffer(8 * 1024 * 1024);
    fin.rdbuf()->pubsetbuf(stream_buffer.data(), stream_buffer.size());

    int64_t halfn = (n + 1)/2;
    std::vector<char> buffer((nomissgrmid[end - 1] + 1) * sizeof(float));
    for (int i = start; i < end; i++) {
        if(i % 10000 == 0) spdlog::info("Reading id: {}", i);
        int64_t nomiss_i = nomissgrmid[i];
        int64_t offset = nomiss_i * (nomiss_i + 1) / 2;
        fin.seekg(offset * sizeof(float), std::ios::beg);
        fin.read(buffer.data(), sizeof(float) * (nomiss_i + 1));
        float* values = reinterpret_cast<float*>(buffer.data());
        for (int j = 0; j <= i; j++) {
            float val = values[nomissgrmid[j]] * var;
            if(UNLIKELY(i == j)) _diag(i) += val;
            else if(j < halfn) _B(i - halfn,j) += val;
            else _A(j - halfn, i - halfn) += val;
        }
    }
    fin.close();
}



void read_grmAB_forrt_parallel(const std::string& file, float var) {
    PerfTimer _perf_timer(__FUNCTION__);

    int64_t halfn = (n + 1) / 2;
    int chunks = std::min(16, omp_get_num_procs());
    int64_t upper_count = gsum(halfn);
    int64_t chunk_size = upper_count / chunks;
    Eigen::MatrixXi chunk_ranges(2, chunks);

    // Compute grmA chunk range
    chunk_ranges(0, 0) = 0;
    for(int i = 1; i < chunks; i++) {
        chunk_ranges(0, i) = floor(sqrt(2 * chunk_size * i - 0.25) + 0.5);
    }
    chunk_ranges.row(1).segment(0, chunks - 1) = chunk_ranges.row(0).segment(1, chunks - 1);
    chunk_ranges(1, chunks - 1) = halfn;

    // Read grmA
    #pragma omp parallel for num_threads(chunks)
    for(int i = 0; i < chunks; i++) {
        read_grmA_oneCPU_forrt_withmiss(file, chunk_ranges(0, i), chunk_ranges(1, i), var);
    }

    // Compute grmAB chunk range
    int64_t lower_count = gsum(n) - upper_count;
    chunk_size = lower_count / chunks;
    chunk_ranges(0, 0) = halfn;
    for(int i = 1; i < chunks; i++) {
        chunk_ranges(0, i) = floor(sqrt(2 * (chunk_size * i + upper_count)  - 0.25) + 0.5);
    }
    chunk_ranges.row(1).segment(0, chunks - 1) = chunk_ranges.row(0).segment(1, chunks - 1);
    chunk_ranges(1, chunks - 1) = n;

    // Read grmAB
    #pragma omp parallel for num_threads(chunks)
    for(int i = 0; i < chunks; i++) {
        read_grmAB_oneCPU_forrt_withmiss(file, chunk_ranges(0, i), chunk_ranges(1, i), var);
    }
}


void read_grmA_oneCPU_withmiss_batch(const std::string& grm_path, int64_t start, int64_t end) {
    std::ifstream fin(grm_path, std::ios::in | std::ios::binary);
    if (!fin.is_open()) std::cout << "can not open the file\n";

    // Increase internal buffer
    std::vector<char> stream_buffer(8 * 1024 * 1024);
    fin.rdbuf()->pubsetbuf(stream_buffer.data(), stream_buffer.size());

    std::vector<char> buffer((nomissgrmid[end - 1] + 1) * sizeof(float));
    for(int i = start; i< end; i++){
        if(i % 10000 == 0) spdlog::info("Reading id: {}", i);

        int64_t nomiss_i = nomissgrmid[i];
        int64_t offset = nomiss_i * (nomiss_i + 1) / 2;
        fin.seekg(offset * sizeof(float), std::ios::beg);
        fin.read(buffer.data(), sizeof(float) * (nomiss_i + 1));
        float* values = reinterpret_cast<float*>(buffer.data());
        for(int j = 0; j <= i; j++) {
            float val = values[nomissgrmid[j]];
            if(UNLIKELY(i == j)) _diag(i) = val;
            else _A(i,j) = val;
        }
    }
    fin.close();
}


void read_grmAB_oneCPU_withmiss_batch(const std::string& grm_path, int64_t start, int64_t end) {
    int64_t halfn = (n + 1) / 2;
    std::ifstream fin(grm_path, std::ios::in | std::ios::binary);
    if (!fin.is_open()) std::cout << "can not open the file\n";

    // Increase internal buffer
    std::vector<char> stream_buffer(8 * 1024 * 1024);
    fin.rdbuf()->pubsetbuf(stream_buffer.data(), stream_buffer.size());

    std::vector<char> buffer((nomissgrmid[end - 1] + 1) * sizeof(float));
    for(int i = start; i < end; i++) {
        if(i % 10000 == 0) spdlog::info("Reading id: {}", i);
        int64_t nomiss_i = nomissgrmid[i];
        int64_t offset = gsum(nomiss_i);
        fin.seekg(offset * sizeof(float), std::ios::beg);
        fin.read(buffer.data(), sizeof(float) * (nomiss_i + 1));
        float* values = reinterpret_cast<float*>(buffer.data());
        for(int j = 0; j <= i; j++){
            float val = values[nomissgrmid[j]];
            if(UNLIKELY(i == j)) _diag(i) = val;
            else if(j < halfn) _B(i - halfn,j) = val;
            else _A(j - halfn, i - halfn) = val;
        }
    }
    fin.close();
}

void read_grmAB_faster_parallel(const std::string& grm_path) {
    PerfTimer _perf_timer(__FUNCTION__);

    int64_t halfn = (n + 1) / 2;
    _A.setZero(halfn, halfn);
    _B.setZero(halfn, halfn);
    _diag.setZero(n);

    int chunks = std::min(16, omp_get_num_procs());
    int64_t upper_count = gsum(halfn);
    int64_t chunk_size = upper_count / chunks;
    Eigen::MatrixXi chunk_ranges(2, chunks);
    chunk_ranges(0, 0) = 0;
    for(int i = 1; i < chunks; i++) {
        chunk_ranges(0, i) = floor(sqrt(2 * chunk_size * i - 0.25) + 0.5);
    }
    chunk_ranges.row(1).segment(0, chunks - 1) = chunk_ranges.row(0).segment(1, chunks - 1);
    chunk_ranges(1, chunks - 1) = halfn;

    #pragma omp parallel for num_threads(chunks)
    for(int i = 0; i < chunks; i++) {
        read_grmA_oneCPU_withmiss_batch(grm_path, chunk_ranges(0, i), chunk_ranges(1, i));
    }

    int64_t lower_count = gsum(n) - upper_count;
    chunk_size = lower_count / chunks;
    chunk_ranges(0, 0) = halfn;
    for(int i = 1; i < chunks; i++) {
        chunk_ranges(0, i) = floor(sqrt(2 * (chunk_size * i + upper_count)  - 0.25) + 0.5);
    }
    chunk_ranges.row(1).segment(0, chunks - 1) = chunk_ranges.row(0).segment(1, chunks - 1);
    chunk_ranges(1, chunks - 1) = n;

    #pragma omp parallel for num_threads(chunks)
    for(int i = 0; i < chunks; i++) {
        read_grmAB_oneCPU_withmiss_batch(grm_path, chunk_ranges(0, i), chunk_ranges(1, i));
    }

}
