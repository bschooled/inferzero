// External iGPU engine-busy and frequency sampler for Linux / Intel i915 & Level Zero Sysman.
// Samples at fixed intervals (default 100 ms):
//   1. i915 PMU counters via perf_event_open (rcs0-busy, bcs0-busy, ccs0-busy, rc6-residency-gt0/gt1, actual-frequency)
//   2. Per-process DRM fdinfo (/proc/<pid>/fdinfo/*): drm-engine-render, drm-engine-compute, drm-engine-copy (ns)
//   3. Sysfs GT RC6 residency (/sys/class/drm/card*/gt/gt*/rc6_residency_ms) and actual frequency (rps_act_freq_mhz)
//   4. Level Zero Sysman zesEngineGetActivity (if exposed by the driver)
//
// Usage:
//   gpu_busy_sampler [--pid PID] [--interval-ms 100] [--duration-ms 0] [--out FILE] [--probe]
//   If --pid is given, stops automatically when that process exits (or after --duration-ms if > 0).

#include <algorithm>
#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <filesystem>
#include <fstream>
#include <linux/perf_event.h>
#include <sstream>
#include <string>
#include <sys/ioctl.h>
#include <sys/syscall.h>
#include <thread>
#include <unistd.h>
#include <vector>

namespace fs = std::filesystem;
using clk = std::chrono::steady_clock;

static std::atomic<bool> g_stop{false};
static void sig_handler(int) { g_stop.store(true); }

static long perf_event_open_sys(struct perf_event_attr* hw_event, pid_t pid, int cpu, int group_fd, unsigned long flags) {
  return syscall(__NR_perf_event_open, hw_event, pid, cpu, group_fd, flags);
}

static int read_int_file(const std::string& path, int64_t& out) {
  std::ifstream f(path);
  if (!f) return -1;
  if (!(f >> out)) return -1;
  return 0;
}

static int read_hex_config(const std::string& path, uint64_t& cfg) {
  std::ifstream f(path);
  if (!f) return -1;
  std::string s;
  std::getline(f, s);
  const size_t p = s.find("config=");
  if (p == std::string::npos) return -1;
  cfg = std::strtoull(s.c_str() + p + 7, nullptr, 0);
  return 0;
}

struct PmuCounter {
  std::string name;
  uint64_t config = 0;
  int fd = -1;
  uint64_t last_val = 0;
  bool open(int pmu_type, uint64_t cfg) {
    config = cfg;
    struct perf_event_attr pe{};
    pe.type = (uint32_t)pmu_type;
    pe.size = sizeof(pe);
    pe.config = cfg;
    pe.disabled = 0;
    pe.inherit = 1;
    // System-wide on CPU 0 for uncore/i915 PMU
    fd = (int)perf_event_open_sys(&pe, -1, 0, -1, 0);
    if (fd >= 0) {
      uint64_t v = 0;
      if (::read(fd, &v, sizeof(v)) == sizeof(v)) last_val = v;
      return true;
    }
    return false;
  }
  uint64_t read_val() {
    if (fd < 0) return 0;
    uint64_t v = 0;
    if (::read(fd, &v, sizeof(v)) != sizeof(v)) return last_val;
    return v;
  }
  ~PmuCounter() {
    if (fd >= 0) ::close(fd);
  }
};

struct FdInfoTotals {
  uint64_t render_ns = 0;
  uint64_t compute_ns = 0;
  uint64_t copy_ns = 0;
  uint64_t video_ns = 0;
  int clients = 0;
};

// Reads /proc/<pid>/fdinfo/* for unique drm-client-id entries of i915/xe
static FdInfoTotals read_pid_fdinfo(pid_t pid) {
  FdInfoTotals tot{};
  if (pid <= 0) return tot;
  std::error_code ec;
  const fs::path dir = fs::path("/proc") / std::to_string(pid) / "fdinfo";
  if (!fs::exists(dir, ec)) return tot;
  std::vector<int64_t> seen_clients;
  for (const auto& entry : fs::directory_iterator(dir, ec)) {
    if (ec) break;
    std::ifstream f(entry.path());
    if (!f) continue;
    std::string line;
    bool is_drm = false;
    int64_t client_id = -1;
    uint64_t r_ns = 0, c_ns = 0, b_ns = 0, v_ns = 0;
    while (std::getline(f, line)) {
      if (line.rfind("drm-driver:", 0) == 0) {
        is_drm = true;
      } else if (line.rfind("drm-client-id:", 0) == 0) {
        client_id = std::strtoll(line.c_str() + 14, nullptr, 10);
      } else if (line.rfind("drm-engine-render:", 0) == 0) {
        r_ns = std::strtoull(line.c_str() + 18, nullptr, 10);
      } else if (line.rfind("drm-engine-compute:", 0) == 0) {
        c_ns = std::strtoull(line.c_str() + 19, nullptr, 10);
      } else if (line.rfind("drm-engine-copy:", 0) == 0) {
        b_ns = std::strtoull(line.c_str() + 16, nullptr, 10);
      } else if (line.rfind("drm-engine-video:", 0) == 0) {
        v_ns = std::strtoull(line.c_str() + 17, nullptr, 10);
      }
    }
    if (is_drm && client_id >= 0) {
      if (std::find(seen_clients.begin(), seen_clients.end(), client_id) == seen_clients.end()) {
        seen_clients.push_back(client_id);
        tot.render_ns += r_ns;
        tot.compute_ns += c_ns;
        tot.copy_ns += b_ns;
        tot.video_ns += v_ns;
        tot.clients++;
      }
    }
  }
  return tot;
}

// Level Zero Sysman engine activity probe
struct ZesEngineProps {
  uint32_t stype = 0x7;  // ZES_STRUCTURE_TYPE_ENGINE_PROPERTIES
  const void* pNext = nullptr;
  int32_t type = -1;     // zes_engine_group_t: 0 ALL, 1 COMPUTE_ALL, 2 MEDIA_ALL, 3 COPY_ALL, 4 RENDER_SINGLE, 5 COMPUTE_SINGLE
  uint8_t onSubdevice = 0;
  uint32_t subdeviceId = 0;
};
struct ZesEngineStats {
  uint64_t activeTime = 0;
  uint64_t timestamp = 0;
};
struct SysmanProbe {
  bool ze_ok = false;
  int engine_domains = 0;
  std::string status = "not probed";
  void* lib = nullptr;
  using PFN_zesEngineGetProperties = int (*)(void*, ZesEngineProps*);
  using PFN_zesEngineGetActivity = int (*)(void*, ZesEngineStats*);
  PFN_zesEngineGetProperties fn_props = nullptr;
  PFN_zesEngineGetActivity fn_act = nullptr;
  std::vector<void*> groups;
  std::vector<int32_t> group_types;
  std::vector<ZesEngineStats> prev_stats;

  void probe() {
    setenv("ZES_ENABLE_SYSMAN", "1", 1);
    lib = dlopen("libze_loader.so.1", RTLD_LAZY);
    if (!lib) { status = "libze_loader.so.1 not found"; return; }
    using PFN_zeInit = int (*)(uint32_t);
    using PFN_zeDriverGet = int (*)(uint32_t*, void**);
    using PFN_zeDeviceGet = int (*)(void*, uint32_t*, void**);
    using PFN_zesDeviceEnumEngineGroups = int (*)(void*, uint32_t*, void**);
    auto fn_init = (PFN_zeInit)dlsym(lib, "zeInit");
    auto fn_drv = (PFN_zeDriverGet)dlsym(lib, "zeDriverGet");
    auto fn_dev = (PFN_zeDeviceGet)dlsym(lib, "zeDeviceGet");
    auto fn_eng = (PFN_zesDeviceEnumEngineGroups)dlsym(lib, "zesDeviceEnumEngineGroups");
    fn_props = (PFN_zesEngineGetProperties)dlsym(lib, "zesEngineGetProperties");
    fn_act = (PFN_zesEngineGetActivity)dlsym(lib, "zesEngineGetActivity");
    if (!fn_init || !fn_drv || !fn_dev || !fn_eng) { status = "Sysman symbols missing"; return; }
    if (fn_init(0) != 0) { status = "zeInit failed"; return; }
    uint32_t nd = 0;
    if (fn_drv(&nd, nullptr) != 0 || nd == 0) { status = "no L0 driver"; return; }
    std::vector<void*> drvs(nd);
    fn_drv(&nd, drvs.data());
    uint32_t ng = 0;
    void* dev = nullptr;
    for (void* d : drvs) {
      uint32_t c = 1;
      if (fn_dev(d, &c, &dev) == 0 && c > 0 && dev) break;
    }
    if (!dev) { status = "no L0 device"; return; }
    int r = fn_eng(dev, &ng, nullptr);
    char buf[256];
    snprintf(buf, sizeof(buf), "zesDeviceEnumEngineGroups returned 0x%x, count=%u", (unsigned)r, ng);
    status = buf;
    engine_domains = (r == 0) ? (int)ng : 0;
    if (r == 0 && ng > 0 && fn_props && fn_act) {
      groups.resize(ng);
      fn_eng(dev, &ng, groups.data());
      group_types.resize(ng, -1);
      prev_stats.resize(ng);
      std::string det = " [types:";
      for (uint32_t i = 0; i < ng; ++i) {
        ZesEngineProps p{};
        if (fn_props(groups[i], &p) == 0) group_types[i] = p.type;
        ZesEngineStats s{};
        int ra = fn_act(groups[i], &s);
        prev_stats[i] = s;
        det += " g" + std::to_string(i) + "(t=" + std::to_string(group_types[i]) + ",act=" + std::to_string(s.activeTime) + ",ts=" + std::to_string(s.timestamp) + ",r=" + std::to_string(ra) + ")";
      }
      det += "]";
      status += det;
    }
  }
};

static std::string find_drm_card() {
  std::error_code ec;
  for (const auto& e : fs::directory_iterator("/sys/class/drm", ec)) {
    const std::string name = e.path().filename().string();
    if (name.rfind("card", 0) == 0 && name.find('-') == std::string::npos) {
      if (fs::exists(e.path() / "gt" / "gt0" / "rc6_residency_ms", ec)) return e.path().string();
    }
  }
  return "/sys/class/drm/card1";
}

int main(int argc, char** argv) {
  pid_t target_pid = 0;
  int interval_ms = 100;
  int duration_ms = 0;
  std::string out_path;
  bool probe_only = false;

  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    if (a == "--pid" && i + 1 < argc) target_pid = (pid_t)atoi(argv[++i]);
    else if (a == "--interval-ms" && i + 1 < argc) interval_ms = std::max(10, atoi(argv[++i]));
    else if (a == "--duration-ms" && i + 1 < argc) duration_ms = atoi(argv[++i]);
    else if (a == "--out" && i + 1 < argc) out_path = argv[++i];
    else if (a == "--probe") probe_only = true;
  }

  std::signal(SIGINT, sig_handler);
  std::signal(SIGTERM, sig_handler);

  const std::string card = find_drm_card();
  int64_t pmu_type = -1;
  read_int_file("/sys/devices/i915/type", pmu_type);

  PmuCounter rcs_busy, ccs_busy, bcs_busy, awake_gt0;
  bool pmu_ok = false;
  int pmu_errno = 0;
  if (pmu_type > 0) {
    uint64_t cfg_rcs = 0, cfg_ccs = 0x4000, cfg_bcs = 0x1000, cfg_awk = 0x100004;
    read_hex_config("/sys/devices/i915/events/rcs0-busy", cfg_rcs);
    read_hex_config("/sys/devices/i915/events/ccs0-busy", cfg_ccs);
    read_hex_config("/sys/devices/i915/events/bcs0-busy", cfg_bcs);
    read_hex_config("/sys/devices/i915/events/software-gt-awake-time-gt0", cfg_awk);
    if (rcs_busy.open((int)pmu_type, cfg_rcs)) {
      pmu_ok = true;
      ccs_busy.open((int)pmu_type, cfg_ccs);
      bcs_busy.open((int)pmu_type, cfg_bcs);
      awake_gt0.open((int)pmu_type, cfg_awk);
    } else {
      pmu_errno = errno;
    }
  }

  int64_t paranoid = -1, i915_paranoid = -1, oa_rate = -1;
  read_int_file("/proc/sys/kernel/perf_event_paranoid", paranoid);
  read_int_file("/proc/sys/dev/i915/perf_stream_paranoid", i915_paranoid);
  read_int_file("/proc/sys/dev/i915/oa_max_sample_rate", oa_rate);

  if (probe_only) {
    SysmanProbe sp;
    sp.probe();
    printf("drm_card: %s\n", card.c_str());
    printf("i915_pmu_type: %lld | pmu_open_ok: %s (errno=%d, euid=%d, perf_event_paranoid=%lld)\n",
           (long long)pmu_type, pmu_ok ? "yes" : "no", pmu_errno, (int)geteuid(), (long long)paranoid);
    printf("i915_oa: perf_stream_paranoid=%lld, oa_max_sample_rate=%lld\n",
           (long long)i915_paranoid, (long long)oa_rate);
    printf("l0_sysman_engine: %s (domains=%d)\n", sp.status.c_str(), sp.engine_domains);
    int64_t rc6_0 = -1, rc6_1 = -1, freq_0 = -1, freq_1 = -1;
    read_int_file(card + "/gt/gt0/rc6_residency_ms", rc6_0);
    read_int_file(card + "/gt/gt1/rc6_residency_ms", rc6_1);
    read_int_file(card + "/gt/gt0/rps_act_freq_mhz", freq_0);
    read_int_file(card + "/gt/gt1/rps_act_freq_mhz", freq_1);
    printf("sysfs_gt0: rc6_residency_ms=%lld, rps_act_freq_mhz=%lld\n", (long long)rc6_0, (long long)freq_0);
    printf("sysfs_gt1: rc6_residency_ms=%lld, rps_act_freq_mhz=%lld\n", (long long)rc6_1, (long long)freq_1);
    if (target_pid > 0) {
      FdInfoTotals ft = read_pid_fdinfo(target_pid);
      printf("fdinfo(pid=%d): clients=%d render_ns=%llu compute_ns=%llu copy_ns=%llu\n",
             (int)target_pid, ft.clients, (unsigned long long)ft.render_ns,
             (unsigned long long)ft.compute_ns, (unsigned long long)ft.copy_ns);
    }
    return 0;
  }

  FILE* out = stdout;
  if (!out_path.empty()) {
    out = fopen(out_path.c_str(), "w");
    if (!out) {
      perror("fopen");
      return 1;
    }
  }

  fprintf(out, "# gpu_busy_sampler card=%s pmu_ok=%d target_pid=%d interval_ms=%d\n",
          card.c_str(), pmu_ok ? 1 : 0, (int)target_pid, interval_ms);
  fprintf(out, "sample,elapsed_ms,dt_ms,pmu_rcs0_busy_pct,pmu_ccs0_busy_pct,pmu_bcs0_busy_pct,pmu_gt0_awake_pct,"
               "fdinfo_render_busy_pct,fdinfo_compute_busy_pct,fdinfo_copy_busy_pct,fdinfo_total_busy_pct,"
               "gt0_awake_rc6_pct,gt1_awake_rc6_pct,gt0_freq_mhz,gt1_freq_mhz\n");
  fflush(out);

  auto t_start = clk::now();
  auto t_prev = t_start;
  uint64_t prev_rcs = rcs_busy.read_val();
  uint64_t prev_ccs = ccs_busy.read_val();
  uint64_t prev_bcs = bcs_busy.read_val();
  uint64_t prev_awk = awake_gt0.read_val();
  FdInfoTotals prev_fd = read_pid_fdinfo(target_pid);
  int64_t prev_rc6_0 = 0, prev_rc6_1 = 0;
  read_int_file(card + "/gt/gt0/rc6_residency_ms", prev_rc6_0);
  read_int_file(card + "/gt/gt1/rc6_residency_ms", prev_rc6_1);

  int sample = 0;
  double sum_dt_ms = 0, sum_pmu_rcs_ms = 0, sum_pmu_ccs_ms = 0, sum_fd_tot_ms = 0, sum_gt0_awk_ms = 0;
  double active_dt_ms = 0, active_pmu_rcs_ms = 0, active_pmu_ccs_ms = 0, active_fd_tot_ms = 0, active_gt0_awk_ms = 0;
  int active_samples = 0;

  while (!g_stop.load()) {
    std::this_thread::sleep_for(std::chrono::milliseconds(interval_ms));
    auto t_now = clk::now();
    const double elapsed_ms = std::chrono::duration<double, std::milli>(t_now - t_start).count();
    const double dt_ms = std::chrono::duration<double, std::milli>(t_now - t_prev).count();
    const double dt_ns = dt_ms * 1e6;
    t_prev = t_now;
    if (dt_ms <= 0) continue;

    if (target_pid > 0 && ::kill(target_pid, 0) != 0) {
      break;
    }

    uint64_t cur_rcs = rcs_busy.read_val();
    uint64_t cur_ccs = ccs_busy.read_val();
    uint64_t cur_bcs = bcs_busy.read_val();
    uint64_t cur_awk = awake_gt0.read_val();
    FdInfoTotals cur_fd = read_pid_fdinfo(target_pid);
    int64_t cur_rc6_0 = prev_rc6_0, cur_rc6_1 = prev_rc6_1, freq_0 = 0, freq_1 = 0;
    read_int_file(card + "/gt/gt0/rc6_residency_ms", cur_rc6_0);
    read_int_file(card + "/gt/gt1/rc6_residency_ms", cur_rc6_1);
    read_int_file(card + "/gt/gt0/rps_act_freq_mhz", freq_0);
    read_int_file(card + "/gt/gt1/rps_act_freq_mhz", freq_1);

    const double pmu_rcs_pct = pmu_ok ? 100.0 * (double)(cur_rcs - prev_rcs) / dt_ns : -1.0;
    const double pmu_ccs_pct = pmu_ok ? 100.0 * (double)(cur_ccs - prev_ccs) / dt_ns : -1.0;
    const double pmu_bcs_pct = pmu_ok ? 100.0 * (double)(cur_bcs - prev_bcs) / dt_ns : -1.0;
    const double pmu_awk_pct = pmu_ok ? 100.0 * (double)(cur_awk - prev_awk) / dt_ns : -1.0;

    const double fd_r_ns = (cur_fd.render_ns >= prev_fd.render_ns) ? (double)(cur_fd.render_ns - prev_fd.render_ns) : 0.0;
    const double fd_c_ns = (cur_fd.compute_ns >= prev_fd.compute_ns) ? (double)(cur_fd.compute_ns - prev_fd.compute_ns) : 0.0;
    const double fd_b_ns = (cur_fd.copy_ns >= prev_fd.copy_ns) ? (double)(cur_fd.copy_ns - prev_fd.copy_ns) : 0.0;
    const double fd_r_pct = 100.0 * fd_r_ns / dt_ns;
    const double fd_c_pct = 100.0 * fd_c_ns / dt_ns;
    const double fd_b_pct = 100.0 * fd_b_ns / dt_ns;
    const double fd_tot_pct = 100.0 * (fd_r_ns + fd_c_ns + fd_b_ns) / dt_ns;

    const double d_rc6_0 = (double)std::max<int64_t>(0, cur_rc6_0 - prev_rc6_0);
    const double d_rc6_1 = (double)std::max<int64_t>(0, cur_rc6_1 - prev_rc6_1);
    const double gt0_awk_pct = std::clamp(100.0 * (1.0 - d_rc6_0 / dt_ms), 0.0, 100.0);
    const double gt1_awk_pct = std::clamp(100.0 * (1.0 - d_rc6_1 / dt_ms), 0.0, 100.0);

    prev_rcs = cur_rcs; prev_ccs = cur_ccs; prev_bcs = cur_bcs; prev_awk = cur_awk;
    prev_fd = cur_fd;
    prev_rc6_0 = cur_rc6_0; prev_rc6_1 = cur_rc6_1;

    ++sample;
    fprintf(out, "%d,%.1f,%.1f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%lld,%lld\n",
            sample, elapsed_ms, dt_ms,
            pmu_rcs_pct, pmu_ccs_pct, pmu_bcs_pct, pmu_awk_pct,
            fd_r_pct, fd_c_pct, fd_b_pct, fd_tot_pct,
            gt0_awk_pct, gt1_awk_pct, (long long)freq_0, (long long)freq_1);

    sum_dt_ms += dt_ms;
    if (pmu_ok) {
      sum_pmu_rcs_ms += dt_ms * (pmu_rcs_pct / 100.0);
      sum_pmu_ccs_ms += dt_ms * (pmu_ccs_pct / 100.0);
    }
    sum_fd_tot_ms += (fd_r_ns + fd_c_ns + fd_b_ns) / 1e6;
    sum_gt0_awk_ms += dt_ms * (gt0_awk_pct / 100.0);

    // Count active inference samples (where GT0 frequency > 0 or engine busy > 5%)
    if (freq_0 > 0 || fd_tot_pct > 5.0 || pmu_rcs_pct > 5.0 || pmu_ccs_pct > 5.0) {
      active_dt_ms += dt_ms;
      if (pmu_ok) {
        active_pmu_rcs_ms += dt_ms * (pmu_rcs_pct / 100.0);
        active_pmu_ccs_ms += dt_ms * (pmu_ccs_pct / 100.0);
      }
      active_fd_tot_ms += (fd_r_ns + fd_c_ns + fd_b_ns) / 1e6;
      active_gt0_awk_ms += dt_ms * (gt0_awk_pct / 100.0);
      ++active_samples;
    }

    if (duration_ms > 0 && elapsed_ms >= duration_ms) break;
  }

  fprintf(out, "# summary total_samples=%d total_ms=%.1f active_samples=%d active_ms=%.1f "
               "active_pmu_rcs0_pct=%.2f active_pmu_ccs0_pct=%.2f active_fdinfo_busy_pct=%.2f active_gt0_awake_pct=%.2f\n",
          sample, sum_dt_ms, active_samples, active_dt_ms,
          active_dt_ms > 0 ? 100.0 * active_pmu_rcs_ms / active_dt_ms : 0.0,
          active_dt_ms > 0 ? 100.0 * active_pmu_ccs_ms / active_dt_ms : 0.0,
          active_dt_ms > 0 ? 100.0 * active_fd_tot_ms / active_dt_ms : 0.0,
          active_dt_ms > 0 ? 100.0 * active_gt0_awk_ms / active_dt_ms : 0.0);
  fflush(out);
  if (out != stdout) fclose(out);
  return 0;
}
