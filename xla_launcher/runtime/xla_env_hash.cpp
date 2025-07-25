/*
 * @copyright
 * BSD 3-Clause License, 2025, He Jia <mofhejia@163.com>
 * BSD 3-Clause License, 2023, pytorch-tpu
 */

#include "xla_launcher/runtime/xla_env_hash.hpp"

#include <algorithm>
#include <string>
#include <unordered_set>
#include <vector>

#include "xla_launcher/runtime/sys_util.hpp"

namespace xla_launcher {
namespace runtime {
namespace hash {

namespace {

static constexpr const char* const XLA_FLAG_PREFIX = "--xla";

// Taken from JAX:
// https://github.com/google/jax/blob/8ee5811/jax/_src/cache_key.py#L325-L346
static const std::unordered_set<std::string> FlagsToExclude = {
  "--xla_dump_compress_protos",
  "--xla_dump_module_metadata",
  "--xla_dump_max_hlo_modules",
  "--xla_dump_include_timestamp",
  "--xla_dump_hlo_pass_re",
  "--xla_dump_hlo_module_re",
  "--xla_dump_hlo_snapshots",
  "--xla_dump_fusion_visualization",
  "--xla_dump_hlo_as_url",
  "--xla_dump_hlo_as_proto",
  "--xla_dump_hlo_as_text",
  "--xla_dump_hlo_as_long_text",
  "--xla_dump_hlo_as_html",
  "--xla_dump_hlo_as_dot",
  "--xla_dump_to",
  "--xla_force_host_platform_device_count",
  "--xla_dump_disable_metadata",
  "--xla_dump_hlo_pipeline_re",
  "--xla_tpu_sdc_checker_streamz_metric",
  "--xla_tpu_sdc_checker_enable_sdc_event_callbacks",
};

hash_t hash_xla_flags(std::string env_var_name) {
  std::stringstream xla_flag_env(
    sys_util::GetEnvString(env_var_name.c_str(), ""));
  std::string current_flag;
  std::vector<std::string> xla_flags;
  hash_t hash = hash_t::from_u64(0);
  // Parse the space-delimited flags once at a time.
  while (std::getline(xla_flag_env, current_flag, ' ')) {
    if (current_flag.rfind(XLA_FLAG_PREFIX, 0) != 0) {
      continue;
    }
    // XLA flags require key and value to be separated by '='.
    int eq_pos = current_flag.find('=');
    std::string flag_key;
    if (eq_pos == std::string::npos) {
      flag_key = current_flag;
    } else {
      flag_key = current_flag.substr(0, eq_pos);
    }
    if (FlagsToExclude.find(flag_key) != FlagsToExclude.end()) {
      continue;
    }
    xla_flags.push_back(current_flag);
  }
  // Ensure the flags are sorted so the input order doesn't impact the hash.
  std::sort(xla_flags.begin(), xla_flags.end());
  for (auto& flag : xla_flags) {
    hash =
      util::HashCombine(hash, hash_t::from_u64(util::StringHash(flag.c_str())));
  }
  return hash;
}

hash_t hash_xla_env(
  std::vector<std::string> flag_vars, std::vector<std::string> raw_vars) {
  hash_t hash;
  // Parse the flag_vars for XLA flags.
  for (auto& env_var_name : flag_vars) {
    hash = util::HashCombine(hash, hash_xla_flags(env_var_name));
  }

  // Include the raw flag value for raw_vars
  for (auto& env_var_name : raw_vars) {
    std::string raw_val = sys_util::GetEnvString(env_var_name.c_str(), "");
    hash = util::HashCombine(
      hash, hash_t::from_u64(util::StringHash(raw_val.c_str())));
  }
  return hash;
}
}  // namespace

hash_t HashXlaEnvVars() {
  // Both XLA_FLAGS and LIBTPU_INIT_ARGS contain XLA flags which impact
  // the compilation result.
  static std::vector<std::string> flag_vars = {"XLA_FLAGS", "LIBTPU_INIT_ARGS"};
  static std::vector<std::string> raw_vars = {
    "TPU_MEGACORE", "XLA_HLO_DEBUG", "XLA_IR_DEBUG"};
  return hash_xla_env(flag_vars, raw_vars);
}

}  // namespace hash
}  // namespace runtime
}  // namespace xla_launcher
