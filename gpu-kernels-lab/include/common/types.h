#pragma once

#include <cstdint>
#include <string>

namespace gkl {

enum class DType { kF16, kF32 };

inline std::string dtype_name(DType dtype) {
  switch (dtype) {
    case DType::kF16:
      return "fp16";
    case DType::kF32:
      return "fp32";
  }
  return "unknown";
}

inline size_t dtype_size(DType dtype) {
  switch (dtype) {
    case DType::kF16:
      return 2;
    case DType::kF32:
      return 4;
  }
  return 0;
}

} // namespace gkl
