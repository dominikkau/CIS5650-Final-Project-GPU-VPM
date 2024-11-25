#ifndef FLOWVLM_DT_H
#define FLOWVLM_DT_H

#include <vector>
#include <variant>
#include <type_traits>

// Number types configuration
constexpr bool dev_flag = true;

using FWrap = std::conditional_t<dev_flag, double, std::variant<float, double, long double>>;
using IWrap = int64_t;

template <typename T = FWrap>
using FArrWrap = std::vector<T>;

template <typename T = FWrap>
using FMWrap = std::vector<std::vector<T>>;

using IArrWrap = std::vector<IWrap>;
using IMWrap = std::vector<std::vector<IWrap>>;

#endif // FLOWVLM_DT_H
