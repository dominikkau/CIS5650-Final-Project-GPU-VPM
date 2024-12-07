#pragma once

#include <vector>
#include <variant>
#include <type_traits>

// Development flag: True for static types, false for dynamic types
constexpr bool dev_flag = true;

// Number type wrappers
using FWrap = std::conditional_t<dev_flag, double, std::variant<float, double, long double>>;
using IWrap = int64_t;

// Array wrappers
template <typename T = FWrap>
using FArrWrap = std::vector<T>; // 1D float array wrapper

template <typename T = FWrap>
using FMWrap = std::vector<std::vector<T>>; // 2D float matrix wrapper

using IArrWrap = std::vector<IWrap>; // 1D int array wrapper
using IMWrap = std::vector<std::vector<IWrap>>; // 2D int matrix wrapper
