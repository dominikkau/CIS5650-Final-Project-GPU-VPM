// ------------ DATA TYPES ------------------------------------------------------
// Number types: Bugs can be easily caught during development by specifying all
// input types; however, in order to get automatic gradients, types must be
// dynamic. The following flag turns number inputs into hard types (floats or
// ints) if true, or into dynamic types if false.

#include <vector>
#include <variant>

constexpr bool dev_flag = true;

// Define type wrappers based on the development flag
using FWrap = std::conditional<dev_flag, double, std::variant<double, int>>::type;
using IWrap = std::conditional<dev_flag, int, std::variant<double, int>>::type;

using FArrWrap = std::vector<FWrap>;
using IArrWrap = std::vector<IWrap>;
using FMWrap = std::vector<std::vector<FWrap>>;
using IMWrap = std::vector<std::vector<IWrap>>;