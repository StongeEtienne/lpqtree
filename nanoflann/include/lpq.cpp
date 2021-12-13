


#ifndef LPQ_CPP_
#define LPQ_CPP_


#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>   // for abs()
#include <cstdio>  // for fwrite()
//#include <cstdlib> // for abs()
#include <functional>
#include <limits> // std::reference_wrapper
#include <stdexcept>
#include <vector>

 /** the PI constant (required to avoid MSVC missing symbols) */
template <typename T> T pi_const() {
  return static_cast<T>(3.14159265358979323846);
}


 // template <typename T> T sqrtMinv() {
 //   return static_cast<T>(1.0) / std::sqrt(static_cast<T>(M_DIM));
 // }

/** Manhattan distance functor (generic version, optimized for
 * high-dimensionality data sets). Corresponding distance traits:
 * nanoflann::metric_L1 \tparam T Type of the elements (e.g. double, float,
 * uint8_t) \tparam _DistanceType Type of distance variables (must be signed)
 * (e.g. float, double, int64_t)
 */

// template <class T, class DataSource, typename _DistanceType = T>
// struct SO2_Adaptor {
//   typedef T ElementType;
//   typedef _DistanceType DistanceType;
//
//   const DataSource &data_source;
//
//   SO2_Adaptor(const DataSource &_data_source) : data_source(_data_source) {}
//
//   inline DistanceType evalMetric(const T *a, const size_t b_idx, size_t size) const {
//     return accum_dist(a[size - 1], data_source.kdtree_get_pt(b_idx, size - 1),
//                       size - 1);
//   }
//
//   /** Note: this assumes that input angles are already in the range [-pi,pi] */
//   template <typename U, typename V>
//   inline DistanceType accum_dist(const U a, const V b, const size_t) const {
//     DistanceType result = DistanceType();
//     const DistanceType PI = pi_const<DistanceType>();
//     result = b - a;
//     if (result > PI)
//       result -= 2 * PI;
//     else if (result < -PI)
//       result += 2 * PI;
//     return result;
//   }
// };
//
//
// template <class T, class DataSource, typename _DistanceType = T>
// struct SO3_Adaptor {
//   typedef T ElementType;
//   typedef _DistanceType DistanceType;
//
//   L2_Simple_Adaptor<T, DataSource> distance_L2_Simple;
//
//   SO3_Adaptor(const DataSource &_data_source)
//       : distance_L2_Simple(_data_source) {}
//
//   inline DistanceType evalMetric(const T *a, const size_t b_idx, size_t size) const {
//     return distance_L2_Simple.evalMetric(a, b_idx, size);
//   }
//
//   template <typename U, typename V>
//   inline DistanceType accum_dist(const U a, const V b, const size_t idx) const {
//     return distance_L2_Simple.accum_dist(a, b, idx);
//   }
// };

#endif /* LPQ_CPP_ */
