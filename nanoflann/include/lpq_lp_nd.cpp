
#ifndef LPQ_Lp_ND_CPP_
#define LPQ_Lp_ND_CPP_

#include <algorithm>
#include <array>
#include <cmath>   // for abs()


// General Lp ND Adaptor
template <class T, class DataSource, typename _DistanceType = T>
struct Lp_ND_Adaptor {
  typedef T ElementType;
  typedef _DistanceType DistanceType;

  int dist_exponent=0; // = P
  int pair_exponent=0; // = P

  const DataSource &data_source;

  Lp_ND_Adaptor(const DataSource &_data_source) : data_source(_data_source) {}

  // setup function to initialize values, required for a general Lpq distance
  void setup_lpq(int p, int q, int n_dim){
    dist_exponent = p;
    pair_exponent = p;
  }

  template <typename U, typename V>
  inline DistanceType accum_dist(const U a, const V b, const size_t) const {
    return std::pow(std::abs(a - b), dist_exponent);
  }

  inline DistanceType eval_pair(const T *a, const T *b, size_t size, DistanceType worst_dist) const {
    DistanceType result = DistanceType();
    const T *last = a + size;
    const T *lastgroup = last - 3;

    /* Process 4 items with each loop for efficiency. */
    while (a < lastgroup) {
      result += (std::pow(std::abs(a[0] - b[0]), dist_exponent)
                 + std::pow(std::abs(a[1] - b[1]), dist_exponent)
                 + std::pow(std::abs(a[2] - b[2]), dist_exponent)
                 + std::pow(std::abs(a[3] - b[3]), dist_exponent));
      a += 4;
      b += 4;
      if ((worst_dist > 0) && (result > worst_dist)) {
        return result;
      }
    }
    /* Process last 0-3 components.  Not needed for standard vector lengths. */
    while (a < last) {
      result += std::pow(std::abs(*a++ - *b++), dist_exponent);
    }
    return result;
  }

  inline DistanceType eval_pair(const T *a, const T *b, size_t size) const {
    DistanceType result = DistanceType();
    const T *last = a + size;
    const T *lastgroup = last - 3;

    /* Process 4 items with each loop for efficiency. */
    while (a < lastgroup) {
      result += (std::pow(std::abs(a[0] - b[0]), dist_exponent)
                 + std::pow(std::abs(a[1] - b[1]), dist_exponent)
                 + std::pow(std::abs(a[2] - b[2]), dist_exponent)
                 + std::pow(std::abs(a[3] - b[3]), dist_exponent));
      a += 4;
      b += 4;
    }
    /* Process last 0-3 components.  Not needed for standard vector lengths. */
    while (a < last) {
      result += std::pow(std::abs(*a++ - *b++), dist_exponent);
    }
    return result;
  }


  inline DistanceType evalMetric(const T *a, const size_t b_idx, size_t size,
                                 DistanceType worst_dist) const {
    DistanceType result = DistanceType();
    const T *last = a + size;
    const T *lastgroup = last - 3;
    size_t d = 0;

    /* Process 4 items with each loop for efficiency. */
    while (a < lastgroup) {
      const DistanceType diff0 = std::pow(std::abs(a[0] - data_source.kdtree_get_pt(b_idx, d++)), dist_exponent);
      const DistanceType diff1 = std::pow(std::abs(a[1] - data_source.kdtree_get_pt(b_idx, d++)), dist_exponent);
      const DistanceType diff2 = std::pow(std::abs(a[2] - data_source.kdtree_get_pt(b_idx, d++)), dist_exponent);
      const DistanceType diff3 = std::pow(std::abs(a[3] - data_source.kdtree_get_pt(b_idx, d++)), dist_exponent);
      result += diff0 + diff1 + diff2 + diff3;
      a += 4;
      if ((worst_dist > 0) && (result > worst_dist)) {
        return result;
      }
    }
    /* Process last 0-3 components.  Not needed for standard vector lengths. */
    while (a < last) {
      result += std::pow(std::abs(*a++ - data_source.kdtree_get_pt(b_idx, d++)), dist_exponent);
    }
    return result;
  }

  inline DistanceType evalMetric(const T *a, const size_t b_idx, size_t size) const {
    DistanceType result = DistanceType();
    const T *last = a + size;
    const T *lastgroup = last - 3;
    size_t d = 0;

    /* Process 4 items with each loop for efficiency. */
    while (a < lastgroup) {
      const DistanceType diff0 = std::pow(std::abs(a[0] - data_source.kdtree_get_pt(b_idx, d++)), dist_exponent);
      const DistanceType diff1 = std::pow(std::abs(a[1] - data_source.kdtree_get_pt(b_idx, d++)), dist_exponent);
      const DistanceType diff2 = std::pow(std::abs(a[2] - data_source.kdtree_get_pt(b_idx, d++)), dist_exponent);
      const DistanceType diff3 = std::pow(std::abs(a[3] - data_source.kdtree_get_pt(b_idx, d++)), dist_exponent);
      result += diff0 + diff1 + diff2 + diff3;
      a += 4;
    }
    /* Process last 0-3 components.  Not needed for standard vector lengths. */
    while (a < last) {
      result += std::pow(std::abs(*a++ - data_source.kdtree_get_pt(b_idx, d++)), dist_exponent);
    }
    return result;
  }

};

#endif /* LPQ_Lp_ND_CPP_ */
