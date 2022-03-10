

#ifndef LPQ_L12_2D_CPP_
#define LPQ_L12_2D_CPP_

#include <algorithm>
#include <array>
#include <cmath>   // for abs()


// Abstract Struct for all L12 2D points function with accum_dist() and eval_pair()
template <class T, class DataSource, typename _DistanceType = T>
struct L12_M_2D {
  typedef T ElementType;
  typedef _DistanceType DistanceType;

  //int dist_exponent = 2; for M > 1
  int pair_exponent = 2;

  //general function, unused for specific Lpq
  void setup_lpq(int p, int q, int n_dim){}

  template <typename U, typename V>
  inline DistanceType accum_dist(const U a, const V b, const size_t) const {
    const DistanceType diff = a - b;
    return diff * diff;
  }

  inline DistanceType eval_pair(const T *a, const T *b, size_t size, DistanceType worst_dist) const {
    DistanceType result = DistanceType();
    const T *last = a + size;

    while (a < last) {
      const DistanceType diff0 = std::abs(a[0] - b[0]);
      const DistanceType diff1 = std::abs(a[1] - b[1]);
      const DistanceType psum_l1 = diff0 + diff1;
      result += psum_l1*psum_l1;
      a += 2;
      b += 2;

      if (result > worst_dist) {
        return result;
      }
    }
    return result;
  }

  inline DistanceType eval_pair(const T *a, const T *b, size_t size) const {
    DistanceType result = DistanceType();
    const T *last = a + size;

    while (a < last) {
      const DistanceType diff0 = std::abs(a[0] - b[0]);
      const DistanceType diff1 = std::abs(a[1] - b[1]);
      const DistanceType psum_l1 = diff0 + diff1;
      result += psum_l1*psum_l1;
      a += 2;
      b += 2;
    }
    return result;
  }
};

//General
template <class T, class DataSource, typename _DistanceType = T>
struct L12_M_2D_Adaptor : L12_M_2D<T, DataSource, _DistanceType> {
  typedef T ElementType;
  typedef _DistanceType DistanceType;

  int dist_exponent = 2;
  const DataSource &data_source;

  L12_M_2D_Adaptor(const DataSource &_data_source) : data_source(_data_source) {}

  inline DistanceType evalMetric(const T *a, const size_t b_idx, size_t size, DistanceType worst_dist) const {
    DistanceType result = DistanceType();
    const T *last = a + size;
    size_t d = 0;

    while (a < last) {
      const DistanceType diff0 = std::abs(a[0] - data_source.kdtree_get_pt(b_idx, d++));
      const DistanceType diff1 = std::abs(a[1] - data_source.kdtree_get_pt(b_idx, d++));
      const DistanceType psum_l1 = diff0 + diff1;
      result += psum_l1*psum_l1;
      a += 2;

      if (result > worst_dist) {
        return result;
      }
    }
    return result;
  }

  inline DistanceType evalMetric(const T *a, const size_t b_idx, size_t size) const {
    DistanceType result = DistanceType();
    const T *last = a + size;
    size_t d = 0;

    while (a < last) {
      const DistanceType diff0 = std::abs(a[0] - data_source.kdtree_get_pt(b_idx, d++));
      const DistanceType diff1 = std::abs(a[1] - data_source.kdtree_get_pt(b_idx, d++));
      const DistanceType psum_l1 = diff0 + diff1;
      result += psum_l1*psum_l1;
      a += 2;
    }
    return result;
  }
};

//1x2D Equivalent to L1
template <class T, class DataSource, typename _DistanceType = T>
struct L12_1_2D_Adaptor : L12_M_2D<T, DataSource, _DistanceType> {
  typedef T ElementType;
  typedef _DistanceType DistanceType;

  int dist_exponent = 1;
  const DataSource &data_source;

  L12_1_2D_Adaptor(const DataSource &_data_source) : data_source(_data_source) {}

  template <typename U, typename V>
  inline DistanceType accum_dist(const U a, const V b, const size_t) const{ //override
    // equivalent to L1
    return std::abs(a - b);
  }

  inline DistanceType evalMetric(const T *a, const size_t b_idx, size_t size, DistanceType worst_dist) const {
    // equivalent to L1_2D
    return (std::abs(a[0] - data_source.kdtree_get_pt(b_idx, 0))
            + std::abs(a[1] - data_source.kdtree_get_pt(b_idx, 1)));
  }

  inline DistanceType evalMetric(const T *a, const size_t b_idx, size_t size) const {
    // equivalent to L1_2D
    return (std::abs(a[0] - data_source.kdtree_get_pt(b_idx, 0))
            + std::abs(a[1] - data_source.kdtree_get_pt(b_idx, 1)));
  }
};

//2x2D
template <class T, class DataSource, typename _DistanceType = T>
struct L12_2_2D_Adaptor : L12_M_2D<T, DataSource, _DistanceType> {
  typedef T ElementType;
  typedef _DistanceType DistanceType;

  int dist_exponent = 2;
  const DataSource &data_source;

  L12_2_2D_Adaptor(const DataSource &_data_source) : data_source(_data_source) {}

  inline DistanceType evalMetric(const T *a, const size_t b_idx, size_t size, DistanceType worst_dist) const {
    const DistanceType psum0 = (std::abs(a[0] - data_source.kdtree_get_pt(b_idx, 0)) +
                                std::abs(a[1] - data_source.kdtree_get_pt(b_idx, 1)));
    const DistanceType psum1 = (std::abs(a[2] - data_source.kdtree_get_pt(b_idx, 2)) +
                                std::abs(a[3] - data_source.kdtree_get_pt(b_idx, 3)));

    return psum0*psum0 + psum1*psum1;
  }

  inline DistanceType evalMetric(const T *a, const size_t b_idx, size_t size) const {
    const DistanceType psum0 = (std::abs(a[0] - data_source.kdtree_get_pt(b_idx, 0)) +
                                std::abs(a[1] - data_source.kdtree_get_pt(b_idx, 1)));
    const DistanceType psum1 = (std::abs(a[2] - data_source.kdtree_get_pt(b_idx, 2)) +
                                std::abs(a[3] - data_source.kdtree_get_pt(b_idx, 3)));

    return psum0*psum0 + psum1*psum1;
  }
};


//3x2D
template <class T, class DataSource, typename _DistanceType = T>
struct L12_3_2D_Adaptor : L12_M_2D<T, DataSource, _DistanceType> {
  typedef T ElementType;
  typedef _DistanceType DistanceType;

  int dist_exponent = 2;
  const DataSource &data_source;

  L12_3_2D_Adaptor(const DataSource &_data_source) : data_source(_data_source) {}

  inline DistanceType evalMetric(const T *a, const size_t b_idx, size_t size, DistanceType worst_dist) const {
    const DistanceType psum0 = (std::abs(a[0] - data_source.kdtree_get_pt(b_idx, 0)) +
                                std::abs(a[1] - data_source.kdtree_get_pt(b_idx, 1)));
    const DistanceType psum1 = (std::abs(a[2] - data_source.kdtree_get_pt(b_idx, 2)) +
                                std::abs(a[3] - data_source.kdtree_get_pt(b_idx, 3)));
    const DistanceType psum2 = (std::abs(a[4] - data_source.kdtree_get_pt(b_idx, 4)) +
                                std::abs(a[5] - data_source.kdtree_get_pt(b_idx, 5)));

    return psum0*psum0 + psum1*psum1 + psum2*psum2;
  }

  inline DistanceType evalMetric(const T *a, const size_t b_idx, size_t size) const {
    const DistanceType psum0 = (std::abs(a[0] - data_source.kdtree_get_pt(b_idx, 0)) +
                                std::abs(a[1] - data_source.kdtree_get_pt(b_idx, 1)));
    const DistanceType psum1 = (std::abs(a[2] - data_source.kdtree_get_pt(b_idx, 2)) +
                                std::abs(a[3] - data_source.kdtree_get_pt(b_idx, 3)));
    const DistanceType psum2 = (std::abs(a[4] - data_source.kdtree_get_pt(b_idx, 4)) +
                                std::abs(a[5] - data_source.kdtree_get_pt(b_idx, 5)));

    return psum0*psum0 + psum1*psum1 + psum2*psum2;
  }
};


//4x2D
template <class T, class DataSource, typename _DistanceType = T>
struct L12_4_2D_Adaptor : L12_M_2D<T, DataSource, _DistanceType> {
  typedef T ElementType;
  typedef _DistanceType DistanceType;

  int dist_exponent = 2;
  const DataSource &data_source;

  L12_4_2D_Adaptor(const DataSource &_data_source) : data_source(_data_source) {}

  inline DistanceType evalMetric(const T *a, const size_t b_idx, size_t size, DistanceType worst_dist) const {
    const DistanceType psum0 = (std::abs(a[0] - data_source.kdtree_get_pt(b_idx, 0)) +
                                std::abs(a[1] - data_source.kdtree_get_pt(b_idx, 1)));
    const DistanceType psum1 = (std::abs(a[2] - data_source.kdtree_get_pt(b_idx, 2)) +
                                std::abs(a[3] - data_source.kdtree_get_pt(b_idx, 3)));
    const DistanceType psum2 = (std::abs(a[4] - data_source.kdtree_get_pt(b_idx, 4)) +
                                std::abs(a[5] - data_source.kdtree_get_pt(b_idx, 5)));
    const DistanceType psum3 = (std::abs(a[6] - data_source.kdtree_get_pt(b_idx, 6)) +
                                std::abs(a[7] - data_source.kdtree_get_pt(b_idx, 7)));

    return psum0*psum0 + psum1*psum1 + psum2*psum2 + psum3*psum3;
  }

  inline DistanceType evalMetric(const T *a, const size_t b_idx, size_t size) const {
    const DistanceType psum0 = (std::abs(a[0] - data_source.kdtree_get_pt(b_idx, 0)) +
                                std::abs(a[1] - data_source.kdtree_get_pt(b_idx, 1)));
    const DistanceType psum1 = (std::abs(a[2] - data_source.kdtree_get_pt(b_idx, 2)) +
                                std::abs(a[3] - data_source.kdtree_get_pt(b_idx, 3)));
    const DistanceType psum2 = (std::abs(a[4] - data_source.kdtree_get_pt(b_idx, 4)) +
                                std::abs(a[5] - data_source.kdtree_get_pt(b_idx, 5)));
    const DistanceType psum3 = (std::abs(a[6] - data_source.kdtree_get_pt(b_idx, 6)) +
                                std::abs(a[7] - data_source.kdtree_get_pt(b_idx, 7)));

    return psum0*psum0 + psum1*psum1 + psum2*psum2 + psum3*psum3;
  }
};

//5x2D
template <class T, class DataSource, typename _DistanceType = T>
struct L12_5_2D_Adaptor : L12_M_2D<T, DataSource, _DistanceType> {
  typedef T ElementType;
  typedef _DistanceType DistanceType;

  int dist_exponent = 2;
  const DataSource &data_source;

  L12_5_2D_Adaptor(const DataSource &_data_source) : data_source(_data_source) {}

  inline DistanceType evalMetric(const T *a, const size_t b_idx, size_t size, DistanceType worst_dist) const {
    const DistanceType psum0 = (std::abs(a[0] - data_source.kdtree_get_pt(b_idx, 0)) +
                                std::abs(a[1] - data_source.kdtree_get_pt(b_idx, 1)));
    const DistanceType psum1 = (std::abs(a[2] - data_source.kdtree_get_pt(b_idx, 2)) +
                                std::abs(a[3] - data_source.kdtree_get_pt(b_idx, 3)));
    const DistanceType psum2 = (std::abs(a[4] - data_source.kdtree_get_pt(b_idx, 4)) +
                                std::abs(a[5] - data_source.kdtree_get_pt(b_idx, 5)));
    const DistanceType psum3 = (std::abs(a[6] - data_source.kdtree_get_pt(b_idx, 6)) +
                                std::abs(a[7] - data_source.kdtree_get_pt(b_idx, 7)));
    const DistanceType psum4 = (std::abs(a[8] - data_source.kdtree_get_pt(b_idx, 8)) +
                                std::abs(a[9] - data_source.kdtree_get_pt(b_idx, 9)));

    return psum0*psum0 + psum1*psum1 + psum2*psum2 + psum3*psum3 + psum4*psum4;
  }

  inline DistanceType evalMetric(const T *a, const size_t b_idx, size_t size) const {
    const DistanceType psum0 = (std::abs(a[0] - data_source.kdtree_get_pt(b_idx, 0)) +
                                std::abs(a[1] - data_source.kdtree_get_pt(b_idx, 1)));
    const DistanceType psum1 = (std::abs(a[2] - data_source.kdtree_get_pt(b_idx, 2)) +
                                std::abs(a[3] - data_source.kdtree_get_pt(b_idx, 3)));
    const DistanceType psum2 = (std::abs(a[4] - data_source.kdtree_get_pt(b_idx, 4)) +
                                std::abs(a[5] - data_source.kdtree_get_pt(b_idx, 5)));
    const DistanceType psum3 = (std::abs(a[6] - data_source.kdtree_get_pt(b_idx, 6)) +
                                std::abs(a[7] - data_source.kdtree_get_pt(b_idx, 7)));
    const DistanceType psum4 = (std::abs(a[8] - data_source.kdtree_get_pt(b_idx, 8)) +
                                std::abs(a[9] - data_source.kdtree_get_pt(b_idx, 9)));

    return psum0*psum0 + psum1*psum1 + psum2*psum2 + psum3*psum3 + psum4*psum4;
  }
};

//6x2D
template <class T, class DataSource, typename _DistanceType = T>
struct L12_6_2D_Adaptor : L12_M_2D<T, DataSource, _DistanceType> {
  typedef T ElementType;
  typedef _DistanceType DistanceType;

  int dist_exponent = 2;
  const DataSource &data_source;

  L12_6_2D_Adaptor(const DataSource &_data_source) : data_source(_data_source) {}

  inline DistanceType evalMetric(const T *a, const size_t b_idx, size_t size, DistanceType worst_dist) const {
    const DistanceType psum0 = (std::abs(a[0] - data_source.kdtree_get_pt(b_idx, 0)) +
                                std::abs(a[1] - data_source.kdtree_get_pt(b_idx, 1)));
    const DistanceType psum1 = (std::abs(a[2] - data_source.kdtree_get_pt(b_idx, 2)) +
                                std::abs(a[3] - data_source.kdtree_get_pt(b_idx, 3)));
    const DistanceType psum2 = (std::abs(a[4] - data_source.kdtree_get_pt(b_idx, 4)) +
                                std::abs(a[5] - data_source.kdtree_get_pt(b_idx, 5)));
    const DistanceType psum3 = (std::abs(a[6] - data_source.kdtree_get_pt(b_idx, 6)) +
                                std::abs(a[7] - data_source.kdtree_get_pt(b_idx, 7)));
    const DistanceType psum4 = (std::abs(a[8] - data_source.kdtree_get_pt(b_idx, 8)) +
                                std::abs(a[9] - data_source.kdtree_get_pt(b_idx, 9)));
    const DistanceType psum5 = (std::abs(a[10] - data_source.kdtree_get_pt(b_idx, 10)) +
                                std::abs(a[11] - data_source.kdtree_get_pt(b_idx, 11)));

    return psum0*psum0 + psum1*psum1 + psum2*psum2 + psum3*psum3 + psum4*psum4 + psum5*psum5;
  }

  inline DistanceType evalMetric(const T *a, const size_t b_idx, size_t size) const {
    const DistanceType psum0 = (std::abs(a[0] - data_source.kdtree_get_pt(b_idx, 0)) +
                                std::abs(a[1] - data_source.kdtree_get_pt(b_idx, 1)));
    const DistanceType psum1 = (std::abs(a[2] - data_source.kdtree_get_pt(b_idx, 2)) +
                                std::abs(a[3] - data_source.kdtree_get_pt(b_idx, 3)));
    const DistanceType psum2 = (std::abs(a[4] - data_source.kdtree_get_pt(b_idx, 4)) +
                                std::abs(a[5] - data_source.kdtree_get_pt(b_idx, 5)));
    const DistanceType psum3 = (std::abs(a[6] - data_source.kdtree_get_pt(b_idx, 6)) +
                                std::abs(a[7] - data_source.kdtree_get_pt(b_idx, 7)));
    const DistanceType psum4 = (std::abs(a[8] - data_source.kdtree_get_pt(b_idx, 8)) +
                                std::abs(a[9] - data_source.kdtree_get_pt(b_idx, 9)));
    const DistanceType psum5 = (std::abs(a[10] - data_source.kdtree_get_pt(b_idx, 10)) +
                                std::abs(a[11] - data_source.kdtree_get_pt(b_idx, 11)));

    return psum0*psum0 + psum1*psum1 + psum2*psum2 + psum3*psum3 + psum4*psum4 + psum5*psum5;
  }
};

//7x2D
template <class T, class DataSource, typename _DistanceType = T>
struct L12_7_2D_Adaptor : L12_M_2D<T, DataSource, _DistanceType> {
  typedef T ElementType;
  typedef _DistanceType DistanceType;

  int dist_exponent = 2;
  const DataSource &data_source;

  L12_7_2D_Adaptor(const DataSource &_data_source) : data_source(_data_source) {}

  inline DistanceType evalMetric(const T *a, const size_t b_idx, size_t size, DistanceType worst_dist) const {
    const DistanceType psum0 = (std::abs(a[0] - data_source.kdtree_get_pt(b_idx, 0)) +
                                std::abs(a[1] - data_source.kdtree_get_pt(b_idx, 1)));
    const DistanceType psum1 = (std::abs(a[2] - data_source.kdtree_get_pt(b_idx, 2)) +
                                std::abs(a[3] - data_source.kdtree_get_pt(b_idx, 3)));
    const DistanceType psum2 = (std::abs(a[4] - data_source.kdtree_get_pt(b_idx, 4)) +
                                std::abs(a[5] - data_source.kdtree_get_pt(b_idx, 5)));
    const DistanceType psum3 = (std::abs(a[6] - data_source.kdtree_get_pt(b_idx, 6)) +
                                std::abs(a[7] - data_source.kdtree_get_pt(b_idx, 7)));
    const DistanceType psum4 = (std::abs(a[8] - data_source.kdtree_get_pt(b_idx, 8)) +
                                std::abs(a[9] - data_source.kdtree_get_pt(b_idx, 9)));
    const DistanceType psum5 = (std::abs(a[10] - data_source.kdtree_get_pt(b_idx, 10)) +
                                std::abs(a[11] - data_source.kdtree_get_pt(b_idx, 11)));
    const DistanceType psum6 = (std::abs(a[12] - data_source.kdtree_get_pt(b_idx, 12)) +
                                std::abs(a[13] - data_source.kdtree_get_pt(b_idx, 13)));

    return psum0*psum0 + psum1*psum1 + psum2*psum2 + psum3*psum3 + psum4*psum4 + psum5*psum5 + psum6*psum6;
  }

  inline DistanceType evalMetric(const T *a, const size_t b_idx, size_t size) const {
    const DistanceType psum0 = (std::abs(a[0] - data_source.kdtree_get_pt(b_idx, 0)) +
                                std::abs(a[1] - data_source.kdtree_get_pt(b_idx, 1)));
    const DistanceType psum1 = (std::abs(a[2] - data_source.kdtree_get_pt(b_idx, 2)) +
                                std::abs(a[3] - data_source.kdtree_get_pt(b_idx, 3)));
    const DistanceType psum2 = (std::abs(a[4] - data_source.kdtree_get_pt(b_idx, 4)) +
                                std::abs(a[5] - data_source.kdtree_get_pt(b_idx, 5)));
    const DistanceType psum3 = (std::abs(a[6] - data_source.kdtree_get_pt(b_idx, 6)) +
                                std::abs(a[7] - data_source.kdtree_get_pt(b_idx, 7)));
    const DistanceType psum4 = (std::abs(a[8] - data_source.kdtree_get_pt(b_idx, 8)) +
                                std::abs(a[9] - data_source.kdtree_get_pt(b_idx, 9)));
    const DistanceType psum5 = (std::abs(a[10] - data_source.kdtree_get_pt(b_idx, 10)) +
                                std::abs(a[11] - data_source.kdtree_get_pt(b_idx, 11)));
    const DistanceType psum6 = (std::abs(a[12] - data_source.kdtree_get_pt(b_idx, 12)) +
                                std::abs(a[13] - data_source.kdtree_get_pt(b_idx, 13)));

    return psum0*psum0 + psum1*psum1 + psum2*psum2 + psum3*psum3 + psum4*psum4 + psum5*psum5 + psum6*psum6;
  }
};

//8x2D
template <class T, class DataSource, typename _DistanceType = T>
struct L12_8_2D_Adaptor : L12_M_2D<T, DataSource, _DistanceType> {
  typedef T ElementType;
  typedef _DistanceType DistanceType;

  int dist_exponent = 2;
  const DataSource &data_source;

  L12_8_2D_Adaptor(const DataSource &_data_source) : data_source(_data_source) {}

  inline DistanceType evalMetric(const T *a, const size_t b_idx, size_t size, DistanceType worst_dist) const {
    const DistanceType psum0 = (std::abs(a[0] - data_source.kdtree_get_pt(b_idx, 0)) +
                                std::abs(a[1] - data_source.kdtree_get_pt(b_idx, 1)));
    const DistanceType psum1 = (std::abs(a[2] - data_source.kdtree_get_pt(b_idx, 2)) +
                                std::abs(a[3] - data_source.kdtree_get_pt(b_idx, 3)));
    const DistanceType psum2 = (std::abs(a[4] - data_source.kdtree_get_pt(b_idx, 4)) +
                                std::abs(a[5] - data_source.kdtree_get_pt(b_idx, 5)));
    const DistanceType psum3 = (std::abs(a[6] - data_source.kdtree_get_pt(b_idx, 6)) +
                                std::abs(a[7] - data_source.kdtree_get_pt(b_idx, 7)));
    const DistanceType psum4 = (std::abs(a[8] - data_source.kdtree_get_pt(b_idx, 8)) +
                                std::abs(a[9] - data_source.kdtree_get_pt(b_idx, 9)));
    const DistanceType psum5 = (std::abs(a[10] - data_source.kdtree_get_pt(b_idx, 10)) +
                                std::abs(a[11] - data_source.kdtree_get_pt(b_idx, 11)));
    const DistanceType psum6 = (std::abs(a[12] - data_source.kdtree_get_pt(b_idx, 12)) +
                                std::abs(a[13] - data_source.kdtree_get_pt(b_idx, 13)));
    const DistanceType psum7 = (std::abs(a[14] - data_source.kdtree_get_pt(b_idx, 14)) +
                                std::abs(a[15] - data_source.kdtree_get_pt(b_idx, 15)));

    return psum0*psum0 + psum1*psum1 + psum2*psum2 + psum3*psum3 + psum4*psum4 + psum5*psum5 + psum6*psum6 + psum7*psum7;
  }

  inline DistanceType evalMetric(const T *a, const size_t b_idx, size_t size) const {
    const DistanceType psum0 = (std::abs(a[0] - data_source.kdtree_get_pt(b_idx, 0)) +
                                std::abs(a[1] - data_source.kdtree_get_pt(b_idx, 1)));
    const DistanceType psum1 = (std::abs(a[2] - data_source.kdtree_get_pt(b_idx, 2)) +
                                std::abs(a[3] - data_source.kdtree_get_pt(b_idx, 3)));
    const DistanceType psum2 = (std::abs(a[4] - data_source.kdtree_get_pt(b_idx, 4)) +
                                std::abs(a[5] - data_source.kdtree_get_pt(b_idx, 5)));
    const DistanceType psum3 = (std::abs(a[6] - data_source.kdtree_get_pt(b_idx, 6)) +
                                std::abs(a[7] - data_source.kdtree_get_pt(b_idx, 7)));
    const DistanceType psum4 = (std::abs(a[8] - data_source.kdtree_get_pt(b_idx, 8)) +
                                std::abs(a[9] - data_source.kdtree_get_pt(b_idx, 9)));
    const DistanceType psum5 = (std::abs(a[10] - data_source.kdtree_get_pt(b_idx, 10)) +
                                std::abs(a[11] - data_source.kdtree_get_pt(b_idx, 11)));
    const DistanceType psum6 = (std::abs(a[12] - data_source.kdtree_get_pt(b_idx, 12)) +
                                std::abs(a[13] - data_source.kdtree_get_pt(b_idx, 13)));
    const DistanceType psum7 = (std::abs(a[14] - data_source.kdtree_get_pt(b_idx, 14)) +
                                std::abs(a[15] - data_source.kdtree_get_pt(b_idx, 15)));

    return psum0*psum0 + psum1*psum1 + psum2*psum2 + psum3*psum3 + psum4*psum4 + psum5*psum5 + psum6*psum6 + psum7*psum7;
  }
};


#endif /* LPQ_L12_2D_CPP_ */
