#ifndef LPQ_L1_ND_CPP_
#define LPQ_L1_ND_CPP_


#include <algorithm>
#include <array>
#include <cmath>   // for abs()

// Abstract Struct for all L1 function with accum_dist() and eval_pair()
template <class T, class DataSource, typename _DistanceType = T>
struct L1_ND {
  typedef T ElementType;
  typedef _DistanceType DistanceType;

  int dist_exponent = 1;
  int pair_exponent = 1;

  //general function, unused for specific Lpq
  void setup_lpq(int p, int q, int n_dim){}

  template <typename U, typename V>
  inline DistanceType accum_dist(const U a, const V b, const size_t) const {
    return std::abs(a - b);
  }

  inline DistanceType eval_pair(const T *a, const T *b, size_t size, DistanceType worst_dist) const {
    DistanceType result = DistanceType();
    const T *last = a + size;
    const T *lastgroup = last - 3;

    /* Process 4 items with each loop for efficiency. */
    while (a < lastgroup) {
      const DistanceType diff0 = std::abs(a[0] - b[0]);
      const DistanceType diff1 = std::abs(a[1] - b[1]);
      const DistanceType diff2 = std::abs(a[2] - b[2]);
      const DistanceType diff3 = std::abs(a[3] - b[3]);
      result += diff0 + diff1 + diff2 + diff3;
      a += 4;
      b += 4;
      if ((worst_dist > 0) && (result > worst_dist)) {
        return result;
      }
    }
    /* Process last 0-3 components.  Not needed for standard vector lengths. */
    while (a < last) {
      result += std::abs(*a++ - *b++);
    }
    return result;
  }

  inline DistanceType eval_pair(const T *a, const T *b, size_t size) const {
    DistanceType result = DistanceType();
    const T *last = a + size;
    const T *lastgroup = last - 3;
    /* Process 4 items with each loop for efficiency. */
    while (a < lastgroup) {
      const DistanceType diff0 = std::abs(a[0] - b[0]);
      const DistanceType diff1 = std::abs(a[1] - b[1]);
      const DistanceType diff2 = std::abs(a[2] - b[2]);
      const DistanceType diff3 = std::abs(a[3] - b[3]);
      result += diff0 + diff1 + diff2 + diff3;
      a += 4;
      b += 4;
    }
    /* Process last 0-3 components.  Not needed for standard vector lengths. */
    while (a < last) {
      result += std::abs(*a++ - *b++);
    }
    return result;
  }

};


// General
template <class T, class DataSource, typename _DistanceType = T>
struct L1_ND_Adaptor : L1_ND<T, DataSource, _DistanceType> {
  typedef T ElementType;
  typedef _DistanceType DistanceType;

  const DataSource &data_source;

  L1_ND_Adaptor(const DataSource &_data_source) : data_source(_data_source) {}

  inline DistanceType evalMetric(const T *a, const size_t b_idx, size_t size, DistanceType worst_dist) const {
    DistanceType result = DistanceType();
    const T *last = a + size;
    const T *lastgroup = last - 3;
    size_t d = 0;

    /* Process 4 items with each loop for efficiency. */
    while (a < lastgroup) {
      const DistanceType diff0 = std::abs(a[0] - data_source.kdtree_get_pt(b_idx, d++));
      const DistanceType diff1 = std::abs(a[1] - data_source.kdtree_get_pt(b_idx, d++));
      const DistanceType diff2 = std::abs(a[2] - data_source.kdtree_get_pt(b_idx, d++));
      const DistanceType diff3 = std::abs(a[3] - data_source.kdtree_get_pt(b_idx, d++));
      result += diff0 + diff1 + diff2 + diff3;
      a += 4;
      if ((worst_dist > 0) && (result > worst_dist)) {
        return result;
      }
    }
    /* Process last 0-3 components.  Not needed for standard vector lengths. */
    while (a < last) {
      result += std::abs(*a++ - data_source.kdtree_get_pt(b_idx, d++));
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
      const DistanceType diff0 = std::abs(a[0] - data_source.kdtree_get_pt(b_idx, d++));
      const DistanceType diff1 = std::abs(a[1] - data_source.kdtree_get_pt(b_idx, d++));
      const DistanceType diff2 = std::abs(a[2] - data_source.kdtree_get_pt(b_idx, d++));
      const DistanceType diff3 = std::abs(a[3] - data_source.kdtree_get_pt(b_idx, d++));
      result += diff0 + diff1 + diff2 + diff3;
      a += 4;
    }
    /* Process last 0-3 components.  Not needed for standard vector lengths. */
    while (a < last) {
      result += std::abs(*a++ - data_source.kdtree_get_pt(b_idx, d++));
    }
    return result;
  }
};


// 1D to 8D version
template <class T, class DataSource, typename _DistanceType = T>
struct L1_1D_Adaptor : L1_ND<T, DataSource, _DistanceType> {
  typedef T ElementType;
  typedef _DistanceType DistanceType;

  const DataSource &data_source;

  L1_1D_Adaptor(const DataSource &_data_source) : data_source(_data_source) {}

  inline DistanceType evalMetric(const T *a, const size_t b_idx, size_t size, DistanceType worst_dist) const {
    return std::abs(a[0] - data_source.kdtree_get_pt(b_idx, 0));
  }

  inline DistanceType evalMetric(const T *a, const size_t b_idx, size_t size) const {
    return std::abs(a[0] - data_source.kdtree_get_pt(b_idx, 0));
  }
};


template <class T, class DataSource, typename _DistanceType = T>
struct L1_2D_Adaptor : L1_ND<T, DataSource, _DistanceType> {
  typedef T ElementType;
  typedef _DistanceType DistanceType;

  const DataSource &data_source;

  L1_2D_Adaptor(const DataSource &_data_source) : data_source(_data_source) {}

  inline DistanceType evalMetric(const T *a, const size_t b_idx, size_t size, DistanceType worst_dist) const {
    return (std::abs(a[0] - data_source.kdtree_get_pt(b_idx, 0))
            + std::abs(a[1] - data_source.kdtree_get_pt(b_idx, 1)));
  }

  inline DistanceType evalMetric(const T *a, const size_t b_idx, size_t size) const {
    return (std::abs(a[0] - data_source.kdtree_get_pt(b_idx, 0))
            + std::abs(a[1] - data_source.kdtree_get_pt(b_idx, 1)));
  }
};


template <class T, class DataSource, typename _DistanceType = T>
struct L1_3D_Adaptor : L1_ND<T, DataSource, _DistanceType> {
  typedef T ElementType;
  typedef _DistanceType DistanceType;

  const DataSource &data_source;

  L1_3D_Adaptor(const DataSource &_data_source) : data_source(_data_source) {}

  inline DistanceType evalMetric(const T *a, const size_t b_idx, size_t size, DistanceType worst_dist) const {
    return (std::abs(a[0] - data_source.kdtree_get_pt(b_idx, 0))
            + std::abs(a[1] - data_source.kdtree_get_pt(b_idx, 1))
            + std::abs(a[2] - data_source.kdtree_get_pt(b_idx, 2)));
  }

  inline DistanceType evalMetric(const T *a, const size_t b_idx, size_t size) const {
    return (std::abs(a[0] - data_source.kdtree_get_pt(b_idx, 0))
            + std::abs(a[1] - data_source.kdtree_get_pt(b_idx, 1))
            + std::abs(a[2] - data_source.kdtree_get_pt(b_idx, 2)));
  }
};


template <class T, class DataSource, typename _DistanceType = T>
struct L1_4D_Adaptor : L1_ND<T, DataSource, _DistanceType> {
  typedef T ElementType;
  typedef _DistanceType DistanceType;

  const DataSource &data_source;

  L1_4D_Adaptor(const DataSource &_data_source) : data_source(_data_source) {}

  inline DistanceType evalMetric(const T *a, const size_t b_idx, size_t size, DistanceType worst_dist) const {
    return (std::abs(a[0] - data_source.kdtree_get_pt(b_idx, 0))
            + std::abs(a[1] - data_source.kdtree_get_pt(b_idx, 1))
            + std::abs(a[2] - data_source.kdtree_get_pt(b_idx, 2))
            + std::abs(a[3] - data_source.kdtree_get_pt(b_idx, 3)));
  }

  inline DistanceType evalMetric(const T *a, const size_t b_idx, size_t size) const {
    return (std::abs(a[0] - data_source.kdtree_get_pt(b_idx, 0))
            + std::abs(a[1] - data_source.kdtree_get_pt(b_idx, 1))
            + std::abs(a[2] - data_source.kdtree_get_pt(b_idx, 2))
            + std::abs(a[3] - data_source.kdtree_get_pt(b_idx, 3)));
  }
};


template <class T, class DataSource, typename _DistanceType = T>
struct L1_5D_Adaptor : L1_ND<T, DataSource, _DistanceType> {
  typedef T ElementType;
  typedef _DistanceType DistanceType;

  const DataSource &data_source;

  L1_5D_Adaptor(const DataSource &_data_source) : data_source(_data_source) {}

  inline DistanceType evalMetric(const T *a, const size_t b_idx, size_t size, DistanceType worst_dist) const {
    return (std::abs(a[0] - data_source.kdtree_get_pt(b_idx, 0))
            + std::abs(a[1] - data_source.kdtree_get_pt(b_idx, 1))
            + std::abs(a[2] - data_source.kdtree_get_pt(b_idx, 2))
            + std::abs(a[3] - data_source.kdtree_get_pt(b_idx, 3))
            + std::abs(a[4] - data_source.kdtree_get_pt(b_idx, 4)));
  }

  inline DistanceType evalMetric(const T *a, const size_t b_idx, size_t size) const {
    return (std::abs(a[0] - data_source.kdtree_get_pt(b_idx, 0))
            + std::abs(a[1] - data_source.kdtree_get_pt(b_idx, 1))
            + std::abs(a[2] - data_source.kdtree_get_pt(b_idx, 2))
            + std::abs(a[3] - data_source.kdtree_get_pt(b_idx, 3))
            + std::abs(a[4] - data_source.kdtree_get_pt(b_idx, 4)));
  }
};


template <class T, class DataSource, typename _DistanceType = T>
struct L1_6D_Adaptor : L1_ND<T, DataSource, _DistanceType> {
  typedef T ElementType;
  typedef _DistanceType DistanceType;

  const DataSource &data_source;

  L1_6D_Adaptor(const DataSource &_data_source) : data_source(_data_source) {}

  inline DistanceType evalMetric(const T *a, const size_t b_idx, size_t size, DistanceType worst_dist) const {
    return (std::abs(a[0] - data_source.kdtree_get_pt(b_idx, 0))
            + std::abs(a[1] - data_source.kdtree_get_pt(b_idx, 1))
            + std::abs(a[2] - data_source.kdtree_get_pt(b_idx, 2))
            + std::abs(a[3] - data_source.kdtree_get_pt(b_idx, 3))
            + std::abs(a[4] - data_source.kdtree_get_pt(b_idx, 4))
            + std::abs(a[5] - data_source.kdtree_get_pt(b_idx, 5)));
  }

  inline DistanceType evalMetric(const T *a, const size_t b_idx, size_t size) const {
    return (std::abs(a[0] - data_source.kdtree_get_pt(b_idx, 0))
            + std::abs(a[1] - data_source.kdtree_get_pt(b_idx, 1))
            + std::abs(a[2] - data_source.kdtree_get_pt(b_idx, 2))
            + std::abs(a[3] - data_source.kdtree_get_pt(b_idx, 3))
            + std::abs(a[4] - data_source.kdtree_get_pt(b_idx, 4))
            + std::abs(a[5] - data_source.kdtree_get_pt(b_idx, 5)));
  }
};


template <class T, class DataSource, typename _DistanceType = T>
struct L1_7D_Adaptor : L1_ND<T, DataSource, _DistanceType> {
  typedef T ElementType;
  typedef _DistanceType DistanceType;

  const DataSource &data_source;

  L1_7D_Adaptor(const DataSource &_data_source) : data_source(_data_source) {}

  inline DistanceType evalMetric(const T *a, const size_t b_idx, size_t size, DistanceType worst_dist) const {
    return (std::abs(a[0] - data_source.kdtree_get_pt(b_idx, 0))
            + std::abs(a[1] - data_source.kdtree_get_pt(b_idx, 1))
            + std::abs(a[2] - data_source.kdtree_get_pt(b_idx, 2))
            + std::abs(a[3] - data_source.kdtree_get_pt(b_idx, 3))
            + std::abs(a[4] - data_source.kdtree_get_pt(b_idx, 4))
            + std::abs(a[5] - data_source.kdtree_get_pt(b_idx, 5))
            + std::abs(a[6] - data_source.kdtree_get_pt(b_idx, 6)));
  }

  inline DistanceType evalMetric(const T *a, const size_t b_idx, size_t size) const {
    return (std::abs(a[0] - data_source.kdtree_get_pt(b_idx, 0))
            + std::abs(a[1] - data_source.kdtree_get_pt(b_idx, 1))
            + std::abs(a[2] - data_source.kdtree_get_pt(b_idx, 2))
            + std::abs(a[3] - data_source.kdtree_get_pt(b_idx, 3))
            + std::abs(a[4] - data_source.kdtree_get_pt(b_idx, 4))
            + std::abs(a[5] - data_source.kdtree_get_pt(b_idx, 5))
            + std::abs(a[6] - data_source.kdtree_get_pt(b_idx, 6)));
  }
};


template <class T, class DataSource, typename _DistanceType = T>
struct L1_8D_Adaptor : L1_ND<T, DataSource, _DistanceType> {
  typedef T ElementType;
  typedef _DistanceType DistanceType;

  const DataSource &data_source;

  L1_8D_Adaptor(const DataSource &_data_source) : data_source(_data_source) {}

  inline DistanceType evalMetric(const T *a, const size_t b_idx, size_t size, DistanceType worst_dist) const {
    return (std::abs(a[0] - data_source.kdtree_get_pt(b_idx, 0))
            + std::abs(a[1] - data_source.kdtree_get_pt(b_idx, 1))
            + std::abs(a[2] - data_source.kdtree_get_pt(b_idx, 2))
            + std::abs(a[3] - data_source.kdtree_get_pt(b_idx, 3))
            + std::abs(a[4] - data_source.kdtree_get_pt(b_idx, 4))
            + std::abs(a[5] - data_source.kdtree_get_pt(b_idx, 5))
            + std::abs(a[6] - data_source.kdtree_get_pt(b_idx, 6))
            + std::abs(a[7] - data_source.kdtree_get_pt(b_idx, 7)));
  }

  inline DistanceType evalMetric(const T *a, const size_t b_idx, size_t size) const {
    return (std::abs(a[0] - data_source.kdtree_get_pt(b_idx, 0))
            + std::abs(a[1] - data_source.kdtree_get_pt(b_idx, 1))
            + std::abs(a[2] - data_source.kdtree_get_pt(b_idx, 2))
            + std::abs(a[3] - data_source.kdtree_get_pt(b_idx, 3))
            + std::abs(a[4] - data_source.kdtree_get_pt(b_idx, 4))
            + std::abs(a[5] - data_source.kdtree_get_pt(b_idx, 5))
            + std::abs(a[6] - data_source.kdtree_get_pt(b_idx, 6))
            + std::abs(a[7] - data_source.kdtree_get_pt(b_idx, 7)));
  }
};


#endif /* LPQ_L1_ND_CPP_ */
