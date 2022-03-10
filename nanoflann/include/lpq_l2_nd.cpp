
#ifndef LPQ_L2_ND_CPP_
#define LPQ_L2_ND_CPP_

#include <algorithm>
#include <array>
#include <cmath>   // for abs()


// Abstract Struct for all L2 function with accum_dist() and eval_pair()
template <class T, class DataSource, typename _DistanceType = T>
struct L2_ND {
  typedef T ElementType;
  typedef _DistanceType DistanceType;

  int dist_exponent = 2;
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
    const T *lastgroup = last - 3;

    /* Process 4 items with each loop for efficiency. */
    while (a < lastgroup) {
      const DistanceType diff0 = a[0] - b[0];
      const DistanceType diff1 = a[1] - b[1];
      const DistanceType diff2 = a[2] - b[2];
      const DistanceType diff3 = a[3] - b[3];
      result += diff0 * diff0 + diff1 * diff1 + diff2 * diff2 + diff3 * diff3;
      a += 4;
      b += 4;
      if ((worst_dist > 0) && (result > worst_dist)) {
        return result;
      }
    }
    /* Process last 0-3 components.  Not needed for standard vector lengths. */
    while (a < last) {
      const DistanceType diff0 = *a++ - *b++;
      result += diff0 * diff0;
    }
    return result;
  }

  inline DistanceType eval_pair(const T *a, const T *b, size_t size) const {
    DistanceType result = DistanceType();
    const T *last = a + size;
    const T *lastgroup = last - 3;

    /* Process 4 items with each loop for efficiency. */
    while (a < lastgroup) {
      const DistanceType diff0 = a[0] - b[0];
      const DistanceType diff1 = a[1] - b[1];
      const DistanceType diff2 = a[2] - b[2];
      const DistanceType diff3 = a[3] - b[3];
      result += diff0 * diff0 + diff1 * diff1 + diff2 * diff2 + diff3 * diff3;
      a += 4;
      b += 4;
    }
    /* Process last 0-3 components.  Not needed for standard vector lengths. */
    while (a < last) {
      const DistanceType diff0 = *a++ - *b++;
      result += diff0 * diff0;
    }
    return result;
  }
};


//General ND
template <class T, class DataSource, typename _DistanceType = T>
struct L2_ND_Adaptor : L2_ND<T, DataSource, _DistanceType> {
  typedef T ElementType;
  typedef _DistanceType DistanceType;

  const DataSource &data_source;

  L2_ND_Adaptor(const DataSource &_data_source) : data_source(_data_source) {}

  inline DistanceType evalMetric(const T *a, const size_t b_idx, size_t size,
                                 DistanceType worst_dist) const {
    DistanceType result = DistanceType();
    const T *last = a + size;
    const T *lastgroup = last - 3;
    size_t d = 0;

    /* Process 4 items with each loop for efficiency. */
    while (a < lastgroup) {
      const DistanceType diff0 = a[0] - data_source.kdtree_get_pt(b_idx, d++);
      const DistanceType diff1 = a[1] - data_source.kdtree_get_pt(b_idx, d++);
      const DistanceType diff2 = a[2] - data_source.kdtree_get_pt(b_idx, d++);
      const DistanceType diff3 = a[3] - data_source.kdtree_get_pt(b_idx, d++);
      result += diff0 * diff0 + diff1 * diff1 + diff2 * diff2 + diff3 * diff3;
      a += 4;
      if ((worst_dist > 0) && (result > worst_dist)) {
        return result;
      }
    }
    /* Process last 0-3 components.  Not needed for standard vector lengths. */
    while (a < last) {
      const DistanceType diff0 = *a++ - data_source.kdtree_get_pt(b_idx, d++);
      result += diff0 * diff0;
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
      const DistanceType diff0 = a[0] - data_source.kdtree_get_pt(b_idx, d++);
      const DistanceType diff1 = a[1] - data_source.kdtree_get_pt(b_idx, d++);
      const DistanceType diff2 = a[2] - data_source.kdtree_get_pt(b_idx, d++);
      const DistanceType diff3 = a[3] - data_source.kdtree_get_pt(b_idx, d++);
      result += diff0 * diff0 + diff1 * diff1 + diff2 * diff2 + diff3 * diff3;
      a += 4;
    }
    /* Process last 0-3 components.  Not needed for standard vector lengths. */
    while (a < last) {
      const DistanceType diff0 = *a++ - data_source.kdtree_get_pt(b_idx, d++);
      result += diff0 * diff0;
    }
    return result;
  }

};




template <class T, class DataSource, typename _DistanceType = T>
struct L2_1D_Adaptor : L2_ND<T, DataSource, _DistanceType> {
  typedef T ElementType;
  typedef _DistanceType DistanceType;

  const DataSource &data_source;

  L2_1D_Adaptor(const DataSource &_data_source) : data_source(_data_source) {}

  inline DistanceType evalMetric(const T *a, const size_t b_idx, size_t size, DistanceType worst_dist) const {
    const DistanceType diff0 = a[0] - data_source.kdtree_get_pt(b_idx, 0);
    return diff0*diff0;
  }

  inline DistanceType evalMetric(const T *a, const size_t b_idx, size_t size) const {
    const DistanceType diff0 = a[0] - data_source.kdtree_get_pt(b_idx, 0);
    return diff0*diff0;
  }

};


template <class T, class DataSource, typename _DistanceType = T>
struct L2_2D_Adaptor : L2_ND<T, DataSource, _DistanceType> {
  typedef T ElementType;
  typedef _DistanceType DistanceType;

  const DataSource &data_source;

  L2_2D_Adaptor(const DataSource &_data_source) : data_source(_data_source) {}

  inline DistanceType evalMetric(const T *a, const size_t b_idx, size_t size, DistanceType worst_dist) const {
    const DistanceType diff0 = a[0] - data_source.kdtree_get_pt(b_idx, 0);
    const DistanceType diff1 = a[1] - data_source.kdtree_get_pt(b_idx, 1);
    return (diff0*diff0 + diff1*diff1);
  }

  inline DistanceType evalMetric(const T *a, const size_t b_idx, size_t size) const {
    const DistanceType diff0 = a[0] - data_source.kdtree_get_pt(b_idx, 0);
    const DistanceType diff1 = a[1] - data_source.kdtree_get_pt(b_idx, 1);
    return (diff0*diff0 + diff1*diff1);
  }

};

template <class T, class DataSource, typename _DistanceType = T>
struct L2_3D_Adaptor : L2_ND<T, DataSource, _DistanceType> {
  typedef T ElementType;
  typedef _DistanceType DistanceType;

  const DataSource &data_source;

  L2_3D_Adaptor(const DataSource &_data_source) : data_source(_data_source) {}

  inline DistanceType evalMetric(const T *a, const size_t b_idx, size_t size, DistanceType worst_dist) const {
    const DistanceType diff0 = a[0] - data_source.kdtree_get_pt(b_idx, 0);
    const DistanceType diff1 = a[1] - data_source.kdtree_get_pt(b_idx, 1);
    const DistanceType diff2 = a[2] - data_source.kdtree_get_pt(b_idx, 2);
    return (diff0*diff0 + diff1*diff1 + diff2*diff2);
  }

  inline DistanceType evalMetric(const T *a, const size_t b_idx, size_t size) const {
    const DistanceType diff0 = a[0] - data_source.kdtree_get_pt(b_idx, 0);
    const DistanceType diff1 = a[1] - data_source.kdtree_get_pt(b_idx, 1);
    const DistanceType diff2 = a[2] - data_source.kdtree_get_pt(b_idx, 2);
    return (diff0*diff0 + diff1*diff1 + diff2*diff2);
  }

};


template <class T, class DataSource, typename _DistanceType = T>
struct L2_4D_Adaptor : L2_ND<T, DataSource, _DistanceType> {
  typedef T ElementType;
  typedef _DistanceType DistanceType;

  const DataSource &data_source;

  L2_4D_Adaptor(const DataSource &_data_source) : data_source(_data_source) {}

  inline DistanceType evalMetric(const T *a, const size_t b_idx, size_t size, DistanceType worst_dist) const {
    const DistanceType diff0 = a[0] - data_source.kdtree_get_pt(b_idx, 0);
    const DistanceType diff1 = a[1] - data_source.kdtree_get_pt(b_idx, 1);
    const DistanceType diff2 = a[2] - data_source.kdtree_get_pt(b_idx, 2);
    const DistanceType diff3 = a[3] - data_source.kdtree_get_pt(b_idx, 3);
    return (diff0*diff0 + diff1*diff1 + diff2*diff2 + diff3*diff3);
  }

  inline DistanceType evalMetric(const T *a, const size_t b_idx, size_t size) const {
    const DistanceType diff0 = a[0] - data_source.kdtree_get_pt(b_idx, 0);
    const DistanceType diff1 = a[1] - data_source.kdtree_get_pt(b_idx, 1);
    const DistanceType diff2 = a[2] - data_source.kdtree_get_pt(b_idx, 2);
    const DistanceType diff3 = a[3] - data_source.kdtree_get_pt(b_idx, 3);
    return (diff0*diff0 + diff1*diff1 + diff2*diff2 + diff3*diff3);
  }

};

template <class T, class DataSource, typename _DistanceType = T>
struct L2_5D_Adaptor : L2_ND<T, DataSource, _DistanceType> {
  typedef T ElementType;
  typedef _DistanceType DistanceType;

  const DataSource &data_source;

  L2_5D_Adaptor(const DataSource &_data_source) : data_source(_data_source) {}

  inline DistanceType evalMetric(const T *a, const size_t b_idx, size_t size, DistanceType worst_dist) const {
    const DistanceType diff0 = a[0] - data_source.kdtree_get_pt(b_idx, 0);
    const DistanceType diff1 = a[1] - data_source.kdtree_get_pt(b_idx, 1);
    const DistanceType diff2 = a[2] - data_source.kdtree_get_pt(b_idx, 2);
    const DistanceType diff3 = a[3] - data_source.kdtree_get_pt(b_idx, 3);
    const DistanceType diff4 = a[4] - data_source.kdtree_get_pt(b_idx, 4);
    return (diff0*diff0 + diff1*diff1 + diff2*diff2 + diff3*diff3
            + diff4*diff4);
  }

  inline DistanceType evalMetric(const T *a, const size_t b_idx, size_t size) const {
    const DistanceType diff0 = a[0] - data_source.kdtree_get_pt(b_idx, 0);
    const DistanceType diff1 = a[1] - data_source.kdtree_get_pt(b_idx, 1);
    const DistanceType diff2 = a[2] - data_source.kdtree_get_pt(b_idx, 2);
    const DistanceType diff3 = a[3] - data_source.kdtree_get_pt(b_idx, 3);
    const DistanceType diff4 = a[4] - data_source.kdtree_get_pt(b_idx, 4);
    return (diff0*diff0 + diff1*diff1 + diff2*diff2 + diff3*diff3
            + diff4*diff4);
  }

};

template <class T, class DataSource, typename _DistanceType = T>
struct L2_6D_Adaptor : L2_ND<T, DataSource, _DistanceType> {
  typedef T ElementType;
  typedef _DistanceType DistanceType;

  const DataSource &data_source;

  L2_6D_Adaptor(const DataSource &_data_source) : data_source(_data_source) {}

  inline DistanceType evalMetric(const T *a, const size_t b_idx, size_t size, DistanceType worst_dist) const {
    const DistanceType diff0 = a[0] - data_source.kdtree_get_pt(b_idx, 0);
    const DistanceType diff1 = a[1] - data_source.kdtree_get_pt(b_idx, 1);
    const DistanceType diff2 = a[2] - data_source.kdtree_get_pt(b_idx, 2);
    const DistanceType diff3 = a[3] - data_source.kdtree_get_pt(b_idx, 3);
    const DistanceType diff4 = a[4] - data_source.kdtree_get_pt(b_idx, 4);
    const DistanceType diff5 = a[5] - data_source.kdtree_get_pt(b_idx, 5);
    return (diff0*diff0 + diff1*diff1 + diff2*diff2 + diff3*diff3
            + diff4*diff4 + diff5*diff5);
  }

  inline DistanceType evalMetric(const T *a, const size_t b_idx, size_t size) const {
    const DistanceType diff0 = a[0] - data_source.kdtree_get_pt(b_idx, 0);
    const DistanceType diff1 = a[1] - data_source.kdtree_get_pt(b_idx, 1);
    const DistanceType diff2 = a[2] - data_source.kdtree_get_pt(b_idx, 2);
    const DistanceType diff3 = a[3] - data_source.kdtree_get_pt(b_idx, 3);
    const DistanceType diff4 = a[4] - data_source.kdtree_get_pt(b_idx, 4);
    const DistanceType diff5 = a[5] - data_source.kdtree_get_pt(b_idx, 5);
    return (diff0*diff0 + diff1*diff1 + diff2*diff2 + diff3*diff3
            + diff4*diff4 + diff5*diff5);
  }

};

template <class T, class DataSource, typename _DistanceType = T>
struct L2_7D_Adaptor : L2_ND<T, DataSource, _DistanceType> {
  typedef T ElementType;
  typedef _DistanceType DistanceType;

  const DataSource &data_source;

  L2_7D_Adaptor(const DataSource &_data_source) : data_source(_data_source) {}

  inline DistanceType evalMetric(const T *a, const size_t b_idx, size_t size, DistanceType worst_dist) const {
    const DistanceType diff0 = a[0] - data_source.kdtree_get_pt(b_idx, 0);
    const DistanceType diff1 = a[1] - data_source.kdtree_get_pt(b_idx, 1);
    const DistanceType diff2 = a[2] - data_source.kdtree_get_pt(b_idx, 2);
    const DistanceType diff3 = a[3] - data_source.kdtree_get_pt(b_idx, 3);
    const DistanceType diff4 = a[4] - data_source.kdtree_get_pt(b_idx, 4);
    const DistanceType diff5 = a[5] - data_source.kdtree_get_pt(b_idx, 5);
    const DistanceType diff6 = a[6] - data_source.kdtree_get_pt(b_idx, 6);
    return (diff0*diff0 + diff1*diff1 + diff2*diff2 + diff3*diff3
            + diff4*diff4 + diff5*diff5 + diff6*diff6);
  }

  inline DistanceType evalMetric(const T *a, const size_t b_idx, size_t size) const {
    const DistanceType diff0 = a[0] - data_source.kdtree_get_pt(b_idx, 0);
    const DistanceType diff1 = a[1] - data_source.kdtree_get_pt(b_idx, 1);
    const DistanceType diff2 = a[2] - data_source.kdtree_get_pt(b_idx, 2);
    const DistanceType diff3 = a[3] - data_source.kdtree_get_pt(b_idx, 3);
    const DistanceType diff4 = a[4] - data_source.kdtree_get_pt(b_idx, 4);
    const DistanceType diff5 = a[5] - data_source.kdtree_get_pt(b_idx, 5);
    const DistanceType diff6 = a[6] - data_source.kdtree_get_pt(b_idx, 6);
    return (diff0*diff0 + diff1*diff1 + diff2*diff2 + diff3*diff3
            + diff4*diff4 + diff5*diff5 + diff6*diff6);
  }
};

template <class T, class DataSource, typename _DistanceType = T>
struct L2_8D_Adaptor : L2_ND<T, DataSource, _DistanceType> {
  typedef T ElementType;
  typedef _DistanceType DistanceType;

  const DataSource &data_source;

  L2_8D_Adaptor(const DataSource &_data_source) : data_source(_data_source) {}

  inline DistanceType evalMetric(const T *a, const size_t b_idx, size_t size, DistanceType worst_dist) const {
    const DistanceType diff0 = a[0] - data_source.kdtree_get_pt(b_idx, 0);
    const DistanceType diff1 = a[1] - data_source.kdtree_get_pt(b_idx, 1);
    const DistanceType diff2 = a[2] - data_source.kdtree_get_pt(b_idx, 2);
    const DistanceType diff3 = a[3] - data_source.kdtree_get_pt(b_idx, 3);
    const DistanceType diff4 = a[4] - data_source.kdtree_get_pt(b_idx, 4);
    const DistanceType diff5 = a[5] - data_source.kdtree_get_pt(b_idx, 5);
    const DistanceType diff6 = a[6] - data_source.kdtree_get_pt(b_idx, 6);
    const DistanceType diff7 = a[7] - data_source.kdtree_get_pt(b_idx, 7);
    return (diff0*diff0 + diff1*diff1 + diff2*diff2 + diff3*diff3
            + diff4*diff4 + diff5*diff5 + diff6*diff6 + diff7*diff7);
  }

  inline DistanceType evalMetric(const T *a, const size_t b_idx, size_t size) const {
    const DistanceType diff0 = a[0] - data_source.kdtree_get_pt(b_idx, 0);
    const DistanceType diff1 = a[1] - data_source.kdtree_get_pt(b_idx, 1);
    const DistanceType diff2 = a[2] - data_source.kdtree_get_pt(b_idx, 2);
    const DistanceType diff3 = a[3] - data_source.kdtree_get_pt(b_idx, 3);
    const DistanceType diff4 = a[4] - data_source.kdtree_get_pt(b_idx, 4);
    const DistanceType diff5 = a[5] - data_source.kdtree_get_pt(b_idx, 5);
    const DistanceType diff6 = a[6] - data_source.kdtree_get_pt(b_idx, 6);
    const DistanceType diff7 = a[7] - data_source.kdtree_get_pt(b_idx, 7);
    return (diff0*diff0 + diff1*diff1 + diff2*diff2 + diff3*diff3
            + diff4*diff4 + diff5*diff5 + diff6*diff6 + diff7*diff7);
  }

};


// SIMPLE
template <class T, class DataSource, typename _DistanceType = T>
struct L2_Simple_Adaptor : L2_ND<T, DataSource, _DistanceType> {
  typedef T ElementType;
  typedef _DistanceType DistanceType;

  const DataSource &data_source;

  L2_Simple_Adaptor(const DataSource &_data_source)
      : data_source(_data_source) {}

  inline DistanceType evalMetric(const T *a, const size_t b_idx, size_t size) const {
    DistanceType result = DistanceType();
    for (size_t i = 0; i < size; ++i) {
      const DistanceType diff = a[i] - data_source.kdtree_get_pt(b_idx, i);
      result += diff * diff;
    }
    return result;
  }

};


#endif /* LPQ_L2_ND_CPP_ */
