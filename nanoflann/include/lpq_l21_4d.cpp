

#ifndef LPQ_L21_4D_CPP_
#define LPQ_L21_4D_CPP_

#include <algorithm>
#include <array>
#include <cmath>   // for abs()


// Abstract Struct for all L21 2D points function with accum_dist() and eval_pair()
template <class T, class DataSource, typename _DistanceType = T>
struct L21_M_4D {
  typedef T ElementType;
  typedef _DistanceType DistanceType;

  //int dist_exponent = 1; for M > 1
  int pair_exponent = 1;

  //general function, unused for specific Lpq
  void setup_lpq(int p, int q, int n_dim){}

  template <typename U, typename V>
  inline DistanceType accum_dist(const U a, const V b, const size_t) const {
    return std::abs(a - b) * static_cast<T>(0.5);
  }

  inline DistanceType eval_pair(const T *a, const T *b, size_t size, DistanceType worst_dist) const {
    DistanceType result = DistanceType();
    const T *last = a + size;

    while (a < last) {
      const DistanceType diff0 = a[0] - b[0];
      const DistanceType diff1 = a[1] - b[1];
      const DistanceType diff2 = a[2] - b[2];
      const DistanceType diff3 = a[3] - b[3];
      result += std::sqrt(diff0*diff0 + diff1*diff1 + diff2*diff2 + diff3*diff3);
      a += 4;
      b += 4;

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
      const DistanceType diff0 = a[0] - b[0];
      const DistanceType diff1 = a[1] - b[1];
      const DistanceType diff2 = a[2] - b[2];
      const DistanceType diff3 = a[3] - b[3];
      result += std::sqrt(diff0*diff0 + diff1*diff1 + diff2*diff2 + diff3*diff3);
      a += 4;
      b += 4;
    }
    return result;
  }
};

//General
template <class T, class DataSource, typename _DistanceType = T>
struct L21_M_4D_Adaptor  : L21_M_4D<T, DataSource, _DistanceType> {
  typedef T ElementType;
  typedef _DistanceType DistanceType;

  int dist_exponent = 1;
  const DataSource &data_source;

  L21_M_4D_Adaptor(const DataSource &_data_source) : data_source(_data_source) {}

  inline DistanceType evalMetric(const T *a, const size_t b_idx, size_t size, DistanceType worst_dist) const {
    DistanceType result = DistanceType();
    const T *last = a + size;
    size_t d = 0;

    while (a < last) {
      const DistanceType diff0 = a[0] - data_source.kdtree_get_pt(b_idx, d++);
      const DistanceType diff1 = a[1] - data_source.kdtree_get_pt(b_idx, d++);
      const DistanceType diff2 = a[2] - data_source.kdtree_get_pt(b_idx, d++);
      const DistanceType diff3 = a[3] - data_source.kdtree_get_pt(b_idx, d++);
      result += std::sqrt(diff0*diff0 + diff1*diff1 + diff2*diff2 + diff3*diff3);
      a += 4;

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
      const DistanceType diff0 = a[0] - data_source.kdtree_get_pt(b_idx, d++);
      const DistanceType diff1 = a[1] - data_source.kdtree_get_pt(b_idx, d++);
      const DistanceType diff2 = a[2] - data_source.kdtree_get_pt(b_idx, d++);
      const DistanceType diff3 = a[3] - data_source.kdtree_get_pt(b_idx, d++);
      result += std::sqrt(diff0*diff0 + diff1*diff1 + diff2*diff2 + diff3*diff3);
      a += 4;
    }
    return result;
  }
};



//1x4D
template <class T, class DataSource, typename _DistanceType = T>
struct L21_1_4D_Adaptor : L21_M_4D<T, DataSource, _DistanceType> {
  typedef T ElementType;
  typedef _DistanceType DistanceType;

  int dist_exponent = 2;
  const DataSource &data_source;

  L21_1_4D_Adaptor(const DataSource &_data_source) : data_source(_data_source) {}

  template <typename U, typename V>
  inline DistanceType accum_dist(const U a, const V b, const size_t) const{ //override
    // equivalent to L2
    DistanceType diff = (a - b);
    return diff*diff;
  }

  inline DistanceType evalMetric(const T *a, const size_t b_idx, size_t size, DistanceType worst_dist) const {
    // equivalent to L2_4D
    const DistanceType diff0 = a[0] - data_source.kdtree_get_pt(b_idx, 0);
    const DistanceType diff1 = a[1] - data_source.kdtree_get_pt(b_idx, 1);
    const DistanceType diff2 = a[2] - data_source.kdtree_get_pt(b_idx, 2);
    const DistanceType diff3 = a[3] - data_source.kdtree_get_pt(b_idx, 3);

    return diff0*diff0 + diff1*diff1 + diff2*diff2 + diff3*diff3;
  }

  inline DistanceType evalMetric(const T *a, const size_t b_idx, size_t size) const {
    // equivalent to L2_4D
    const DistanceType diff0 = a[0] - data_source.kdtree_get_pt(b_idx, 0);
    const DistanceType diff1 = a[1] - data_source.kdtree_get_pt(b_idx, 1);
    const DistanceType diff2 = a[2] - data_source.kdtree_get_pt(b_idx, 2);
    const DistanceType diff3 = a[3] - data_source.kdtree_get_pt(b_idx, 3);

    return diff0*diff0 + diff1*diff1 + diff2*diff2 + diff3*diff3;
  }
};


//2x4D
template <class T, class DataSource, typename _DistanceType = T>
struct L21_2_4D_Adaptor : L21_M_4D<T, DataSource, _DistanceType> {
  typedef T ElementType;
  typedef _DistanceType DistanceType;

  int dist_exponent = 1;
  const DataSource &data_source;

  L21_2_4D_Adaptor(const DataSource &_data_source) : data_source(_data_source) {}

  inline DistanceType evalMetric(const T *a, const size_t b_idx, size_t size, DistanceType worst_dist) const {
    const DistanceType diff0 = a[0] - data_source.kdtree_get_pt(b_idx, 0);
    const DistanceType diff1 = a[1] - data_source.kdtree_get_pt(b_idx, 1);
    const DistanceType diff2 = a[2] - data_source.kdtree_get_pt(b_idx, 2);
    const DistanceType diff3 = a[3] - data_source.kdtree_get_pt(b_idx, 3);

    const DistanceType diff4 = a[4] - data_source.kdtree_get_pt(b_idx, 4);
    const DistanceType diff5 = a[5] - data_source.kdtree_get_pt(b_idx, 5);
    const DistanceType diff6 = a[6] - data_source.kdtree_get_pt(b_idx, 6);
    const DistanceType diff7 = a[7] - data_source.kdtree_get_pt(b_idx, 7);

    return (std::sqrt(diff0*diff0 + diff1*diff1 + diff2*diff2 + diff3*diff3)
            + std::sqrt(diff4*diff4 + diff5*diff5 + diff6*diff6 + diff7*diff7));
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

    return (std::sqrt(diff0*diff0 + diff1*diff1 + diff2*diff2 + diff3*diff3)
            + std::sqrt(diff4*diff4 + diff5*diff5 + diff6*diff6 + diff7*diff7));
  }
};

//3x4D
template <class T, class DataSource, typename _DistanceType = T>
struct L21_3_4D_Adaptor : L21_M_4D<T, DataSource, _DistanceType> {
  typedef T ElementType;
  typedef _DistanceType DistanceType;

  int dist_exponent = 1;
  const DataSource &data_source;

  L21_3_4D_Adaptor(const DataSource &_data_source) : data_source(_data_source) {}

  inline DistanceType evalMetric(const T *a, const size_t b_idx, size_t size, DistanceType worst_dist) const {
    const DistanceType diff0 = a[0] - data_source.kdtree_get_pt(b_idx, 0);
    const DistanceType diff1 = a[1] - data_source.kdtree_get_pt(b_idx, 1);
    const DistanceType diff2 = a[2] - data_source.kdtree_get_pt(b_idx, 2);
    const DistanceType diff3 = a[3] - data_source.kdtree_get_pt(b_idx, 3);

    const DistanceType diff4 = a[4] - data_source.kdtree_get_pt(b_idx, 4);
    const DistanceType diff5 = a[5] - data_source.kdtree_get_pt(b_idx, 5);
    const DistanceType diff6 = a[6] - data_source.kdtree_get_pt(b_idx, 6);
    const DistanceType diff7 = a[7] - data_source.kdtree_get_pt(b_idx, 7);

    const DistanceType diff8 = a[8] - data_source.kdtree_get_pt(b_idx, 8);
    const DistanceType diff9 = a[9] - data_source.kdtree_get_pt(b_idx, 9);
    const DistanceType diff10 = a[10] - data_source.kdtree_get_pt(b_idx, 10);
    const DistanceType diff11 = a[11] - data_source.kdtree_get_pt(b_idx, 11);


    return (std::sqrt(diff0*diff0 + diff1*diff1 + diff2*diff2 + diff3*diff3)
            + std::sqrt(diff4*diff4 + diff5*diff5 + diff6*diff6 + diff7*diff7)
            + std::sqrt(diff8*diff8 + diff9*diff9 + diff10*diff10 + diff11*diff11));
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

    const DistanceType diff8 = a[8] - data_source.kdtree_get_pt(b_idx, 8);
    const DistanceType diff9 = a[9] - data_source.kdtree_get_pt(b_idx, 9);
    const DistanceType diff10 = a[10] - data_source.kdtree_get_pt(b_idx, 10);
    const DistanceType diff11 = a[11] - data_source.kdtree_get_pt(b_idx, 11);

    return (std::sqrt(diff0*diff0 + diff1*diff1 + diff2*diff2 + diff3*diff3)
            + std::sqrt(diff4*diff4 + diff5*diff5 + diff6*diff6 + diff7*diff7)
            + std::sqrt(diff8*diff8 + diff9*diff9 + diff10*diff10 + diff11*diff11));
  }
};

//4x4D
template <class T, class DataSource, typename _DistanceType = T>
struct L21_4_4D_Adaptor : L21_M_4D<T, DataSource, _DistanceType> {
  typedef T ElementType;
  typedef _DistanceType DistanceType;

  int dist_exponent = 1;
  const DataSource &data_source;

  L21_4_4D_Adaptor(const DataSource &_data_source) : data_source(_data_source) {}

  inline DistanceType evalMetric(const T *a, const size_t b_idx, size_t size, DistanceType worst_dist) const {
    const DistanceType diff0 = a[0] - data_source.kdtree_get_pt(b_idx, 0);
    const DistanceType diff1 = a[1] - data_source.kdtree_get_pt(b_idx, 1);
    const DistanceType diff2 = a[2] - data_source.kdtree_get_pt(b_idx, 2);
    const DistanceType diff3 = a[3] - data_source.kdtree_get_pt(b_idx, 3);

    const DistanceType diff4 = a[4] - data_source.kdtree_get_pt(b_idx, 4);
    const DistanceType diff5 = a[5] - data_source.kdtree_get_pt(b_idx, 5);
    const DistanceType diff6 = a[6] - data_source.kdtree_get_pt(b_idx, 6);
    const DistanceType diff7 = a[7] - data_source.kdtree_get_pt(b_idx, 7);

    const DistanceType diff8 = a[8] - data_source.kdtree_get_pt(b_idx, 8);
    const DistanceType diff9 = a[9] - data_source.kdtree_get_pt(b_idx, 9);
    const DistanceType diff10 = a[10] - data_source.kdtree_get_pt(b_idx, 10);
    const DistanceType diff11 = a[11] - data_source.kdtree_get_pt(b_idx, 11);

    const DistanceType diff12 = a[12] - data_source.kdtree_get_pt(b_idx, 12);
    const DistanceType diff13 = a[13] - data_source.kdtree_get_pt(b_idx, 13);
    const DistanceType diff14 = a[14] - data_source.kdtree_get_pt(b_idx, 14);
    const DistanceType diff15 = a[15] - data_source.kdtree_get_pt(b_idx, 15);

    return (std::sqrt(diff0*diff0 + diff1*diff1 + diff2*diff2 + diff3*diff3)
            + std::sqrt(diff4*diff4 + diff5*diff5 + diff6*diff6 + diff7*diff7)
            + std::sqrt(diff8*diff8 + diff9*diff9 + diff10*diff10 + diff11*diff11)
            + std::sqrt(diff12*diff12 + diff13*diff13 + diff14*diff14 + diff15*diff15));
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

    const DistanceType diff8 = a[8] - data_source.kdtree_get_pt(b_idx, 8);
    const DistanceType diff9 = a[9] - data_source.kdtree_get_pt(b_idx, 9);
    const DistanceType diff10 = a[10] - data_source.kdtree_get_pt(b_idx, 10);
    const DistanceType diff11 = a[11] - data_source.kdtree_get_pt(b_idx, 11);

    const DistanceType diff12 = a[12] - data_source.kdtree_get_pt(b_idx, 12);
    const DistanceType diff13 = a[13] - data_source.kdtree_get_pt(b_idx, 13);
    const DistanceType diff14 = a[14] - data_source.kdtree_get_pt(b_idx, 14);
    const DistanceType diff15 = a[15] - data_source.kdtree_get_pt(b_idx, 15);

    return (std::sqrt(diff0*diff0 + diff1*diff1 + diff2*diff2 + diff3*diff3)
            + std::sqrt(diff4*diff4 + diff5*diff5 + diff6*diff6 + diff7*diff7)
            + std::sqrt(diff8*diff8 + diff9*diff9 + diff10*diff10 + diff11*diff11)
            + std::sqrt(diff12*diff12 + diff13*diff13 + diff14*diff14 + diff15*diff15));
  }
};

#endif /* LPQ_L21_4D_CPP_ */
