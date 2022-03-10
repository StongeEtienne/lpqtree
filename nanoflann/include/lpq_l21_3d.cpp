

#ifndef LPQ_L21_3D_CPP_
#define LPQ_L21_3D_CPP_

#include <algorithm>
#include <array>
#include <cmath>   // for abs()


// Abstract Struct for all L21 2D points function with accum_dist() and eval_pair()
template <class T, class DataSource, typename _DistanceType = T>
struct L21_M_3D {
  typedef T ElementType;
  typedef _DistanceType DistanceType;

  //int dist_exponent = 1; for M > 1
  int pair_exponent = 1;

  //general function, unused for specific Lpq
  void setup_lpq(int p, int q, int n_dim){}

  template <typename U, typename V>
  inline DistanceType accum_dist(const U a, const V b, const size_t) const {
    return std::abs(a - b) * static_cast<T>(0.577350269);
  }

  inline DistanceType eval_pair(const T *a, const T *b, size_t size, DistanceType worst_dist) const {
    DistanceType result = DistanceType();
    const T *last = a + size;

    while (a < last) {
      const DistanceType diff0 = a[0] - b[0];
      const DistanceType diff1 = a[1] - b[1];
      const DistanceType diff2 = a[2] - b[2];
      result += std::sqrt(diff0*diff0 + diff1*diff1 + diff2*diff2);
      a += 3;
      b += 3;

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
      result += std::sqrt(diff0*diff0 + diff1*diff1 + diff2*diff2);
      a += 3;
      b += 3;
    }
    return result;
  }
};

//General
template <class T, class DataSource, typename _DistanceType = T>
struct L21_M_3D_Adaptor : L21_M_3D<T, DataSource, _DistanceType> {
  typedef T ElementType;
  typedef _DistanceType DistanceType;

  int dist_exponent = 1;
  const DataSource &data_source;

  L21_M_3D_Adaptor(const DataSource &_data_source) : data_source(_data_source) {}

  inline DistanceType evalMetric(const T *a, const size_t b_idx, size_t size, DistanceType worst_dist) const {
    DistanceType result = DistanceType();
    const T *last = a + size;
    size_t d = 0;

    while (a < last) {
      const DistanceType diff0 = a[0] - data_source.kdtree_get_pt(b_idx, d++);
      const DistanceType diff1 = a[1] - data_source.kdtree_get_pt(b_idx, d++);
      const DistanceType diff2 = a[2] - data_source.kdtree_get_pt(b_idx, d++);
      result += std::sqrt(diff0*diff0 + diff1*diff1 + diff2*diff2);
      a += 3;

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
      result += std::sqrt(diff0*diff0 + diff1*diff1 + diff2*diff2);
      a += 3;
    }
    return result;
  }

};


//1x3D
template <class T, class DataSource, typename _DistanceType = T>
struct L21_1_3D_Adaptor : L21_M_3D<T, DataSource, _DistanceType> {
  typedef T ElementType;
  typedef _DistanceType DistanceType;

  int dist_exponent = 2;
  const DataSource &data_source;

  L21_1_3D_Adaptor(const DataSource &_data_source) : data_source(_data_source) {}

  template <typename U, typename V>
  inline DistanceType accum_dist(const U a, const V b, const size_t) const{ //override
    // equivalent to L2
    DistanceType diff = (a - b);
    return diff*diff;
  }

  inline DistanceType evalMetric(const T *a, const size_t b_idx, size_t size, DistanceType worst_dist) const {
    // equivalent to L2_3D
    const DistanceType diff0 = a[0] - data_source.kdtree_get_pt(b_idx, 0);
    const DistanceType diff1 = a[1] - data_source.kdtree_get_pt(b_idx, 1);
    const DistanceType diff2 = a[2] - data_source.kdtree_get_pt(b_idx, 2);

    return diff0*diff0 + diff1*diff1 + diff2*diff2;
  }

  inline DistanceType evalMetric(const T *a, const size_t b_idx, size_t size) const {
    // equivalent to L2_3D
    const DistanceType diff0 = a[0] - data_source.kdtree_get_pt(b_idx, 0);
    const DistanceType diff1 = a[1] - data_source.kdtree_get_pt(b_idx, 1);
    const DistanceType diff2 = a[2] - data_source.kdtree_get_pt(b_idx, 2);

    return diff0*diff0 + diff1*diff1 + diff2*diff2;
  }
};


//2x3D
template <class T, class DataSource, typename _DistanceType = T>
struct L21_2_3D_Adaptor : L21_M_3D<T, DataSource, _DistanceType> {
  typedef T ElementType;
  typedef _DistanceType DistanceType;

  int dist_exponent = 1;
  const DataSource &data_source;

  L21_2_3D_Adaptor(const DataSource &_data_source) : data_source(_data_source) {}

  inline DistanceType evalMetric(const T *a, const size_t b_idx, size_t size, DistanceType worst_dist) const {
    const DistanceType diff0 = a[0] - data_source.kdtree_get_pt(b_idx, 0);
    const DistanceType diff1 = a[1] - data_source.kdtree_get_pt(b_idx, 1);
    const DistanceType diff2 = a[2] - data_source.kdtree_get_pt(b_idx, 2);
    const DistanceType diff3 = a[3] - data_source.kdtree_get_pt(b_idx, 3);
    const DistanceType diff4 = a[4] - data_source.kdtree_get_pt(b_idx, 4);
    const DistanceType diff5 = a[5] - data_source.kdtree_get_pt(b_idx, 5);

    return (std::sqrt(diff0*diff0 + diff1*diff1 + diff2*diff2)
            + std::sqrt(diff3*diff3 + diff4*diff4 + diff5*diff5));
  }

  inline DistanceType evalMetric(const T *a, const size_t b_idx, size_t size) const {
    const DistanceType diff0 = a[0] - data_source.kdtree_get_pt(b_idx, 0);
    const DistanceType diff1 = a[1] - data_source.kdtree_get_pt(b_idx, 1);
    const DistanceType diff2 = a[2] - data_source.kdtree_get_pt(b_idx, 2);
    const DistanceType diff3 = a[3] - data_source.kdtree_get_pt(b_idx, 3);
    const DistanceType diff4 = a[4] - data_source.kdtree_get_pt(b_idx, 4);
    const DistanceType diff5 = a[5] - data_source.kdtree_get_pt(b_idx, 5);

    return (std::sqrt(diff0*diff0 + diff1*diff1 + diff2*diff2)
            + std::sqrt(diff3*diff3 + diff4*diff4 + diff5*diff5));
  }
};


//3x3D
template <class T, class DataSource, typename _DistanceType = T>
struct L21_3_3D_Adaptor : L21_M_3D<T, DataSource, _DistanceType> {
  typedef T ElementType;
  typedef _DistanceType DistanceType;

  int dist_exponent = 1;
  const DataSource &data_source;

  L21_3_3D_Adaptor(const DataSource &_data_source) : data_source(_data_source) {}

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


    return (std::sqrt(diff0*diff0 + diff1*diff1 + diff2*diff2)
            + std::sqrt(diff3*diff3 + diff4*diff4 + diff5*diff5)
            + std::sqrt(diff6*diff6 + diff7*diff7 + diff8*diff8));
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


    return (std::sqrt(diff0*diff0 + diff1*diff1 + diff2*diff2)
            + std::sqrt(diff3*diff3 + diff4*diff4 + diff5*diff5)
            + std::sqrt(diff6*diff6 + diff7*diff7 + diff8*diff8));
  }
};

//4x3D
template <class T, class DataSource, typename _DistanceType = T>
struct L21_4_3D_Adaptor : L21_M_3D<T, DataSource, _DistanceType> {
  typedef T ElementType;
  typedef _DistanceType DistanceType;

  int dist_exponent = 1;
  const DataSource &data_source;

  L21_4_3D_Adaptor(const DataSource &_data_source) : data_source(_data_source) {}

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


    return (std::sqrt(diff0*diff0 + diff1*diff1 + diff2*diff2)
            + std::sqrt(diff3*diff3 + diff4*diff4 + diff5*diff5)
            + std::sqrt(diff6*diff6 + diff7*diff7 + diff8*diff8)
            + std::sqrt(diff9*diff9 + diff10*diff10 + diff11*diff11));
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


    return (std::sqrt(diff0*diff0 + diff1*diff1 + diff2*diff2)
            + std::sqrt(diff3*diff3 + diff4*diff4 + diff5*diff5)
            + std::sqrt(diff6*diff6 + diff7*diff7 + diff8*diff8)
            + std::sqrt(diff9*diff9 + diff10*diff10 + diff11*diff11));
  }
};


//2x3D
template <class T, class DataSource, typename _DistanceType = T>
struct L21_2_3D_Adaptor_row : L21_M_3D<T, DataSource, _DistanceType> {
  typedef T ElementType;
  typedef _DistanceType DistanceType;

  int dist_exponent = 1;
  const DataSource &data_source;

  L21_2_3D_Adaptor_row(const DataSource &_data_source) : data_source(_data_source) {}

  inline DistanceType evalMetric(const T *a, const size_t b_idx, size_t size, DistanceType worst_dist) const {
    const T* vals = data_source.kdtree_get_row(b_idx);
    const DistanceType diff0 = a[0] - vals[0];
    const DistanceType diff1 = a[1] - vals[1];
    const DistanceType diff2 = a[2] - vals[2];
    const DistanceType diff3 = a[3] - vals[3];
    const DistanceType diff4 = a[4] - vals[4];
    const DistanceType diff5 = a[5] - vals[5];

    return (std::sqrt(diff0*diff0 + diff1*diff1 + diff2*diff2)
            + std::sqrt(diff3*diff3 + diff4*diff4 + diff5*diff5));
  }

  inline DistanceType evalMetric(const T *a, const size_t b_idx, size_t size) const {
    const T* vals = data_source.kdtree_get_row(b_idx);
    const DistanceType diff0 = a[0] - vals[0];
    const DistanceType diff1 = a[1] - vals[1];
    const DistanceType diff2 = a[2] - vals[2];
    const DistanceType diff3 = a[3] - vals[3];
    const DistanceType diff4 = a[4] - vals[4];
    const DistanceType diff5 = a[5] - vals[5];

    return (sqrt(diff0*diff0 + diff1*diff1 + diff2*diff2)
            + sqrt(diff3*diff3 + diff4*diff4 + diff5*diff5));
  }
};


//2x3D
//template <class T, class DataSource, typename _DistanceType = T>
//struct L21_2_3D_Adaptor_row2 : L21_M_3D<T, DataSource, _DistanceType> {
//  typedef T ElementType;
//  typedef _DistanceType DistanceType;
//
//  int dist_exponent = 1;
//  const DataSource &data_source;
//
//  L21_2_3D_Adaptor_row2(const DataSource &_data_source) : data_source(_data_source) {}
//
//  inline DistanceType evalMetric(const T *a, const size_t b_idx, size_t size, DistanceType worst_dist) const {
//    const T* vals = data_source.kdtree_get_row(b_idx);
//    const DistanceType diff[6] = {
//      a[0] - vals[0],
//      a[1] - vals[1],
//      a[2] - vals[2],
//      a[3] - vals[3],
//      a[4] - vals[4],
//      a[5] - vals[5]};
//
//    return (std::sqrt(diff[0]*diff[0] + diff[1]*diff[1] + diff[2]*diff[2])
//            + std::sqrt(diff[3]*diff[3] + diff[4]*diff[4] + diff[5]*diff[5]));
//  }
//
//  inline DistanceType evalMetric(const T *a, const size_t b_idx, size_t size) const {
//    const T* vals = data_source.kdtree_get_row(b_idx);
//    const DistanceType diff[6] = {
//      a[0] - vals[0],
//      a[1] - vals[1],
//      a[2] - vals[2],
//      a[3] - vals[3],
//      a[4] - vals[4],
//      a[5] - vals[5]};
//
//    return (std::sqrt(diff[0]*diff[0] + diff[1]*diff[1] + diff[2]*diff[2])
//            + std::sqrt(diff[3]*diff[3] + diff[4]*diff[4] + diff[5]*diff[5]));
//  }
//};

//
//
//// ROW opt
//template <class T, class DataSource, typename _DistanceType = T>
//struct L21_3D_Adaptor_row : L21_M_3D<T, DataSource, _DistanceType> {
//  typedef T ElementType;
//  typedef _DistanceType DistanceType;
//
//  int dist_exponent = 1;
//  const DataSource &data_source;
//
//  L21_3D_Adaptor_row(const DataSource &_data_source) : data_source(_data_source) {}
//
//  inline DistanceType evalMetric(const T *a, const size_t b_idx, size_t size, DistanceType worst_dist) const {
//    DistanceType result = T();
//    const T* vals = data_source.kdtree_get_row(b_idx);
//    const T* last = a + size;
//
//    while (a < last) {
//      const DistanceType diff0 = a[0] - vals[0];
//      const DistanceType diff1 = a[1] - vals[1];
//      const DistanceType diff2 = a[2] - vals[2];
//      result += std::sqrt(diff0 * diff0 + diff1 * diff1 + diff2 * diff2);
//
//      if ((worst_dist > 0) && (result > worst_dist)) {
//        return result;
//      }
//      a += 3;
//      vals +=3;
//    }
//    return result;
//  }
//
//  inline DistanceType evalMetric(const T *a, const size_t b_idx, size_t size) const {
//    DistanceType result = T();
//    const T* vals = data_source.kdtree_get_row(b_idx);
//    const T* last = a + size;
//
//    while (a < last) {
//      const DistanceType diff0 = a[0] - vals[0];
//      const DistanceType diff1 = a[1] - vals[1];
//      const DistanceType diff2 = a[2] - vals[2];
//      result += std::sqrt(diff0 * diff0 + diff1 * diff1 + diff2 * diff2);
//      a += 3;
//      vals +=3;
//    }
//    return result;
//  }
//};

#endif /* LPQ_L21_3D_CPP_ */
