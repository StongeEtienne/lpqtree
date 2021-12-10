


#ifndef LPQ_L21_M_ND_HPP_
#define LPQ_L21_M_ND_HPP_


#include <algorithm>
#include <array>
#include <cmath>   // for abs()


template <typename T> T sqrt2inv() {
  return static_cast<T>(0.707106781);
}


template <typename T> T sqrt3inv() {
  return static_cast<T>(0.577350269);
}


template <typename T> T sqrt4inv() {
  return static_cast<T>(0.5);
}


//General
template <class T, class DataSource, typename _DistanceType = T>
struct L21_M_2D_Adaptor {
  typedef T ElementType;
  typedef _DistanceType DistanceType;

  const DataSource &data_source;

  L21_M_2D_Adaptor(const DataSource &_data_source) : data_source(_data_source) {}

  inline DistanceType evalMetric(const T *a, const size_t b_idx, size_t size, DistanceType worst_dist) const {
    DistanceType result = DistanceType();
    const T *last = a + size;
    size_t d = 0;

    while (a < last) {
      const DistanceType diff0 = a[0] - data_source.kdtree_get_pt(b_idx, d++);
      const DistanceType diff1 = a[1] - data_source.kdtree_get_pt(b_idx, d++);
      result += std::sqrt(diff0*diff0 + diff1*diff1);
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

    // if (size%3 != 0)
    //   throw std::runtime_error("Error: 'dimensionality' must be a multiple of 3");

    /* Process 3 items with each loop for efficiency. */
    while (a < last) {
      const DistanceType diff0 = a[0] - data_source.kdtree_get_pt(b_idx, d++);
      const DistanceType diff1 = a[1] - data_source.kdtree_get_pt(b_idx, d++);
      result += std::sqrt(diff0*diff0 + diff1*diff1);
      a += 2;
    }
    return result;
  }

  template <typename U, typename V>
  inline DistanceType accum_dist(const U a, const V b, const size_t) const {
    return std::abs(a - b) * sqrt2inv<DistanceType>();
  }

  inline DistanceType eval_pair(const T *a, const T *b, size_t size, DistanceType worst_dist) const {
    DistanceType result = DistanceType();
    const T *last = a + size;

    while (a < last) {
      const DistanceType diff0 = a[0] - b[0];
      const DistanceType diff1 = a[1] - b[1];
      result += std::sqrt(diff0*diff0 + diff1*diff1);
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
      const DistanceType diff0 = a[0] - b[0];
      const DistanceType diff1 = a[1] - b[1];
      result += std::sqrt(diff0*diff0 + diff1*diff1);
      a += 2;
      b += 2;
    }
    return result;
  }
};


template <class T, class DataSource, typename _DistanceType = T>
struct L21_M_3D_Adaptor {
  typedef T ElementType;
  typedef _DistanceType DistanceType;

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

  template <typename U, typename V>
  inline DistanceType accum_dist(const U a, const V b, const size_t) const {
    return std::abs(a - b) * sqrt3inv<DistanceType>();
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


template <class T, class DataSource, typename _DistanceType = T>
struct L21_M_4D_Adaptor {
  typedef T ElementType;
  typedef _DistanceType DistanceType;

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

  template <typename U, typename V>
  inline DistanceType accum_dist(const U a, const V b, const size_t) const {
    return std::abs(a - b) * sqrt4inv<DistanceType>();
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


// template <class T, class DataSource, typename _DistanceType = T>
// struct L21_3D_Adaptor {
//   typedef T ElementType;
//   typedef _DistanceType DistanceType;
//
//   const DataSource &data_source;
//
//   L21_3D_Adaptor(const DataSource &_data_source) : data_source(_data_source) {}
//
//   inline DistanceType evalMetric(const T *a, const size_t b_idx, size_t size, DistanceType worst_dist) const {
//     DistanceType result = DistanceType();
//     const T *last = a + size;
//     size_t d = 0;
//
//     // if (size%3 != 0)
//     //   throw std::runtime_error("Error: 'dimensionality' must be a multiple of 3");
//
//     /* Process 3 items with each loop for efficiency. */
//     while (a < last) {
//       const DistanceType diff0 = a[0] - data_source.kdtree_get_pt(b_idx, d++);
//       const DistanceType diff1 = a[1] - data_source.kdtree_get_pt(b_idx, d++);
//       const DistanceType diff2 = a[2] - data_source.kdtree_get_pt(b_idx, d++);
//       result += std::sqrt(diff0 * diff0 + diff1 * diff1 + diff2 * diff2);
//       a += 3;
//
//       if (result > worst_dist) {
//         return result;
//       }
//     }
//     return result;
//   }
//
//   inline DistanceType evalMetric(const T *a, const size_t b_idx, size_t size) const {
//     DistanceType result = DistanceType();
//     const T *last = a + size;
//     size_t d = 0;
//
//     // if (size%3 != 0)
//     //   throw std::runtime_error("Error: 'dimensionality' must be a multiple of 3");
//
//     /* Process 3 items with each loop for efficiency. */
//     while (a < last) {
//       const DistanceType diff0 = a[0] - data_source.kdtree_get_pt(b_idx, d++);
//       const DistanceType diff1 = a[1] - data_source.kdtree_get_pt(b_idx, d++);
//       const DistanceType diff2 = a[2] - data_source.kdtree_get_pt(b_idx, d++);
//       result += std::sqrt(diff0 * diff0 + diff1 * diff1 + diff2 * diff2);
//       a += 3;
//     }
//     return result;
//   }
//
//   template <typename U, typename V>
//   inline DistanceType accum_dist(const U a, const V b, const size_t) const {
//     return std::abs(a - b) * sqrt3inv<DistanceType>();
//   }
//
//   inline DistanceType eval_pair(const T *a, const T *b, size_t size, DistanceType worst_dist) const {
//     DistanceType result = DistanceType();
//     const T *last = a + size;
//
//     while (a < last) {
//       const DistanceType diff0 = a[0] - b[0];
//       const DistanceType diff1 = a[1] - b[1];
//       const DistanceType diff2 = a[2] - b[2];
//       result += std::sqrt(diff0 * diff0 + diff1 * diff1 + diff2 * diff2);
//       a += 3;
//       b += 3;
//
//       if (result > worst_dist) {
//         return result;
//       }
//     }
//     return result;
//   }
//
//   inline DistanceType eval_pair(const T *a, const T *b, size_t size) const {
//     DistanceType result = DistanceType();
//     const T *last = a + size;
//
//     while (a < last) {
//       const DistanceType diff0 = a[0] - b[0];
//       const DistanceType diff1 = a[1] - b[1];
//       const DistanceType diff2 = a[2] - b[2];
//       result += std::sqrt(diff0 * diff0 + diff1 * diff1 + diff2 * diff2);
//       a += 3;
//       b += 3;
//     }
//     return result;
//   }
// };
//
template <class T, class DataSource, typename _DistanceType = T>
struct L21_3D_Adaptor_row {
  typedef T ElementType;
  typedef _DistanceType DistanceType;

  const DataSource &data_source;

  L21_3D_Adaptor_row(const DataSource &_data_source) : data_source(_data_source) {}

  inline DistanceType evalMetric(const T *a, const size_t b_idx, size_t size,
                                 DistanceType worst_dist) const {
    DistanceType result = T();
    const T* vals = data_source.kdtree_get_row(b_idx);
    const T* last = a + size;

    while (a < last) {
      const DistanceType diff0 = a[0] - vals[0];
      const DistanceType diff1 = a[1] - vals[1];
      const DistanceType diff2 = a[2] - vals[2];
      result += std::sqrt(diff0 * diff0 + diff1 * diff1 + diff2 * diff2);

      if ((worst_dist > 0) && (result > worst_dist)) {
        return result;
      }
      a += 3;
      vals +=3;
    }
    return result;
  }

  inline DistanceType evalMetric(const T *a, const size_t b_idx, size_t size) const {
    DistanceType result = T();
    const T* vals = data_source.kdtree_get_row(b_idx);
    const T* last = a + size;

    while (a < last) {
      const DistanceType diff0 = a[0] - vals[0];
      const DistanceType diff1 = a[1] - vals[1];
      const DistanceType diff2 = a[2] - vals[2];
      result += std::sqrt(diff0 * diff0 + diff1 * diff1 + diff2 * diff2);
      a += 3;
      vals +=3;
    }
    return result;
  }

 template <typename U, typename V>
 inline DistanceType accum_dist(const U a, const V b, const size_t) const {
   return std::abs(a - b) * sqrt3inv<DistanceType>();
 }


 inline DistanceType eval_pair(const T *a, const T *b, size_t size, DistanceType worst_dist) const {
   DistanceType result = DistanceType();
   const T *last = a + size;

   while (a < last) {
     const DistanceType diff0 = a[0] - b[0];
     const DistanceType diff1 = a[1] - b[1];
     const DistanceType diff2 = a[2] - b[2];
     result += std::sqrt(diff0 * diff0 + diff1 * diff1 + diff2 * diff2);
     a += 3;
     b += 3;

     if ((worst_dist > 0) && (result > worst_dist)) {
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
     result += std::sqrt(diff0 * diff0 + diff1 * diff1 + diff2 * diff2);
     a += 3;
     b += 3;
   }
   return result;
 }
};


#endif /* LPQ_L21_M_ND_HPP_ */
