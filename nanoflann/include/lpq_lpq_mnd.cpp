
#ifndef LPQ_Lpq_MND_CPP_
#define LPQ_Lpq_MND_CPP_

#include <algorithm>
#include <array>
#include <cmath>   // for abs()
#include <iostream>



// General Lpq MND Adaptor, when P != Q  (otherwise use Lp where P=Q), and P or Q > 2
template <class T, class DataSource, typename _DistanceType = T>
struct Lpq_MND_Adaptor {
  typedef T ElementType;
  typedef _DistanceType DistanceType;

  DistanceType lpq_factor; // std::min(std::pow(N, (1.0/static_cast<T>(P)) - (1.0/static_cast<T>(Q))), 1.0)
  DistanceType q_div_p; // static_cast<T>(Q) / static_cast<T>(P);
  int p_val; //= P = dist_exponent
  //int q; //= Q; use q_div_p instead
  int p_ndim; //= N;
  int dist_exponent = 0; // = Q;
  int pair_exponent = 0; // = Q;

  const DataSource &data_source;

  Lpq_MND_Adaptor(const DataSource &_data_source) : data_source(_data_source) {}

  // setup function to initialize values, required for a general Lpq distance
  void setup_lpq(int p, int q, int n_dim){
    dist_exponent = q;
    pair_exponent = q;
    p_val = p;
    p_ndim = n_dim;
    q_div_p = static_cast<T>(q) / static_cast<T>(p);
    if (q >= p){
      lpq_factor = 1.0;
    }
    else{ // p > q
      lpq_factor = std::pow(static_cast<T>(n_dim), (1.0/static_cast<T>(p)) - (1.0/static_cast<T>(q))) - 1e-8; // always < 1.0
    }
  }

  template <typename U, typename V>
  inline DistanceType accum_dist(const U a, const V b, const size_t) const {
    return std::pow(std::abs(a - b)*lpq_factor, dist_exponent);
  }

  inline DistanceType eval_pair(const T *a, const T *b, size_t size, DistanceType worst_dist) const {
    const T *last = a + size;
    DistanceType result = DistanceType();
    DistanceType resultp;

    // n associated with p,  m associated with q
    while (a < last) {
      resultp = DistanceType(); // init to 0.0float  0.0double
      for (int i = 0; i < p_ndim; ++i) {
        resultp += std::pow(std::abs(*a++ - *b++), p_val); // Sum A^p
      }
      result += std::pow(resultp, q_div_p); // Sum [Sum A^p]^(q/p)

      if ((worst_dist > 0) && (result > worst_dist)) {
        return result;
      }
    }
    return result;
  }

  inline DistanceType eval_pair(const T *a, const T *b, size_t size) const {
    const T *last = a + size;
    DistanceType result = DistanceType();
    DistanceType resultp;

    // n associated with p,  m associated with q
    while (a < last) {
      resultp = DistanceType(); // init to 0.0float  0.0double
      for (int i = 0; i < p_ndim; ++i) {
        resultp += std::pow(std::abs(*a++ - *b++), p_val); // Sum A^p
      }
      result += std::pow(resultp, q_div_p); // Sum [Sum A^p]^(q/p)
    }
    return result;
  }


  inline DistanceType evalMetric(const T *a, const size_t b_idx, size_t size,
                                 DistanceType worst_dist) const {
    const T *last = a + size;
    DistanceType result = DistanceType();
    DistanceType resultp;
    size_t d = 0;

    // n associated with p,  m associated with q
    while (a < last) {
      resultp = DistanceType(); // init to 0.0float  0.0double
      for (int i = 0; i < p_ndim; ++i) {
        resultp += std::pow(std::abs(*a++ - data_source.kdtree_get_pt(b_idx, d++)), p_val); // Sum A^p
      }
      result += std::pow(resultp, q_div_p); // Sum [Sum A^p]^(q/p)

      if ((worst_dist > 0) && (result > worst_dist)) {
        return result;
      }
    }
    return result;
  }

  inline DistanceType evalMetric(const T *a, const size_t b_idx, size_t size) const {
    const T *last = a + size;
    DistanceType result = DistanceType();
    DistanceType resultp;
    size_t d = 0;

    // n associated with p,  m associated with q
    while (a < last) {
      resultp = DistanceType(); // init to 0.0float  0.0double
      for (int i = 0; i < p_ndim; ++i) {
        resultp += std::pow(std::abs(*a++ - data_source.kdtree_get_pt(b_idx, d++)), p_val); // Sum A^p
      }
      result += std::pow(resultp, q_div_p); // Sum [Sum A^p]^(q/p)
    }
    return result;
  }

};

#endif /* LPQ_Lpq_MND_CPP_ */
