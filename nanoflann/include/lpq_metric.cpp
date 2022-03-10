

#ifndef LPQ_METRIC_CPP_
#define LPQ_METRIC_CPP_

#include <lpq_l1_nd.cpp>
#include <lpq_l2_nd.cpp>
//#include <lpq_l21.cpp>
#include <lpq_l21_2d.cpp>
#include <lpq_l21_3d.cpp>
#include <lpq_l21_4d.cpp>

#include <lpq_l12_2d.cpp>
#include <lpq_l12_3d.cpp>
#include <lpq_l12_4d.cpp>

#include <lpq_lp_nd.cpp>
#include <lpq_lpq_mnd.cpp>

namespace nanoflann {
/** @addtogroup metric_grp Metric (distance) classes
 * @{ */

struct Metric {};

/** Metaprogramming helper traits class for the L1 (Manhattan) metric */
struct metric_L1_ND : public Metric {
  template <class T, class DataSource> struct traits {
    typedef L1_ND_Adaptor<T, DataSource> distance_t;
  };
};

struct metric_L1_1D : public Metric {
  template <class T, class DataSource> struct traits {
    typedef L1_1D_Adaptor<T, DataSource> distance_t;
  };
};

struct metric_L1_2D : public Metric {
  template <class T, class DataSource> struct traits {
    typedef L1_2D_Adaptor<T, DataSource> distance_t;
  };
};

struct metric_L1_3D : public Metric {
  template <class T, class DataSource> struct traits {
    typedef L1_3D_Adaptor<T, DataSource> distance_t;
  };
};

struct metric_L1_4D : public Metric {
  template <class T, class DataSource> struct traits {
    typedef L1_4D_Adaptor<T, DataSource> distance_t;
  };
};

struct metric_L1_5D : public Metric {
  template <class T, class DataSource> struct traits {
    typedef L1_5D_Adaptor<T, DataSource> distance_t;
  };
};

struct metric_L1_6D : public Metric {
  template <class T, class DataSource> struct traits {
    typedef L1_6D_Adaptor<T, DataSource> distance_t;
  };
};

struct metric_L1_7D : public Metric {
  template <class T, class DataSource> struct traits {
    typedef L1_7D_Adaptor<T, DataSource> distance_t;
  };
};

struct metric_L1_8D : public Metric {
  template <class T, class DataSource> struct traits {
    typedef L1_8D_Adaptor<T, DataSource> distance_t;
  };
};


/** Metaprogramming helper traits class for the L2 (Euclidean) metric */
struct metric_L2_ND : public Metric {
  template <class T, class DataSource> struct traits {
    typedef L2_ND_Adaptor<T, DataSource> distance_t;
  };
};

struct metric_L2_1D : public Metric {
  template <class T, class DataSource> struct traits {
    typedef L2_1D_Adaptor<T, DataSource> distance_t;
  };
};

struct metric_L2_2D : public Metric {
  template <class T, class DataSource> struct traits {
    typedef L2_2D_Adaptor<T, DataSource> distance_t;
  };
};

struct metric_L2_3D : public Metric {
  template <class T, class DataSource> struct traits {
    typedef L2_3D_Adaptor<T, DataSource> distance_t;
  };
};

struct metric_L2_4D : public Metric {
  template <class T, class DataSource> struct traits {
    typedef L2_4D_Adaptor<T, DataSource> distance_t;
  };
};

struct metric_L2_5D : public Metric {
  template <class T, class DataSource> struct traits {
    typedef L2_5D_Adaptor<T, DataSource> distance_t;
  };
};

struct metric_L2_6D : public Metric {
  template <class T, class DataSource> struct traits {
    typedef L2_6D_Adaptor<T, DataSource> distance_t;
  };
};

struct metric_L2_7D : public Metric {
  template <class T, class DataSource> struct traits {
    typedef L2_7D_Adaptor<T, DataSource> distance_t;
  };
};

struct metric_L2_8D : public Metric {
  template <class T, class DataSource> struct traits {
    typedef L2_8D_Adaptor<T, DataSource> distance_t;
  };
};

struct metric_L2_Simple : public Metric {
  template <class T, class DataSource> struct traits {
    typedef L2_Simple_Adaptor<T, DataSource> distance_t;
  };
};

// /** Metaprogramming helper traits class for the SO3_InnerProdQuat metric */
// struct metric_SO2 : public Metric {
//   template <class T, class DataSource> struct traits {
//     typedef SO2_Adaptor<T, DataSource> distance_t;
//   };
// };
// /** Metaprogramming helper traits class for the SO3_InnerProdQuat metric */
// struct metric_SO3 : public Metric {
//   template <class T, class DataSource> struct traits {
//     typedef SO3_Adaptor<T, DataSource> distance_t;
//   };
// };

/** Etienne St-Onge Lpq adaptor */
// L21: M x 2D
struct metric_L21_M_2D : public Metric {
  template <class T, class DataSource> struct traits {
    typedef L21_M_2D_Adaptor<T, DataSource> distance_t;
  };
};

struct metric_L21_1_2D : public Metric {
  template <class T, class DataSource> struct traits {
    typedef L21_1_2D_Adaptor<T, DataSource> distance_t;
  };
};

struct metric_L21_2_2D : public Metric {
  template <class T, class DataSource> struct traits {
    typedef L21_2_2D_Adaptor<T, DataSource> distance_t;
  };
};

struct metric_L21_3_2D : public Metric {
  template <class T, class DataSource> struct traits {
    typedef L21_3_2D_Adaptor<T, DataSource> distance_t;
  };
};

struct metric_L21_4_2D : public Metric {
  template <class T, class DataSource> struct traits {
    typedef L21_4_2D_Adaptor<T, DataSource> distance_t;
  };
};

struct metric_L21_5_2D : public Metric {
  template <class T, class DataSource> struct traits {
    typedef L21_5_2D_Adaptor<T, DataSource> distance_t;
  };
};

struct metric_L21_6_2D : public Metric {
  template <class T, class DataSource> struct traits {
    typedef L21_6_2D_Adaptor<T, DataSource> distance_t;
  };
};

struct metric_L21_7_2D : public Metric {
  template <class T, class DataSource> struct traits {
    typedef L21_7_2D_Adaptor<T, DataSource> distance_t;
  };
};

struct metric_L21_8_2D : public Metric {
  template <class T, class DataSource> struct traits {
    typedef L21_8_2D_Adaptor<T, DataSource> distance_t;
  };
};

// L21: M x 3D
struct metric_L21_M_3D : public Metric {
  template <class T, class DataSource> struct traits {
    typedef L21_M_3D_Adaptor<T, DataSource> distance_t;
  };
};

struct metric_L21_1_3D : public Metric {
  template <class T, class DataSource> struct traits {
    typedef L21_1_3D_Adaptor<T, DataSource> distance_t;
  };
};

struct metric_L21_2_3D : public Metric {
  template <class T, class DataSource> struct traits {
    typedef L21_2_3D_Adaptor<T, DataSource> distance_t;
  };
};

struct metric_L21_3_3D : public Metric {
  template <class T, class DataSource> struct traits {
    typedef L21_3_3D_Adaptor<T, DataSource> distance_t;
  };
};

struct metric_L21_4_3D : public Metric {
  template <class T, class DataSource> struct traits {
    typedef L21_4_3D_Adaptor<T, DataSource> distance_t;
  };
};


// L21: M x 4D
struct metric_L21_M_4D : public Metric {
  template <class T, class DataSource> struct traits {
    typedef L21_M_4D_Adaptor<T, DataSource> distance_t;
  };
};

struct metric_L21_1_4D : public Metric {
  template <class T, class DataSource> struct traits {
    typedef L21_1_4D_Adaptor<T, DataSource> distance_t;
  };
};

struct metric_L21_2_4D : public Metric {
  template <class T, class DataSource> struct traits {
    typedef L21_2_4D_Adaptor<T, DataSource> distance_t;
  };
};

struct metric_L21_3_4D : public Metric {
  template <class T, class DataSource> struct traits {
    typedef L21_3_4D_Adaptor<T, DataSource> distance_t;
  };
};

struct metric_L21_4_4D : public Metric {
  template <class T, class DataSource> struct traits {
    typedef L21_4_4D_Adaptor<T, DataSource> distance_t;
  };
};


//struct metric_L21_2_3D_r : public Metric {
//  template <class T, class DataSource> struct traits {
//    typedef L21_2_3D_Adaptor_row<T, DataSource> distance_t;
//  };
//};


// L12: N x 2D
struct metric_L12_M_2D : public Metric {
  template <class T, class DataSource> struct traits {
    typedef L12_M_2D_Adaptor<T, DataSource> distance_t;
  };
};

struct metric_L12_1_2D : public Metric {
  template <class T, class DataSource> struct traits {
    typedef L12_1_2D_Adaptor<T, DataSource> distance_t;
  };
};

struct metric_L12_2_2D : public Metric {
  template <class T, class DataSource> struct traits {
    typedef L12_2_2D_Adaptor<T, DataSource> distance_t;
  };
};

struct metric_L12_3_2D : public Metric {
  template <class T, class DataSource> struct traits {
    typedef L12_3_2D_Adaptor<T, DataSource> distance_t;
  };
};

struct metric_L12_4_2D : public Metric {
  template <class T, class DataSource> struct traits {
    typedef L12_4_2D_Adaptor<T, DataSource> distance_t;
  };
};

struct metric_L12_5_2D : public Metric {
  template <class T, class DataSource> struct traits {
    typedef L12_5_2D_Adaptor<T, DataSource> distance_t;
  };
};

struct metric_L12_6_2D : public Metric {
  template <class T, class DataSource> struct traits {
    typedef L12_6_2D_Adaptor<T, DataSource> distance_t;
  };
};

struct metric_L12_7_2D : public Metric {
  template <class T, class DataSource> struct traits {
    typedef L12_7_2D_Adaptor<T, DataSource> distance_t;
  };
};

struct metric_L12_8_2D : public Metric {
  template <class T, class DataSource> struct traits {
    typedef L12_8_2D_Adaptor<T, DataSource> distance_t;
  };
};


// L12: N x 3D
struct metric_L12_M_3D : public Metric {
  template <class T, class DataSource> struct traits {
    typedef L12_M_3D_Adaptor<T, DataSource> distance_t;
  };
};

struct metric_L12_1_3D : public Metric {
  template <class T, class DataSource> struct traits {
    typedef L12_1_3D_Adaptor<T, DataSource> distance_t;
  };
};

struct metric_L12_2_3D : public Metric {
  template <class T, class DataSource> struct traits {
    typedef L12_2_3D_Adaptor<T, DataSource> distance_t;
  };
};

struct metric_L12_3_3D : public Metric {
  template <class T, class DataSource> struct traits {
    typedef L12_3_3D_Adaptor<T, DataSource> distance_t;
  };
};

struct metric_L12_4_3D : public Metric {
  template <class T, class DataSource> struct traits {
    typedef L12_4_3D_Adaptor<T, DataSource> distance_t;
  };
};

// L12: N x 4D
struct metric_L12_M_4D : public Metric {
  template <class T, class DataSource> struct traits {
    typedef L12_M_4D_Adaptor<T, DataSource> distance_t;
  };
};

struct metric_L12_1_4D : public Metric {
  template <class T, class DataSource> struct traits {
    typedef L12_1_4D_Adaptor<T, DataSource> distance_t;
  };
};

struct metric_L12_2_4D : public Metric {
  template <class T, class DataSource> struct traits {
    typedef L12_2_4D_Adaptor<T, DataSource> distance_t;
  };
};

struct metric_L12_3_4D : public Metric {
  template <class T, class DataSource> struct traits {
    typedef L12_3_4D_Adaptor<T, DataSource> distance_t;
  };
};

struct metric_L12_4_4D : public Metric {
  template <class T, class DataSource> struct traits {
    typedef L12_4_4D_Adaptor<T, DataSource> distance_t;
  };
};


struct metric_Lp_ND : public Metric {
  template <class T, class DataSource> struct traits {
    typedef Lp_ND_Adaptor<T, DataSource, T> distance_t;
  };
};

struct metric_Lpq_MND : public Metric {
  template <class T, class DataSource> struct traits {
    typedef Lpq_MND_Adaptor<T, DataSource> distance_t;
  };
};



} // namespace nanoflann

#endif /* LPQ_METRIC_CPP_ */
