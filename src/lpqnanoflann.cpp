/*
<%
setup_pybind11(cfg)
%>
*/
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cstdlib>
#include <ctime>
#include <iostream>
#include <nanoflann.cpp>
#include <lpq_metric.cpp>
#include <thread>

using namespace std;
using namespace nanoflann;

using i_np_arr_t = pybind11::array_t<size_t, pybind11::array::c_style | pybind11::array::forcecast>;
using vvi = std::vector<std::vector<size_t>>;

template <typename num_t>
class AbstractKDTree {
 public:
  virtual void findNeighbors(nanoflann::KNNResultSet<num_t>, const num_t *query,
                             nanoflann::SearchParams params) = 0;
  virtual size_t radiusSearch(
      const num_t *query, num_t radius,
      std::vector<std::pair<size_t, num_t>> &ret_matches,
      nanoflann::SearchParams params) = 0;
  virtual size_t radiusSearchIdx(
      const num_t *query, num_t radius,
      std::vector<size_t> &ret_matches,
      nanoflann::SearchParams params) = 0;
  virtual void knnSearch(const num_t *query, size_t num_closest,
                         size_t *out_indices, num_t *out_distances_sq) = 0;
  virtual int saveIndex(const std::string &path) const = 0;
  virtual int loadIndex(const std::string &path) = 0;
  virtual num_t eval_pair(const num_t *a, const num_t *b, size_t size) const = 0;
  virtual num_t eval_pair(const num_t *a, const num_t *b, size_t size, num_t max_dist) const = 0;
  virtual int get_radius_exp() const = 0;
  virtual int get_radius_full_exp() const = 0;
  virtual num_t scale_radius(const num_t radius) const = 0;
  virtual num_t scale_radius_full(const num_t radius) const = 0;
  virtual void buildIndex() = 0;

  virtual void setup_lpq(int p, int q=0, int n=0) = 0;

  virtual ~AbstractKDTree(){};
};


template <typename num_t, int DIM = -1, class Distance = nanoflann::metric_L2_Simple>
struct KDTreeNumpyAdaptor : public AbstractKDTree<num_t> {
  using self_t = KDTreeNumpyAdaptor<num_t, DIM, Distance>;
  using metric_t = typename Distance::template traits<num_t, self_t>::distance_t;
  using index_t = nanoflann::KDTreeSingleIndexAdaptor<metric_t, self_t, DIM>;
  using f_np_arr_t = pybind11::array_t<num_t, pybind11::array::c_style | pybind11::array::forcecast>;


  index_t *index;
  const num_t *buf;
  size_t n_points;
  size_t dim;

  KDTreeNumpyAdaptor(const f_np_arr_t &points, const int leaf_max_size = 10) {
    buf = points.template unchecked<2>().data(0, 0);
    n_points = points.shape(0);
    dim = points.shape(1);
    index = new index_t(dim, *this, nanoflann::KDTreeSingleIndexAdaptorParams(leaf_max_size));
  }

  ~KDTreeNumpyAdaptor() { delete index; }
  void buildIndex() { index->buildIndex(); }

  void findNeighbors(nanoflann::KNNResultSet<num_t> result_set,
                     const num_t *query, nanoflann::SearchParams params) {
    index->findNeighbors(result_set, query, params);
  }
  void knnSearch(const num_t *query, size_t num_closest, size_t *out_indices,
                 num_t *out_distances_sq) {
    index->knnSearch(query, num_closest, out_indices, out_distances_sq);
  }

  size_t radiusSearch(const num_t *query, num_t radius,
                      std::vector<std::pair<size_t, num_t>> &ret_matches,
                      nanoflann::SearchParams params) {
    return index->radiusSearch(query, radius, ret_matches, params);
  }

  size_t radiusSearchIdx(const num_t *query, num_t radius,
                         std::vector<size_t> &ret_matches,
                         nanoflann::SearchParams params) {
    return index->radiusSearchIdx(query, radius, ret_matches, params);
  }

  const self_t &derived() const { return *this; }
  self_t &derived() { return *this; }

  inline size_t kdtree_get_point_count() const { return n_points; }

  inline num_t kdtree_get_pt(const size_t idx, const size_t dim) const {
    return buf[idx * this->dim + dim];
  }

  inline const num_t* kdtree_get_row(const size_t idx) const {
    return &buf[idx * this->dim];
  }

  inline num_t eval_pair(const num_t *a, const num_t *b, size_t size) const {
    return index->distance.eval_pair(a, b, size);
  }

  inline num_t eval_pair(const num_t *a, const num_t *b, size_t size, num_t max_dist) const {
    return index->distance.eval_pair(a, b, size, max_dist);
  }

  inline int get_radius_exp() const {
    return index->distance.dist_exponent;
  }

  inline int get_radius_full_exp() const {
    return index->distance.pair_exponent;
  }

  inline void setup_lpq(int p, int q=0, int n=0) {
    // method to initialize the lpq distance metric
    //  when p or q > 2, or ndim > 4d
    index->distance.setup_lpq(p, q, n);
  }

  inline num_t scale_radius(const num_t radius) const {
    return std::pow(radius, this->get_radius_exp());
  }

  inline num_t scale_radius_full(const num_t radius) const {
    return std::pow(radius, this->get_radius_full_exp());
  }

  template <class BBOX>
  bool kdtree_get_bbox(BBOX &) const {
    return false;
  }

  int saveIndex(const std::string &path) const {
    FILE *f = fopen(path.c_str(), "wb");
    if (!f) {
      throw std::runtime_error("Error writing index file!");
    }
    index->saveIndex(f);
    int ret_val = fclose(f);
    return ret_val;
  }

  int loadIndex(const std::string &path) {
    FILE *f = fopen(path.c_str(), "rb");
    if (!f) {
      throw std::runtime_error("Error reading index file!");
    }
    index->loadIndex(f);
    return fclose(f);
  }
};

template <typename num_t>
class KDTree {
 public:
  using f_np_arr_t = pybind11::array_t<num_t, pybind11::array::c_style | pybind11::array::forcecast>;

  KDTree(size_t n_neighbors = 10, size_t leaf_size = 10, std::string metric = "l2", num_t radius = 1.0f);
  ~KDTree() { delete index; }
  void fit(f_np_arr_t points, std::string index_path, size_t ndim = 1);

  // kneighbors search
  void kneighbors(f_np_arr_t array, size_t n_neighbors);
  void kneighbors_multithreaded(f_np_arr_t array, size_t n_neighbors, size_t nThreads = 1);


  // radius search
  void radius_neighbors_idx(f_np_arr_t, num_t radius = 1.0f);
  void radius_neighbors_idx_dists(f_np_arr_t, num_t radius = 1.0f);
  void radius_neighbors_idx_dists_full(f_np_arr_t arr, f_np_arr_t full_tree,
    f_np_arr_t full_arr, num_t radius = 1.0f, num_t radius_full = 1.0f);
  void radius_neighbors_idx_dists_full_multithreaded(f_np_arr_t arr, f_np_arr_t full_tree,
    f_np_arr_t full_arr, num_t radius = 1.0f, num_t radius_full = 1.0f, size_t nThreads = 1);
  void radius_neighbors_idx_multithreaded(
      f_np_arr_t array, num_t radius = 1.0f, size_t nThreads = 1);
  void radius_neighbors_idx_dists_multithreaded(
      f_np_arr_t array, num_t radius = 1.0f, size_t nThreads = 1);

  int save_index(const std::string &path);

  AbstractKDTree<num_t> *index;

  // result getter
  i_np_arr_t getResultLenghts();
  i_np_arr_t getResultIndicesPtr();
  i_np_arr_t getResultIndicesRow();
  i_np_arr_t getResultIndicesCol();
  f_np_arr_t getResultDists();
  f_np_arr_t getResultRawDists();

  size_t n_neighbors;
  size_t leaf_size;
  std::string metric;
  num_t radius;

  std::vector<size_t> m_nbmatches;
  std::vector<std::vector<size_t>> m_indices;
  std::vector<std::vector<num_t>> m_dists;
  int dists_exponent = 0;
};


template <typename num_t>
KDTree<num_t>::KDTree(size_t n_neighbors, size_t leaf_size, std::string metric,
                      num_t radius)
    : n_neighbors(n_neighbors),
      leaf_size(leaf_size),
      metric(metric),
      radius(radius) {}


template <typename num_t>
void KDTree<num_t>::fit(f_np_arr_t points, std::string index_path, size_t ndim) {
  // Dynamic template instantiation for the popular use cases
  // separate in   ndim x mdim = total_dim
  const int total_dim = points.shape(1);
  const int mdim = points.shape(1) / ndim;

  if (total_dim % ndim > 0 )
    throw std::runtime_error("Error: total_dim != nb_values * ndim");

  if (metric[0] != 'l')
    throw std::runtime_error("Error: metric should start with  'l'");

  // Change  "lpq" string to  integers p and q
  int p;
  int q;
  switch (metric.length()) {
    case 1:
        throw std::runtime_error("Error: should have at least 2 characters");
      break;
    case 2:
        p = metric[1] - '0';
        q = p;
      break;
    case 3:
        p = metric[1] - '0';
        q = metric[2] - '0';
      break;
    default:
        throw std::runtime_error("Error: should have maximum 3 characters");
      break;
  }

  if (p==0 || q==0){
    throw std::runtime_error("Error: L0 is not a metric");
  }
  else if (p==q){ // if only P is given or if P=Q
    if (p==1) { // L1 | L11
      switch (total_dim) {
        case 1:
          index = new KDTreeNumpyAdaptor<num_t, 1, nanoflann::metric_L1_1D>(points, leaf_size);
          break;
        case 2:
          index = new KDTreeNumpyAdaptor<num_t, 2, nanoflann::metric_L1_2D>(points, leaf_size);
          break;
        case 3:
          index = new KDTreeNumpyAdaptor<num_t, 3, nanoflann::metric_L1_3D>(points, leaf_size);
          break;
        case 4:
          index = new KDTreeNumpyAdaptor<num_t, 4, nanoflann::metric_L1_4D>(points, leaf_size);
          break;
        case 5:
          index = new KDTreeNumpyAdaptor<num_t, 5, nanoflann::metric_L1_5D>(points, leaf_size);
          break;
        case 6:
          index = new KDTreeNumpyAdaptor<num_t, 6, nanoflann::metric_L1_6D>(points, leaf_size);
          break;
        case 7:
          index = new KDTreeNumpyAdaptor<num_t, 7, nanoflann::metric_L1_7D>(points, leaf_size);
          break;
        case 8:
          index = new KDTreeNumpyAdaptor<num_t, 8, nanoflann::metric_L1_8D>(points, leaf_size);
          break;
        default:
          index = new KDTreeNumpyAdaptor<num_t, -1, nanoflann::metric_L1_ND>(points, leaf_size);
          break;
      }
    }
    else if (p==2) { // L2 | L22
      switch (total_dim) {
        case 1:
          index = new KDTreeNumpyAdaptor<num_t, 1, nanoflann::metric_L2_1D>(points, leaf_size);
          break;
        case 2:
          index = new KDTreeNumpyAdaptor<num_t, 2, nanoflann::metric_L2_2D>(points, leaf_size);
          break;
        case 3:
          index = new KDTreeNumpyAdaptor<num_t, 3, nanoflann::metric_L2_3D>(points, leaf_size);
          break;
        case 4:
          index = new KDTreeNumpyAdaptor<num_t, 4, nanoflann::metric_L2_4D>(points, leaf_size);
          break;
        case 5:
          index = new KDTreeNumpyAdaptor<num_t, 5, nanoflann::metric_L2_5D>(points, leaf_size);
          break;
        case 6:
          index = new KDTreeNumpyAdaptor<num_t, 6, nanoflann::metric_L2_6D>(points, leaf_size);
          break;
        case 7:
          index = new KDTreeNumpyAdaptor<num_t, 7, nanoflann::metric_L2_7D>(points, leaf_size);
          break;
        case 8:
          index = new KDTreeNumpyAdaptor<num_t, 8, nanoflann::metric_L2_8D>(points, leaf_size);
          break;
        default:
          index = new KDTreeNumpyAdaptor<num_t, -1, nanoflann::metric_L2_ND>(points, leaf_size);
          break;
      }
    }
    else {
      index = new KDTreeNumpyAdaptor<num_t, -1, nanoflann::metric_Lp_ND>(points, leaf_size);
    }
  }
  else if (p==2 && q==1) { // L21
    switch (ndim) {
      case 1:
        throw std::runtime_error("Error: L21 with ndim==1, use L2 distance");
        break;
      case 2:
        switch (mdim) {
          case 1:
            index = new KDTreeNumpyAdaptor<num_t, 2, nanoflann::metric_L21_1_2D>(points, leaf_size);
            break;
          case 2:
            index = new KDTreeNumpyAdaptor<num_t, 4, nanoflann::metric_L21_2_2D>(points, leaf_size);
            break;
          case 3:
            index = new KDTreeNumpyAdaptor<num_t, 6, nanoflann::metric_L21_3_2D>(points, leaf_size);
            break;
          case 4:
            index = new KDTreeNumpyAdaptor<num_t, 8, nanoflann::metric_L21_4_2D>(points, leaf_size);
            break;
          case 5:
            index = new KDTreeNumpyAdaptor<num_t, 10, nanoflann::metric_L21_5_2D>(points, leaf_size);
            break;
          case 6:
            index = new KDTreeNumpyAdaptor<num_t, 12, nanoflann::metric_L21_6_2D>(points, leaf_size);
            break;
          case 7:
            index = new KDTreeNumpyAdaptor<num_t, 14, nanoflann::metric_L21_7_2D>(points, leaf_size);
            break;
          case 8:
            index = new KDTreeNumpyAdaptor<num_t, 16, nanoflann::metric_L21_8_2D>(points, leaf_size);
            break;
          default:
            index = new KDTreeNumpyAdaptor<num_t, -1, nanoflann::metric_L21_M_2D>(points, leaf_size);
            break;
        }
        break;
      case 3:
        switch (mdim) {
          case 1:
            index = new KDTreeNumpyAdaptor<num_t, 3, nanoflann::metric_L21_1_3D>(points, leaf_size);
            break;
          case 2:
            index = new KDTreeNumpyAdaptor<num_t, 6, nanoflann::metric_L21_2_3D>(points, leaf_size);
            break;
          case 3:
            index = new KDTreeNumpyAdaptor<num_t, 9, nanoflann::metric_L21_3_3D>(points, leaf_size);
            break;
          case 4:
            index = new KDTreeNumpyAdaptor<num_t, 12, nanoflann::metric_L21_4_3D>(points, leaf_size);
            break;
          default:
            index = new KDTreeNumpyAdaptor<num_t, -1, nanoflann::metric_L21_M_3D>(points, leaf_size);
            break;
        }
        break;
      case 4:
        switch (mdim) {
          case 1:
            index = new KDTreeNumpyAdaptor<num_t, 4, nanoflann::metric_L21_1_4D>(points, leaf_size);
            break;
          case 2:
            index = new KDTreeNumpyAdaptor<num_t, 8, nanoflann::metric_L21_2_4D>(points, leaf_size);
            break;
          case 3:
            index = new KDTreeNumpyAdaptor<num_t, 12, nanoflann::metric_L21_3_4D>(points, leaf_size);
            break;
          case 4:
            index = new KDTreeNumpyAdaptor<num_t, 16, nanoflann::metric_L21_4_4D>(points, leaf_size);
            break;
          default:
            index = new KDTreeNumpyAdaptor<num_t, -1, nanoflann::metric_L21_M_4D>(points, leaf_size);
            break;
        }
        break;
      default:
        // TODO optimize for L21 in Nd ?
        index = new KDTreeNumpyAdaptor<num_t, -1, nanoflann::metric_Lpq_MND>(points, leaf_size);
        break;
    }
  }
  else if (p==1 && q==2) { // L12
    switch (ndim) {
      case 1:
        throw std::runtime_error("Error: L12 with ndim==1, use L1 distance");
        break;
      case 2:
        switch (mdim) {
          case 1:
            index = new KDTreeNumpyAdaptor<num_t, 2, nanoflann::metric_L12_1_2D>(points, leaf_size);
            break;
          case 2:
            index = new KDTreeNumpyAdaptor<num_t, 4, nanoflann::metric_L12_2_2D>(points, leaf_size);
            break;
          case 3:
            index = new KDTreeNumpyAdaptor<num_t, 6, nanoflann::metric_L12_3_2D>(points, leaf_size);
            break;
          case 4:
            index = new KDTreeNumpyAdaptor<num_t, 8, nanoflann::metric_L12_4_2D>(points, leaf_size);
            break;
          case 5:
            index = new KDTreeNumpyAdaptor<num_t, 10, nanoflann::metric_L12_5_2D>(points, leaf_size);
            break;
          case 6:
            index = new KDTreeNumpyAdaptor<num_t, 12, nanoflann::metric_L12_6_2D>(points, leaf_size);
            break;
          case 7:
            index = new KDTreeNumpyAdaptor<num_t, 14, nanoflann::metric_L12_7_2D>(points, leaf_size);
            break;
          case 8:
            index = new KDTreeNumpyAdaptor<num_t, 16, nanoflann::metric_L12_8_2D>(points, leaf_size);
            break;
          default:
            index = new KDTreeNumpyAdaptor<num_t, -1, nanoflann::metric_L12_M_2D>(points, leaf_size);
            break;
        }
        break;
      case 3:
        switch (mdim) {
          case 1:
            index = new KDTreeNumpyAdaptor<num_t, 3, nanoflann::metric_L12_1_3D>(points, leaf_size);
            break;
          case 2:
            index = new KDTreeNumpyAdaptor<num_t, 6, nanoflann::metric_L12_2_3D>(points, leaf_size);
            break;
          case 3:
            index = new KDTreeNumpyAdaptor<num_t, 9, nanoflann::metric_L12_3_3D>(points, leaf_size);
            break;
          case 4:
            index = new KDTreeNumpyAdaptor<num_t, 12, nanoflann::metric_L12_4_3D>(points, leaf_size);
            break;
          default:
            index = new KDTreeNumpyAdaptor<num_t, -1, nanoflann::metric_L12_M_3D>(points, leaf_size);
            break;
        }
        break;
      case 4:
        switch (mdim) {
          case 1:
            index = new KDTreeNumpyAdaptor<num_t, 4, nanoflann::metric_L12_1_4D>(points, leaf_size);
            break;
          case 2:
            index = new KDTreeNumpyAdaptor<num_t, 8, nanoflann::metric_L12_2_4D>(points, leaf_size);
            break;
          case 3:
            index = new KDTreeNumpyAdaptor<num_t, 12, nanoflann::metric_L12_3_4D>(points, leaf_size);
            break;
          case 4:
            index = new KDTreeNumpyAdaptor<num_t, 16, nanoflann::metric_L12_4_4D>(points, leaf_size);
            break;
          default:
            index = new KDTreeNumpyAdaptor<num_t, -1, nanoflann::metric_L12_M_4D>(points, leaf_size);
            break;
        }
        break;
      default:
        // TODO optimize for L12 in Nd ?
        index = new KDTreeNumpyAdaptor<num_t, -1, nanoflann::metric_Lpq_MND>(points, leaf_size);
        break;
    }
  }
  else{
    // General Lpq distance where p != q
    index = new KDTreeNumpyAdaptor<num_t, -1, nanoflann::metric_Lpq_MND>(points, leaf_size);
  }
  // TODO enforce at initialization
  index->setup_lpq(p, q, ndim);


  if (index_path.size()) {
    index->loadIndex(index_path);
  } else {
    index->buildIndex();
  }

}

template <typename num_t>
void KDTree<num_t>::kneighbors(f_np_arr_t array, size_t n_neighbors) {
  auto mat = array.template unchecked<2>();
  const num_t *query_data = mat.data(0, 0);
  size_t n_points = mat.shape(0);
  size_t dim = mat.shape(1);

  this->m_nbmatches.clear();
  this->m_nbmatches.resize(n_points, n_neighbors);
  this->m_indices.resize(n_points);
  this->m_dists.resize(n_points);
  this->dists_exponent = index->get_radius_exp();

  for (size_t i = 0; i < n_points; i++) {
      const num_t *query_point = &query_data[i * dim];
      m_indices[i].resize(n_neighbors);
      m_dists[i].resize(n_neighbors);
      index->knnSearch(query_point, n_neighbors, this->m_indices[i].data(), this->m_dists[i].data());
  }

  return;
}

template <typename num_t>
void KDTree<num_t>::kneighbors_multithreaded(f_np_arr_t array, size_t n_neighbors, size_t nThreads) {
  auto mat = array.template unchecked<2>();
  const num_t *query_data = mat.data(0, 0);
  size_t n_points = mat.shape(0);
  size_t dim = mat.shape(1);

  this->m_nbmatches.resize(n_points, n_neighbors);
  this->m_indices.resize(n_points);
  this->m_dists.resize(n_points);
  this->dists_exponent = index->get_radius_exp();

  auto searchBatch = [&](size_t startIdx, size_t endIdx) {
    for (size_t i = startIdx; i < endIdx; i++) {
      const num_t *query_point = &query_data[i * dim];
      m_indices[i].resize(n_neighbors);
      m_dists[i].resize(n_neighbors);
      index->knnSearch(query_point, n_neighbors, this->m_indices[i].data(), this->m_dists[i].data());
    }
  };

  std::vector<std::thread> threadPool;
  size_t batchSize = std::ceil(static_cast<float>(n_points) / nThreads);
  for (size_t i = 0; i < nThreads; i++) {
    size_t startIdx = i * batchSize;
    size_t endIdx = (i + 1) * batchSize;
    endIdx = std::min(endIdx, n_points);
    threadPool.push_back(std::thread(searchBatch, startIdx, endIdx));
  }
  for (auto &t : threadPool) {
    t.join();
  }

  return;
}

template <typename num_t>
void KDTree<num_t>::radius_neighbors_idx(
    f_np_arr_t array, num_t radius) {
  const auto mat = array.template unchecked<2>();
  const num_t *query_data = mat.data(0, 0);
  const size_t n_points = mat.shape(0);
  const size_t dim = mat.shape(1);

  const num_t search_radius = index->scale_radius(radius);

  this->m_nbmatches.resize(n_points);
  this->m_indices.resize(n_points);
  this->m_dists.clear();
  this->dists_exponent = 0;

  for (size_t i = 0; i < n_points; i++) {
    this->m_nbmatches[i] = index->radiusSearchIdx(
        &query_data[i * dim], search_radius, this->m_indices[i], nanoflann::SearchParams());
  }
  return;
}

template <typename num_t>
void KDTree<num_t>::radius_neighbors_idx_dists(f_np_arr_t array, num_t radius) {
  const auto mat = array.template unchecked<2>();
  const num_t *query_data = mat.data(0, 0);
  const size_t n_points = mat.shape(0);
  const size_t dim = mat.shape(1);


  const num_t search_radius = index->scale_radius(radius);
  std::vector<std::pair<size_t, num_t>> ret_matches;
  this->m_nbmatches.resize(n_points);
  this->m_indices.resize(n_points);
  this->m_dists.resize(n_points);
  this->dists_exponent = index->get_radius_exp();

  for (size_t i = 0; i < n_points; i++) {
    const size_t nb_match = index->radiusSearch(
        &query_data[i * dim], search_radius, ret_matches, nanoflann::SearchParams());
    this->m_nbmatches[i] = nb_match;

    this->m_indices[i].resize(nb_match);
    this->m_dists[i].resize(nb_match);
    for (size_t j = 0; j < nb_match; j++) {
      this->m_indices[i][j] = ret_matches[j].first;
      this->m_dists[i][j] = ret_matches[j].second;
    }
  }
  return;
}

template <typename num_t>
void KDTree<num_t>::radius_neighbors_idx_multithreaded(f_np_arr_t array, num_t radius, size_t nThreads) {

  const auto mat = array.template unchecked<2>();
  const num_t *query_data = mat.data(0, 0);
  const size_t n_points = mat.shape(0);
  const size_t dim = mat.shape(1);


  const num_t search_radius = index->scale_radius(radius);

  this->m_nbmatches.resize(n_points);
  this->m_indices.resize(n_points);
  this->m_dists.clear();
  this->dists_exponent = index->get_radius_exp();

  auto searchBatch = [&](size_t startIdx, size_t endIdx) {
    for (size_t i = startIdx; i < endIdx; i++) {
      this->m_nbmatches[i] = index->radiusSearchIdx(
          &query_data[i * dim], search_radius, this->m_indices[i], nanoflann::SearchParams());
    }
  };

  std::vector<std::thread> threadPool;
  size_t batchSize = std::ceil(static_cast<float>(n_points) / nThreads);
  for (size_t i = 0; i < nThreads; i++) {
    size_t startIdx = i * batchSize;
    size_t endIdx = (i + 1) * batchSize;
    endIdx = std::min(endIdx, n_points);
    threadPool.push_back(std::thread(searchBatch, startIdx, endIdx));
  }
  for (auto &t : threadPool) {
    t.join();
  }

  return;
}


template <typename num_t>
void KDTree<num_t>::radius_neighbors_idx_dists_multithreaded(f_np_arr_t array, num_t radius, size_t nThreads) {
  const auto mat = array.template unchecked<2>();
  const num_t *query_data = mat.data(0, 0);
  const size_t n_points = mat.shape(0);
  const size_t dim = mat.shape(1);


  const num_t search_radius = index->scale_radius(radius);
  std::vector<std::pair<size_t, num_t>> ret_matches;

  this->m_nbmatches.resize(n_points);
  this->m_indices.resize(n_points);
  this->m_dists.resize(n_points);
  this->dists_exponent = index->get_radius_exp();

  auto searchBatch = [&](size_t startIdx, size_t endIdx) {
    std::vector<std::pair<size_t, num_t>> ret_matches;
    for (size_t i = startIdx; i < endIdx; i++) {
      const size_t nb_match = index->radiusSearch(&query_data[i * dim], search_radius, ret_matches, nanoflann::SearchParams());

      this->m_nbmatches[i] = nb_match;
      this->m_indices[i].resize(nb_match);
      this->m_dists[i].resize(nb_match);
      for (size_t j = 0; j < nb_match; j++) {
        this->m_indices[i][j] = ret_matches[j].first;
        this->m_dists[i][j] = ret_matches[j].second;
      }
    }
  };

  std::vector<std::thread> threadPool;
  size_t batchSize = std::ceil(static_cast<float>(n_points) / nThreads);
  for (size_t i = 0; i < nThreads; i++) {
    size_t startIdx = i * batchSize;
    size_t endIdx = (i + 1) * batchSize;
    endIdx = std::min(endIdx, n_points);
    threadPool.push_back(std::thread(searchBatch, startIdx, endIdx));
  }
  for (auto &t : threadPool) {
    t.join();
  }

  return;
}



template <typename num_t>
int KDTree<num_t>::save_index(const std::string &path) {
  return index->saveIndex(path);
}


// ESO Getter for results in python
template <typename num_t>
i_np_arr_t KDTree<num_t>::getResultLenghts(){
  return pybind11::array(this->m_nbmatches.size(), this->m_nbmatches.data());
}

template <typename num_t>
i_np_arr_t KDTree<num_t>::getResultIndicesPtr(){
  const size_t n_points = this->m_nbmatches.size();
  std::vector<size_t>* seq_ptr = new std::vector<size_t>(n_points + 1);

  // reformating in a single array
  (*seq_ptr)[0] = 0;
  for (size_t i = 0; i < n_points; ++i) {
    (*seq_ptr)[i+1] = (*seq_ptr)[i] + this->m_nbmatches[i];
  }

  auto capsule = pybind11::capsule(seq_ptr, [](void* p) { delete reinterpret_cast<std::vector<size_t>*>(p); });
  return pybind11::array(seq_ptr->size(),seq_ptr->data(), capsule);
}

template <typename num_t>
i_np_arr_t KDTree<num_t>::getResultIndicesRow(){
  const size_t n_points = this->m_nbmatches.size();
  const size_t total_nb_match = std::accumulate(this->m_nbmatches.begin(), this->m_nbmatches.end(), 0);
  std::vector<size_t>* seq_ptr = new std::vector<size_t>(total_nb_match);
  size_t d = 0;

  // reformating in a single array
  for (size_t i = 0; i < n_points; ++i) {
    const size_t nb_match = this->m_nbmatches[i];
    for (size_t j = 0; j < nb_match; ++j) {
      (*seq_ptr)[d++] = i;
    }
  }

  auto capsule = pybind11::capsule(seq_ptr, [](void* p) { delete reinterpret_cast<std::vector<size_t>*>(p); });
  return pybind11::array(seq_ptr->size(),seq_ptr->data(), capsule);
}

template <typename num_t>
i_np_arr_t KDTree<num_t>::getResultIndicesCol(){
  const size_t n_points = this->m_nbmatches.size();
  const size_t total_nb_match = std::accumulate(this->m_nbmatches.begin(), this->m_nbmatches.end(), 0);
  std::vector<size_t>* seq_ptr = new std::vector<size_t>(total_nb_match);
  size_t d = 0;

  // reformating in a single array
  for (size_t i = 0; i < n_points; ++i) {
    const size_t nb_match = this->m_nbmatches[i];
    for (size_t j = 0; j < nb_match; ++j) {
      (*seq_ptr)[d++] = this->m_indices[i][j];
    }
  }

  auto capsule = pybind11::capsule(seq_ptr, [](void* p) { delete reinterpret_cast<std::vector<size_t>*>(p); });
  return pybind11::array(seq_ptr->size(),seq_ptr->data(), capsule);
}


template <typename num_t>
pybind11::array_t<num_t, pybind11::array::c_style | pybind11::array::forcecast> KDTree<num_t>::getResultDists(){
  const size_t n_points = this->m_nbmatches.size();
  const size_t total_nb_match = std::accumulate(this->m_nbmatches.begin(), this->m_nbmatches.end(), 0);
  std::vector<num_t>* seq_ptr = new std::vector<num_t>(total_nb_match);
  size_t d = 0;

  // reformating in a single array
  if(this->dists_exponent < 1){
      throw std::runtime_error("Error: dists_exponent < 1, need to be set for the chosen distance");
  }
  else if(this->dists_exponent == 1){
    for (size_t i = 0; i < n_points; ++i) {
      const size_t nb_match = this->m_nbmatches[i];
      for (size_t j = 0; j < nb_match; ++j) {
        (*seq_ptr)[d++] = this->m_dists[i][j];
      }
    }
  }
  else if(this->dists_exponent == 2){
    for (size_t i = 0; i < n_points; ++i) {
      const size_t nb_match = this->m_nbmatches[i];
      for (size_t j = 0; j < nb_match; ++j) {
        (*seq_ptr)[d++] = std::sqrt(this->m_dists[i][j]);
      }
    }
  }
  else{
    for (size_t i = 0; i < n_points; ++i) {
      const size_t nb_match = this->m_nbmatches[i];
      for (size_t j = 0; j < nb_match; ++j) {
        (*seq_ptr)[d++] = std::pow(this->m_dists[i][j], 1.0/this->dists_exponent);
      }
    }
  }

  auto capsule = pybind11::capsule(seq_ptr, [](void* p) { delete reinterpret_cast<std::vector<num_t>*>(p); });
  return pybind11::array(seq_ptr->size(),seq_ptr->data(), capsule);
}


template <typename num_t>
pybind11::array_t<num_t, pybind11::array::c_style | pybind11::array::forcecast> KDTree<num_t>::getResultRawDists(){
  const size_t n_points = this->m_nbmatches.size();
  const size_t total_nb_match = std::accumulate(this->m_nbmatches.begin(), this->m_nbmatches.end(), 0);
  std::vector<num_t>* seq_ptr = new std::vector<num_t>(total_nb_match);
  size_t d = 0;

  // reformating in a single array
  for (size_t i = 0; i < n_points; ++i) {
    const size_t nb_match = this->m_nbmatches[i];
    for (size_t j = 0; j < nb_match; ++j) {
      (*seq_ptr)[d++] = this->m_dists[i][j];
    }
  }

  auto capsule = pybind11::capsule(seq_ptr, [](void* p) { delete reinterpret_cast<std::vector<num_t>*>(p); });
  return pybind11::array(seq_ptr->size(),seq_ptr->data(), capsule);
}

template <typename num_t>
void KDTree<num_t>::radius_neighbors_idx_dists_full(f_np_arr_t array, f_np_arr_t full_tree, f_np_arr_t full_array, num_t radius, num_t radius_full) {
  const auto mat = array.template unchecked<2>();
  const auto fmat_t = full_tree.template unchecked<2>();
  const auto fmat = full_array.template unchecked<2>();
  const num_t *query_data = mat.data(0, 0);
  const num_t *query_full = fmat.data(0, 0);
  const num_t *query_fullt = fmat_t.data(0, 0);
  const size_t n_points = mat.shape(0);
  const size_t dim = mat.shape(1);

  const size_t full_dim = fmat.shape(1);

  num_t full_dist;

  const num_t search_radius = index->scale_radius(radius);
  const num_t search_radius_full = index->scale_radius_full(radius_full);

  std::vector<std::pair<size_t, num_t>> ret_matches;
  this->m_nbmatches.resize(n_points);
  this->m_indices.resize(n_points);
  this->m_dists.resize(n_points);
  this->dists_exponent = index->get_radius_full_exp();

  for (size_t i = 0; i < n_points; i++) {
    const size_t nb_match = index->radiusSearch(
        &query_data[i * dim], search_radius, ret_matches, nanoflann::SearchParams());

    for (size_t j = 0; j < nb_match; j++) {
      full_dist = index->eval_pair(&query_full[i * full_dim], &query_fullt[ret_matches[j].first * full_dim], full_dim);

      if (full_dist < search_radius_full){
        this->m_indices[i].push_back(ret_matches[j].first);
        this->m_dists[i].push_back(full_dist);
      }
    }
    this->m_nbmatches[i] = this->m_indices[i].size();
  }
  return;
}


template <typename num_t>
void KDTree<num_t>::radius_neighbors_idx_dists_full_multithreaded(f_np_arr_t array, f_np_arr_t full_tree, f_np_arr_t full_array, num_t radius, num_t radius_full, size_t nThreads) {
  // reset search results
  const auto mat = array.template unchecked<2>();
  const auto fmat_t = full_tree.template unchecked<2>();
  const auto fmat = full_array.template unchecked<2>();
  const num_t *query_data = mat.data(0, 0);
  const num_t *query_full = fmat.data(0, 0);
  const num_t *query_fullt = fmat_t.data(0, 0);
  const size_t n_points = mat.shape(0);
  const size_t dim = mat.shape(1);

  const size_t full_dim = fmat.shape(1);

  const num_t search_radius = index->scale_radius(radius);
  const num_t search_radius_full = index->scale_radius_full(radius_full);

  this->m_nbmatches.resize(n_points);
  this->m_indices.resize(n_points);
  this->m_dists.resize(n_points);
  this->dists_exponent = index->get_radius_full_exp();

  auto searchBatch = [&](size_t startIdx, size_t endIdx) {
    std::vector<std::pair<size_t, num_t>> ret_matches;
    num_t full_dist;
    for (size_t i = startIdx; i < endIdx; i++) {
      const size_t nb_match = index->radiusSearch(&query_data[i * dim], search_radius, ret_matches, nanoflann::SearchParams());

      this->m_indices[i].resize(0);
      this->m_dists[i].clear();
      for (size_t j = 0; j < nb_match; j++) {
        full_dist = index->eval_pair(&query_full[i * full_dim], &query_fullt[ret_matches[j].first * full_dim], full_dim);

        if (full_dist < search_radius_full){
          this->m_indices[i].push_back(ret_matches[j].first);
          this->m_dists[i].push_back(full_dist);
        }
      }
      this->m_nbmatches[i] = this->m_indices[i].size();
    }
  };

  std::vector<std::thread> threadPool;
  size_t batchSize = std::ceil(static_cast<float>(n_points) / nThreads);
  for (size_t i = 0; i < nThreads; i++) {
    size_t startIdx = i * batchSize;
    size_t endIdx = (i + 1) * batchSize;
    endIdx = std::min(endIdx, n_points);
    threadPool.push_back(std::thread(searchBatch, startIdx, endIdx));
  }
  for (auto &t : threadPool) {
    t.join();
  }

  return;
}


PYBIND11_MODULE(nanoflann_ext, m) {
  pybind11::class_<KDTree<float>>(m, "KDTree32")
      .def(pybind11::init<size_t, size_t, std::string, float>())
      .def("fit", &KDTree<float>::fit)
      .def("kneighbors", &KDTree<float>::kneighbors)
      .def("kneighbors_multithreaded", &KDTree<float>::kneighbors_multithreaded)
      .def("radius_neighbors_idx", &KDTree<float>::radius_neighbors_idx)
      .def("radius_neighbors_idx_dists", &KDTree<float>::radius_neighbors_idx_dists)
      .def("radius_neighbors_idx_multithreaded", &KDTree<float>::radius_neighbors_idx_multithreaded)
      .def("radius_neighbors_idx_dists_multithreaded", &KDTree<float>::radius_neighbors_idx_dists_multithreaded)
      .def("radius_neighbors_idx_dists_full", &KDTree<float>::radius_neighbors_idx_dists_full)
      .def("radius_neighbors_idx_dists_full_multithreaded", &KDTree<float>::radius_neighbors_idx_dists_full_multithreaded)
      .def("getResultLenghts", &KDTree<float>::getResultLenghts)
      .def("getResultIndicesPtr", &KDTree<float>::getResultIndicesPtr)
      .def("getResultIndicesRow", &KDTree<float>::getResultIndicesRow)
      .def("getResultIndicesCol", &KDTree<float>::getResultIndicesCol)
      .def("getResultDists", &KDTree<float>::getResultDists)
      .def("getResultRawDists", &KDTree<float>::getResultRawDists)
      .def("save_index", &KDTree<float>::save_index);

  pybind11::class_<KDTree<double>>(m, "KDTree64")
      .def(pybind11::init<size_t, size_t, std::string, float>())
      .def("fit", &KDTree<double>::fit)
      .def("kneighbors", &KDTree<double>::kneighbors)
      .def("kneighbors_multithreaded", &KDTree<double>::kneighbors_multithreaded)
      .def("radius_neighbors_idx", &KDTree<double>::radius_neighbors_idx)
      .def("radius_neighbors_idx_dists", &KDTree<double>::radius_neighbors_idx_dists)
      .def("radius_neighbors_idx_multithreaded", &KDTree<double>::radius_neighbors_idx_multithreaded)
      .def("radius_neighbors_idx_dists_multithreaded", &KDTree<double>::radius_neighbors_idx_dists_multithreaded)
      .def("radius_neighbors_idx_dists_full", &KDTree<double>::radius_neighbors_idx_dists_full)
      .def("radius_neighbors_idx_dists_full_multithreaded", &KDTree<double>::radius_neighbors_idx_dists_full_multithreaded)
      .def("getResultLenghts", &KDTree<double>::getResultLenghts)
      .def("getResultIndicesPtr", &KDTree<double>::getResultIndicesPtr)
      .def("getResultIndicesRow", &KDTree<double>::getResultIndicesRow)
      .def("getResultIndicesCol", &KDTree<double>::getResultIndicesCol)
      .def("getResultDists", &KDTree<double>::getResultDists)
      .def("getResultRawDists", &KDTree<double>::getResultRawDists)
      .def("save_index", &KDTree<double>::save_index);

}
