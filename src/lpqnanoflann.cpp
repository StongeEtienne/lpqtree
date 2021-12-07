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
#include <nanoflann.hpp>
#include <thread>

using namespace std;
using namespace nanoflann;

using i_numpy_array_t = pybind11::array_t<size_t, pybind11::array::c_style | pybind11::array::forcecast>;
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
  virtual void buildIndex() = 0;
  virtual ~AbstractKDTree(){};
};


template <typename num_t, int DIM = -1, class Distance = nanoflann::metric_L2_Simple>
struct KDTreeNumpyAdaptor : public AbstractKDTree<num_t> {
  using self_t = KDTreeNumpyAdaptor<num_t, DIM, Distance>;
  using metric_t = typename Distance::template traits<num_t, self_t>::distance_t;
  using index_t = nanoflann::KDTreeSingleIndexAdaptor<metric_t, self_t, DIM>;
  using f_numpy_array_t = pybind11::array_t<num_t, pybind11::array::c_style | pybind11::array::forcecast>;

  index_t *index;
  const num_t *buf;
  size_t n_points, dim;

  KDTreeNumpyAdaptor(const f_numpy_array_t &points, const int leaf_max_size = 10) {
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
  using f_numpy_array_t = pybind11::array_t<num_t, pybind11::array::c_style | pybind11::array::forcecast>;

  KDTree(size_t n_neighbors = 10, size_t leaf_size = 10, std::string metric = "l2", num_t radius = 1.0f);
  void fit(f_numpy_array_t points, std::string index_path) {
    // Dynamic template instantiation for the popular use cases

    if (metric == "l21")
      if (points.shape(1)%3 != 0)
        throw std::runtime_error("Error L21 only work with 3D multiple values");

    switch (points.shape(1)) {
      case 1:
        if (metric == "l2")
          index = new KDTreeNumpyAdaptor<num_t, 1>(points, leaf_size);
        else
          index = new KDTreeNumpyAdaptor<num_t, 1, nanoflann::metric_L1>(
              points, leaf_size);
        break;
      case 2:
        if (metric == "l2")
          index = new KDTreeNumpyAdaptor<num_t, 2>(points, leaf_size);
        else
          index = new KDTreeNumpyAdaptor<num_t, 2, nanoflann::metric_L1>(
              points, leaf_size);
        break;
      case 3:
        if (metric == "l2")
          index = new KDTreeNumpyAdaptor<num_t, 3>(points, leaf_size);
        else if (metric == "l21")
          index = new KDTreeNumpyAdaptor<num_t, 3, nanoflann::metric_L21_3D_row>(
              points, leaf_size);
        else
          index = new KDTreeNumpyAdaptor<num_t, 3, nanoflann::metric_L1>(
              points, leaf_size);
        break;
      case 4:
        if (metric == "l2")
          index = new KDTreeNumpyAdaptor<num_t, 4>(points, leaf_size);
        else
          index = new KDTreeNumpyAdaptor<num_t, 4, nanoflann::metric_L1>(
              points, leaf_size);
        break;
      case 5:
        if (metric == "l2")
          index = new KDTreeNumpyAdaptor<num_t, 5>(points, leaf_size);
        else
          index = new KDTreeNumpyAdaptor<num_t, 5, nanoflann::metric_L1>(
              points, leaf_size);
        break;
      case 6:
        if (metric == "l2")
          index = new KDTreeNumpyAdaptor<num_t, 6>(points, leaf_size);
        else if (metric == "l21")
          index = new KDTreeNumpyAdaptor<num_t, 6, nanoflann::metric_L21_3D_row>(
              points, leaf_size);
        else
          index = new KDTreeNumpyAdaptor<num_t, 6, nanoflann::metric_L1>(
              points, leaf_size);
        break;
      case 7:
        if (metric == "l2")
          index = new KDTreeNumpyAdaptor<num_t, 7>(points, leaf_size);
        else
          index = new KDTreeNumpyAdaptor<num_t, 7, nanoflann::metric_L1>(
              points, leaf_size);
        break;
      case 8:
        if (metric == "l2")
          index = new KDTreeNumpyAdaptor<num_t, 8>(points, leaf_size);
        else
          index = new KDTreeNumpyAdaptor<num_t, 8, nanoflann::metric_L1>(
              points, leaf_size);
        break;
      case 9:
        if (metric == "l2")
          index = new KDTreeNumpyAdaptor<num_t, 9>(points, leaf_size);
        else if (metric == "l21")
          index = new KDTreeNumpyAdaptor<num_t, 9, nanoflann::metric_L21_3D_row>(
              points, leaf_size);
        else
          index = new KDTreeNumpyAdaptor<num_t, 9, nanoflann::metric_L1>(
              points, leaf_size);
        break;
      case 10:
        if (metric == "l2")
          index = new KDTreeNumpyAdaptor<num_t, 10>(points, leaf_size);
        else
          index = new KDTreeNumpyAdaptor<num_t, 10, nanoflann::metric_L1>(
              points, leaf_size);
        break;
      case 11:
        if (metric == "l2")
          index = new KDTreeNumpyAdaptor<num_t, 11>(points, leaf_size);
        else
          index = new KDTreeNumpyAdaptor<num_t, 11, nanoflann::metric_L1>(
              points, leaf_size);
        break;
      case 12:
        if (metric == "l2")
          index = new KDTreeNumpyAdaptor<num_t, 12>(points, leaf_size);
        else if (metric == "l21")
          index = new KDTreeNumpyAdaptor<num_t, 12, nanoflann::metric_L21_3D_row>(
              points, leaf_size);
        else
          index = new KDTreeNumpyAdaptor<num_t, 12, nanoflann::metric_L1>(
              points, leaf_size);
        break;
      case 13:
        if (metric == "l2")
          index = new KDTreeNumpyAdaptor<num_t, 13>(points, leaf_size);
        else
          index = new KDTreeNumpyAdaptor<num_t, 13, nanoflann::metric_L1>(
              points, leaf_size);
        break;
      case 14:
        if (metric == "l2")
          index = new KDTreeNumpyAdaptor<num_t, 14>(points, leaf_size);
        else
          index = new KDTreeNumpyAdaptor<num_t, 14, nanoflann::metric_L1>(
              points, leaf_size);
        break;
      case 15:
        if (metric == "l2")
          index = new KDTreeNumpyAdaptor<num_t, 15>(points, leaf_size);
        else if (metric == "l21")
          index = new KDTreeNumpyAdaptor<num_t, 15, nanoflann::metric_L21_3D_row>(
              points, leaf_size);
        else
          index = new KDTreeNumpyAdaptor<num_t, 15, nanoflann::metric_L1>(
              points, leaf_size);
        break;

      default:
        // Arbitrary dim but works slightly slower
        if (metric == "l2")
          index = new KDTreeNumpyAdaptor<num_t, -1>(points, leaf_size);
        else if (metric == "l21")
          index = new KDTreeNumpyAdaptor<num_t, -1, nanoflann::metric_L21_3D_row>(
              points, leaf_size);
        else
          index = new KDTreeNumpyAdaptor<num_t, -1, nanoflann::metric_L1>(
              points, leaf_size);
        break;
    }
    if (index_path.size()) {
      index->loadIndex(index_path);
    } else {
      index->buildIndex();
    }
  }

  ~KDTree() { delete index; }

  // kneighbors search
  std::pair<f_numpy_array_t, i_numpy_array_t> kneighbors(f_numpy_array_t array, size_t n_neighbors);
  std::pair<f_numpy_array_t, i_numpy_array_t> kneighbors_multithreaded(
      f_numpy_array_t array, size_t n_neighbors, size_t nThreads = 1);


  // radius search
  void radius_neighbors_idx(f_numpy_array_t, num_t radius = 1.0f);
  void radius_neighbors_idx_dists(f_numpy_array_t, num_t radius = 1.0f);
  void radius_neighbors_idx_multithreaded(
      f_numpy_array_t array, num_t radius = 1.0f, size_t nThreads = 1);
  void radius_neighbors_idx_dists_multithreaded(
      f_numpy_array_t array, num_t radius = 1.0f, size_t nThreads = 1);

  int save_index(const std::string &path);

  AbstractKDTree<num_t> *index;

  // result getter
  i_numpy_array_t getResultLenghts();
  i_numpy_array_t getResultIndicesPtr();
  i_numpy_array_t getResultIndicesRow();
  i_numpy_array_t getResultIndicesCol();
  f_numpy_array_t getResultDists();

 private:
  size_t n_neighbors;
  size_t leaf_size;
  std::string metric;
  num_t radius;

  std::vector<size_t> m_nbmatches;
  std::vector<std::vector<size_t>> m_indices;
  std::vector<std::vector<num_t>> m_dists;
};


// ESO Getter for results in python
template <typename num_t>
i_numpy_array_t KDTree<num_t>::getResultLenghts(){
  return pybind11::array(this->m_nbmatches.size(), this->m_nbmatches.data());
}

template <typename num_t>
i_numpy_array_t KDTree<num_t>::getResultIndicesPtr(){
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
i_numpy_array_t KDTree<num_t>::getResultIndicesRow(){
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
i_numpy_array_t KDTree<num_t>::getResultIndicesCol(){
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
KDTree<num_t>::KDTree(size_t n_neighbors, size_t leaf_size, std::string metric,
                      num_t radius)
    : n_neighbors(n_neighbors),
      leaf_size(leaf_size),
      metric(metric),
      radius(radius) {}

template <typename num_t>
std::pair<pybind11::array_t<num_t, pybind11::array::c_style | pybind11::array::forcecast>, i_numpy_array_t>
KDTree<num_t>::kneighbors(f_numpy_array_t array, size_t n_neighbors) {
  auto mat = array.template unchecked<2>();
  const num_t *query_data = mat.data(0, 0);
  size_t n_points = mat.shape(0);
  size_t dim = mat.shape(1);

  nanoflann::KNNResultSet<num_t> resultSet(n_neighbors);
  f_numpy_array_t results_dists({n_points, n_neighbors});
  i_numpy_array_t results_idxs({n_points, n_neighbors});

  // https://pybind11.readthedocs.io/en/stable/advanced/pycpp/numpy.html#direct-access
  num_t *res_dis_data =
      results_dists.template mutable_unchecked<2>().mutable_data(0, 0);
  size_t *res_idx_data =
      results_idxs.template mutable_unchecked<2>().mutable_data(0, 0);

  for (size_t i = 0; i < n_points; i++) {
    const num_t *query_point = &query_data[i * dim];
    resultSet.init(&res_idx_data[i * n_neighbors],
                   &res_dis_data[i * n_neighbors]);
    index->findNeighbors(resultSet, query_point, nanoflann::SearchParams());
  }

  return std::make_pair(results_dists, results_idxs);
}

template <typename num_t>
void KDTree<num_t>::radius_neighbors_idx(
    f_numpy_array_t array, num_t radius) {
  const auto mat = array.template unchecked<2>();
  const num_t *query_data = mat.data(0, 0);
  const size_t n_points = mat.shape(0);
  const size_t dim = mat.shape(1);

  const num_t search_radius = static_cast<num_t>(radius);
  this->m_nbmatches.resize(n_points);
  this->m_indices.resize(n_points);
  this->m_dists.clear();

  for (size_t i = 0; i < n_points; i++) {
    this->m_nbmatches[i] = index->radiusSearchIdx(
        &query_data[i * dim], search_radius, this->m_indices[i], nanoflann::SearchParams());
  }
  return;
}

template <typename num_t>
void KDTree<num_t>::radius_neighbors_idx_dists(
    f_numpy_array_t array, num_t radius) {
  const auto mat = array.template unchecked<2>();
  const num_t *query_data = mat.data(0, 0);
  const size_t n_points = mat.shape(0);
  const size_t dim = mat.shape(1);

  const num_t search_radius = static_cast<num_t>(radius);
  std::vector<std::pair<size_t, num_t>> ret_matches;
  this->m_nbmatches.resize(n_points);
  this->m_indices.resize(n_points);
  this->m_dists.resize(n_points);

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
void KDTree<num_t>::radius_neighbors_idx_multithreaded(
    f_numpy_array_t array, num_t radius, size_t nThreads) {

  const auto mat = array.template unchecked<2>();
  const num_t *query_data = mat.data(0, 0);
  const size_t n_points = mat.shape(0);
  const size_t dim = mat.shape(1);

  const num_t search_radius = static_cast<num_t>(radius);

  this->m_nbmatches.resize(n_points);
  this->m_indices.resize(n_points);
  this->m_dists.clear();

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
void KDTree<num_t>::radius_neighbors_idx_dists_multithreaded(
    f_numpy_array_t array, num_t radius, size_t nThreads) {
  // reset search results

  const auto mat = array.template unchecked<2>();
  const num_t *query_data = mat.data(0, 0);
  const size_t n_points = mat.shape(0);
  const size_t dim = mat.shape(1);

  const num_t search_radius = static_cast<num_t>(radius);
  std::vector<std::pair<size_t, num_t>> ret_matches;

  this->m_nbmatches.resize(n_points);
  this->m_indices.resize(n_points);
  this->m_dists.resize(n_points);

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
std::pair<pybind11::array_t<num_t, pybind11::array::c_style | pybind11::array::forcecast>, i_numpy_array_t>
KDTree<num_t>::kneighbors_multithreaded(f_numpy_array_t array, size_t n_neighbors, size_t nThreads) {
  auto mat = array.template unchecked<2>();
  const num_t *query_data = mat.data(0, 0);
  size_t n_points = mat.shape(0);
  size_t dim = mat.shape(1);

  f_numpy_array_t results_dists({n_points, n_neighbors});
  i_numpy_array_t results_idxs({n_points, n_neighbors});

  num_t *res_dis_data = results_dists.template mutable_unchecked<2>().mutable_data(0, 0);
  size_t *res_idx_data = results_idxs.template mutable_unchecked<2>().mutable_data(0, 0);

  auto searchBatch = [&](size_t startIdx, size_t endIdx) {
    for (size_t i = startIdx; i < endIdx; i++) {
      const num_t *query_point = &query_data[i * dim];
      index->knnSearch(query_point, n_neighbors, &res_idx_data[i * n_neighbors],
                       &res_dis_data[i * n_neighbors]);
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

  return std::make_pair(results_dists, results_idxs);
}

template <typename num_t>
int KDTree<num_t>::save_index(const std::string &path) {
  return index->saveIndex(path);
}

template <typename T>
using f_numpy_array = pybind11::array_t<T, pybind11::array::c_style | pybind11::array::forcecast>;

template <typename T>
std::pair<pybind11::list, pybind11::list> batched_kneighbors(
    pybind11::list index_points, pybind11::list query_points,
    size_t n_neighbors, std::string metric, size_t leaf_size,
    size_t n_threads = 1) {
  // Allocate memory before any computations
  size_t n_batches = index_points.size();
  pybind11::list g_results_dists;
  pybind11::list g_results_idxs;
  for (size_t i = 0; i < n_batches; i++) {
    f_numpy_array<T> batch = pybind11::cast<pybind11::array>(query_points[i]);

    size_t n_points = batch.shape(0);
    f_numpy_array<T> results_dists({n_points, n_neighbors});
    i_numpy_array_t results_idxs({n_points, n_neighbors});

    g_results_dists.append(results_dists);
    g_results_idxs.append(results_idxs);
  }

  auto SearchBatch = [&](size_t startIdx, size_t endIdx) {
    for (size_t j = startIdx; j < endIdx; j++) {
      KDTree<T> tree(n_neighbors, leaf_size, metric);
      f_numpy_array<T> batch_index =
          pybind11::cast<pybind11::array>(index_points[j]);
      f_numpy_array<T> batch_query =
          pybind11::cast<pybind11::array>(query_points[j]);
      tree.fit(batch_index, "");

      auto mat = batch_query.template unchecked<2>();
      const T *query_data = mat.data(0, 0);

      f_numpy_array<T> b_results_dists =
          pybind11::cast<pybind11::array>(g_results_dists[j]);
      i_numpy_array_t b_results_idxs =
          pybind11::cast<pybind11::array>(g_results_idxs[j]);
      T *res_dis_data =
          b_results_dists.template mutable_unchecked<2>().mutable_data(0, 0);
      size_t *res_idx_data =
          b_results_idxs.template mutable_unchecked<2>().mutable_data(0, 0);

      size_t n_points = batch_query.shape(0);
      size_t dim = batch_query.shape(1);
      for (size_t i = 0; i < n_points; i++) {
        const T *query_point = &query_data[i * dim];
        tree.index->knnSearch(query_point, n_neighbors,
                              &res_idx_data[i * n_neighbors],
                              &res_dis_data[i * n_neighbors]);
      }
    }
  };

  std::vector<std::thread> threadPool;
  size_t batchSize = std::ceil(static_cast<float>(n_batches) / n_threads);
  for (size_t i = 0; i < n_threads; i++) {
    size_t startIdx = i * batchSize;
    size_t endIdx = (i + 1) * batchSize;
    endIdx = std::min(endIdx, n_batches);
    threadPool.push_back(std::thread(SearchBatch, startIdx, endIdx));
  }
  for (auto &t : threadPool) {
    t.join();
  }

  return std::make_pair(g_results_dists, g_results_idxs);
}



template <typename num_t>
class KDTreeMulti : public KDTree<num_t>{
 public:
  using f_numpy_array_t = pybind11::array_t<num_t, pybind11::array::c_style | pybind11::array::forcecast>;

  const num_t *buf2;
};


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
      .def("getResultLenghts", &KDTree<float>::getResultLenghts)
      .def("getResultIndicesPtr", &KDTree<float>::getResultIndicesPtr)
      .def("getResultIndicesRow", &KDTree<float>::getResultIndicesRow)
      .def("getResultIndicesCol", &KDTree<float>::getResultIndicesCol)
      .def("getResultDists", &KDTree<float>::getResultDists)
      .def("save_index", &KDTree<float>::save_index);

  pybind11::class_<KDTree<double>>(m, "KDTree64")
      .def(pybind11::init<size_t, size_t, std::string, float>())
      .def("fit", &KDTree<double>::fit)
      .def("kneighbors", &KDTree<double>::kneighbors)
      .def("kneighbors_multithreaded",
           &KDTree<double>::kneighbors_multithreaded)
      .def("radius_neighbors_idx", &KDTree<double>::radius_neighbors_idx)
      .def("radius_neighbors_idx_dists", &KDTree<double>::radius_neighbors_idx_dists)
      .def("radius_neighbors_idx_multithreaded", &KDTree<double>::radius_neighbors_idx_multithreaded)
      .def("radius_neighbors_idx_dists_multithreaded", &KDTree<double>::radius_neighbors_idx_dists_multithreaded)
      .def("getResultLenghts", &KDTree<double>::getResultLenghts)
      .def("getResultIndicesPtr", &KDTree<double>::getResultIndicesPtr)
      .def("getResultIndicesRow", &KDTree<double>::getResultIndicesRow)
      .def("getResultIndicesCol", &KDTree<double>::getResultIndicesCol)
      .def("getResultDists", &KDTree<double>::getResultDists)
      .def("save_index", &KDTree<double>::save_index);

  m.def("batched_kneighbors32", &batched_kneighbors<float>,
        "Fit & query multiple independent kd-trees");
  m.def("batched_kneighbors64", &batched_kneighbors<double>,
        "Fit & query multiple independent kd-trees");
}
