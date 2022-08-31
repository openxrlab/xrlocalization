#include <iostream>
#include <pybind11/complex.h>
#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <string>

#include <pnp_solver.h>

namespace py = pybind11;

py::dict prior_guided_pnp(
    const Eigen::Ref<Eigen::Matrix<float, 2, Eigen::Dynamic, Eigen::RowMajor>>
        points2D,
    const Eigen::Ref<Eigen::Matrix<float, 3, Eigen::Dynamic, Eigen::RowMajor>>
        points3D,
    const Eigen::Ref<Eigen::Matrix<float, 1, Eigen::Dynamic, Eigen::RowMajor>>
        priors,
    const py::dict camera, const py::dict ransac_option) {

  assert(points2D.cols() == points3D.cols());
  assert(points3D.cols() == priors.cols());

  std::string camera_model_name = camera["model_name"].cast<std::string>();
  std::vector<double> params = camera["params"].cast<std::vector<double>>();

  std::vector<Eigen::Vector2d> point2D_vec(points2D.cols());
  std::vector<Eigen::Vector3d> point3D_vec(points3D.cols());
  std::vector<double> priors_vec(priors.cols());
  for (size_t i = 0; i != point2D_vec.size(); ++i) {
    point2D_vec[i][0] = static_cast<double>(points2D(0, i));
    point2D_vec[i][1] = static_cast<double>(points2D(1, i));
    point3D_vec[i][0] = static_cast<double>(points3D(0, i));
    point3D_vec[i][1] = static_cast<double>(points3D(1, i));
    point3D_vec[i][2] = static_cast<double>(points3D(2, i));
    priors_vec[i] = static_cast<double>(priors(0, i));
  }

  Eigen::Vector4d qvec;
  Eigen::Vector3d tvec;
  double error_thres = ransac_option["error_thres"].cast<double>();
  double inlier_ratio = ransac_option["inlier_ratio"].cast<double>();
  double confidence = ransac_option["confidence"].cast<double>();
  double max_iter = ransac_option["max_iter"].cast<double>();
  std::vector<char> mask;

  colpnp::Robustor robustor = colpnp::RANSAC;
  bool lo = ransac_option["local_optimal"].cast<bool>();
  if (lo) {
    robustor = colpnp::LORANSAC;
  }

  py::dict result;
  result["ninlier"] = 0;
  result["mask"] = mask;
  result["qvec"] = qvec;
  result["tvec"] = tvec;

  size_t num_inliers = 0;
  bool success = colpnp::sovle_pnp_ransac(
      point2D_vec, point3D_vec, camera_model_name, params, qvec, tvec,
      num_inliers, error_thres, inlier_ratio, confidence, max_iter, &mask,
      robustor, colpnp::WEIGHT_SAMPLE, &priors_vec);
  if (success) {
    result["ninlier"] = num_inliers;
    result["mask"] = mask;
    result["qvec"] = qvec;
    result["tvec"] = tvec;
  }

  return result;
}

PYBIND11_MODULE(solver, m) {
  m.doc() = "pybind11 pnp solver depend on colmap";

  m.def("prior_guided_pnp", &prior_guided_pnp, py::return_value_policy::copy);

#ifdef VERSION_INFO
  m.attr("__version__") = VERSION_INFO;
#else
  m.attr("__version__") = "dev";
#endif
}
