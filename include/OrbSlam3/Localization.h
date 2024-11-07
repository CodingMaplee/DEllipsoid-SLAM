//
// Created by user on 2023/7/13.
//

#ifndef MAINLOOP_LOCALIZATION_H
#define MAINLOOP_LOCALIZATION_H
#include <vector>
#include <unordered_map>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include "Ellipsoid.h"
#include "Utils.h"

namespace ORB_SLAM3
{

    using Mapping = std::vector<std::vector<size_t>>;

    std::pair<Mapping, Mapping> generate_possible_mappings(const std::vector<size_t> objects_category, const std::vector<size_t> detections_category);



  void  solveP3P_ransac(const std::vector<Ellipsoid, Eigen::aligned_allocator<Ellipsoid>>& ellipsoids,
                    const std::vector<size_t>& ellipsoids_categories,
                    const std::vector<BBox2, Eigen::aligned_allocator<BBox2>>& detections,
                    const std::vector<size_t>& detections_categories, const Eigen::Matrix3d& K, double THRESHOLD, int &best_index,
                    std::vector<Matrix34d,Eigen::aligned_allocator<Matrix34d>>& poses,
                    std::vector<double>& score,
                    std::vector<std::vector<std::pair<size_t, size_t>>>& used_pairs,
                    std::vector<std::vector<std::pair<size_t, size_t>>>& used_inliers);

    cv::Mat OptimizePoseFromObjects(const std::vector<Ellipse, Eigen::aligned_allocator<Ellipse>>& ellipses,
                                    const std::vector<Ellipsoid, Eigen::aligned_allocator<Ellipsoid>>& ellipsoids,
                                    cv::Mat Rt, const Eigen::Matrix3d& K);


}
#endif //MAINLOOP_LOCALIZATION_H
