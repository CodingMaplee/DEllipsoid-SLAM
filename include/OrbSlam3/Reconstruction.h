//
// Created by user on 2023/5/17.
//

#ifndef MAINLOOP_RECONSTRUCTION_H
#define MAINLOOP_RECONSTRUCTION_H
#include "Ellipsoid.h"
#include "Utils.h"

namespace ORB_SLAM3
{



    std::pair<bool, Ellipsoid>
    ReconstructEllipsoidFromCenters(const std::vector<BBox2, Eigen::aligned_allocator<BBox2>>& boxes,
                                    const std::vector<Matrix34d, Eigen::aligned_allocator<Matrix34d>>& Rts,
                                    const std::vector<CenterPoint3d, Eigen::aligned_allocator<CenterPoint3d>>& CenterPoint3ds,
                                    const Eigen::Matrix3d& K);

    Eigen::Vector3d TriangulatePoints2(const Eigen::Vector2d& uv1, const Eigen::Vector2d& uv2,
                                       const Matrix34d& P1, const Matrix34d& P2);

    Eigen::Vector3d TriangulatePoints(const std::vector<Eigen::Vector2d, Eigen::aligned_allocator<Eigen::Vector2d>>& points,
                                      const std::vector<Matrix34d, Eigen::aligned_allocator<Matrix34d>>& projections);

    Eigen::Vector3d TriangulatePointsRansac(const std::vector<Eigen::Vector2d, Eigen::aligned_allocator<Eigen::Vector2d>>& points,
                                            const std::vector<Matrix34d, Eigen::aligned_allocator<Matrix34d>>& projections,
                                            int max_iter);

    std::pair<bool, Ellipsoid>
    ReconstructEllipsoidFromLandmarks(const std::vector<BBox2, Eigen::aligned_allocator<BBox2>>& bboxes,
                                      const std::vector<Matrix34d, Eigen::aligned_allocator<Matrix34d>>& Rts,
                                      const Eigen::Matrix3d& K, const Eigen::Matrix<double, 3, Eigen::Dynamic>& pts);

    Ellipsoid reconstruct_ellipsoid_lstsq(const std::vector<Ellipse, Eigen::aligned_allocator<Ellipse>>& ellipses,
                                          const std::vector<Matrix34d, Eigen::aligned_allocator<Matrix34d>>& projections);

    std::pair<bool, Ellipsoid>
    ReconstructEllipsoidCrocco(const std::vector<BBox2, Eigen::aligned_allocator<BBox2>>& bboxes,
                               const std::vector<Matrix34d, Eigen::aligned_allocator<Matrix34d>>& Rts,
                               const Eigen::Matrix3d& K, bool use_two_passes);

}
#endif //MAINLOOP_RECONSTRUCTION_H
