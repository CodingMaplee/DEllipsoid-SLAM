//
// Created by user on 2023/5/17.
//

#ifndef MAINLOOP_DISTANCE_H
#define MAINLOOP_DISTANCE_H
#include <iostream>
#include <vector>

#include <Eigen/Dense>

#include "Ellipse.h"
#include "Ellipsoid.h"
#include "Utils.h"


namespace ORB_SLAM3
{

    double gaussian_wasserstein_2d(const Ellipse& ell1, const Ellipse& ell2);

    double gaussian_bhattacharrya_2d(const Ellipse& ell1, const Ellipse& ell2);

    std::vector<std::vector<double>> generate_sampling_points(const Ellipse& ell, int count_az, int count_dist, double scale);

    std::tuple<double, std::vector<double>, std::vector<double>, std::vector<std::vector<double>>>
    ellipses_sampling_metric_ell(const Ellipse& ell1, const Ellipse& ell2, int N_az, int N_dist, double sampling_scale);

    std::tuple<bool, BBox2> find_on_image_bbox(const Ellipse& ell, int width, int height);

    inline double algebraic_distance(const Ellipse& ell1, const Ellipse& ell2)
    {
        return (sym2vec<double, 3>(ell1.AsDual()) - sym2vec<double, 3>(ell2.AsDual())).norm();
    }

    inline double tangency_segment_ellipsoid(const Eigen::Vector3d& line, const Ellipsoid& ellipsoid, const Matrix34d& P)
    {
        Eigen::Vector4d pl = P.transpose() * line;
        return pl.transpose() * ellipsoid.AsDual() * pl;
    }

}
#endif //MAINLOOP_DISTANCE_H
