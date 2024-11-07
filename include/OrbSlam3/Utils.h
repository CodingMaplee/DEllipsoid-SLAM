//
// Created by user on 2023/5/17.
//

#ifndef MAINLOOP_UTILS_H
#define MAINLOOP_UTILS_H

#include <Eigen/Dense>
#include <opencv2/core.hpp>
#include <unordered_set>
#include <unordered_map>

#define TO_RAD(x) 0.01745329251 * (x)
#define TO_DEG(x) 57.2957795131 * (x)

namespace ORB_SLAM3{
    using BBox2 = Eigen::Vector4d;
    using CenterPoint3d = Eigen::Vector3d;
    using BBox3 = Eigen::Matrix<double, 6, 1>;
    using Matrix34d = Eigen::Matrix<double, 3, 4>;

    inline BBox2 scale_bbox(const BBox2& bb, double scale) {
        double mx = (bb[0] + bb[2]) / 2;
        double my = (bb[1] + bb[3]) / 2;
        double dx = (bb[2] - bb[0]) / 2;
        double dy = (bb[3] - bb[1]) / 2;
        dx *= scale;
        dy *= scale;
        BBox2 scaled_bb(mx-dx, my-dy, mx+dx, my+dy);
        return scaled_bb;
    }

    inline Eigen::Vector2d bbox_center(const BBox2& bb) {
        return Eigen::Vector2d((bb[0]+bb[2])/2, (bb[1]+bb[3])/2);
    }

    inline bool is_inside_bbox(float u, float v, const BBox2& bbox)
    {
        return u >= bbox[0] && u <= bbox[2] && v >= bbox[1] && v <= bbox[3];
    }

    inline double bbox_area(const BBox2& bb)
    {
        return (bb[2] - bb[0]) * (bb[3] - bb[1]);
    }

    inline double bboxes_intersection(const BBox2& bb1, const BBox2& bb2)
    {
        double inter_w = std::max(std::min(bb1[2], bb2[2]) - std::max(bb1[0], bb2[0]), 0.0);
        double inter_h = std::max(std::min(bb1[3], bb2[3]) - std::max(bb1[1], bb2[1]), 0.0);
        return inter_h * inter_w;
    }

    inline double bboxes_iou(const BBox2& bb1, const BBox2& bb2)
    {
        double area_inter = bboxes_intersection(bb1, bb2);
        return area_inter / (bbox_area(bb1) + bbox_area(bb2) - area_inter);
    }

    inline double bbox3d_volume(const BBox3& bb)
    {
        return (bb[3] - bb[0]) * (bb[4] - bb[1]) * (bb[5] - bb[2]);
    }

    inline double bboxes3d_intersection(const BBox3& bb1, const BBox3& bb2)
    {
        double inter_x = std::max(std::min(bb1[3], bb2[3]) - std::max(bb1[0], bb2[0]), 0.0);
        double inter_y = std::max(std::min(bb1[4], bb2[4]) - std::max(bb1[1], bb2[1]), 0.0);
        double inter_z = std::max(std::min(bb1[5], bb2[5]) - std::max(bb1[2], bb2[2]), 0.0);
        return inter_x * inter_y * inter_z;
    }

    inline double bboxes3d_iou(const BBox3& bb1, const BBox3& bb2)
    {
        double area_inter = bboxes3d_intersection(bb1, bb2);
        return area_inter / (bbox3d_volume(bb1) + bbox3d_volume(bb2) - area_inter);
    }

    template <class T,int N>
    Eigen::Matrix<T, N, N> vec2sym(const Eigen::Matrix<T, (N*(N+1))/2, 1>& M)
    {
        Eigen::Matrix<T, N, N> Q;
        int a = 0;
        for (int i = 0; i < N; ++i) {
            for (int j = i; j < N; ++j) {
                Q(i, j) = M[a++];
                Q(j, i) = Q(i, j);
            }
        }
        return Q;
    }

    template <class T,int N>
    Eigen::Matrix<T, (N*(N+1))/2, 1> sym2vec(const Eigen::Matrix<T, N, N>& M)
    {
        Eigen::Matrix<T, (N*(N+1))/2, 1> v;
        int a = 0;
        for (int i = 0; i < N; ++i) {
            for (int j = i; j < N; ++j) {
                v[a++] = M(i, j);
            }
        }
        return v;
    }

    template <class T_out, class T_in, int N, int M>
    Eigen::Matrix<T_out, N, M> cvToEigenMatrix(cv::Mat m) {
        Eigen::Matrix<T_out, N, M> mat;
        for (int i = 0; i < N; ++i)
            for (int j = 0; j < M; ++j)
                mat(i, j) = m.at<T_in>(i, j);
        return mat;
    }

    template <class T_out, class T_in, int N, int M>
    cv::Mat eigenToCvMatrix(const Eigen::Matrix<T_in, N, M>& m) {
        cv::Mat mat(N, M);
        for (int i = 0; i < N; ++i)
            for (int j = 0; j < M; ++j)
                mat.at<T_out>(i, j) = m(i, j);
        return mat;
    }

    void writeOBJ(const std::string& filename, const Eigen::Matrix<double, Eigen::Dynamic, 3>& pts,
                  const Eigen::Matrix<int, Eigen::Dynamic, 3>& colors=Eigen::Matrix<int, Eigen::Dynamic, 3>(0, 3));



    template <class T>
    size_t count_set_intersection(const std::unordered_set<T>& s0, const std::unordered_set<T>& s1) {
        size_t res = 0;
        for (const auto& x : s0) {
            res += s1.count(x);
        }
        return res;
    }



    template <class T, class V>
    size_t count_map_intersection(const std::unordered_map<T, V>& s0, const std::unordered_map<T, V>& s1) {
        size_t res = 0;
        for (const auto& x : s0) {
            res += s1.count(x.first);
        }
        return res;
    }


    template <class T, class V>
    size_t count_set_map_intersection(const std::unordered_set<T>& s0, const std::unordered_map<T, V>& s1) {
        size_t res = 0;
        for (const auto& x : s0) {
            res += s1.count(x);
        }
        return res;
    }
}


#endif //MAINLOOP_UTILS_H
