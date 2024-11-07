//
// Created by user on 2023/5/17.
//
#pragma once
#ifndef MAINLOOP_ELLIPSOID_H
#define MAINLOOP_ELLIPSOID_H
#include <iostream>
#include <Eigen/Dense>
#include <sophus/se3.hpp>
#include "Ellipse.h"

namespace ORB_SLAM3
{

    class Ellipsoid
    {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

        Ellipsoid() {
            Q_ = Eigen::Matrix4d::Identity();
            Q_(3, 3) = -1;
            has_changed_ = true;
        }


        Ellipsoid(const Eigen::Matrix4d& Q) {
            if ((Q.transpose() - Q).cwiseAbs().sum() > 1e-3) {
                std::cerr << "Warning: Matrix should be symmetric" << "\n";
            }
            Q_ = Q;
            Q_ /= -Q(3, 3);
            has_changed_ = true;
        }

        Ellipsoid(const Eigen::Vector3d& axes,
                  const Eigen::Matrix3d& R,
                  const Eigen::Vector3d& center);

        void operator= (const Ellipsoid& ell) {
            Q_ = ell.Q_;

            has_changed_ = ell.has_changed_;
            axes_ = ell.axes_;
            R_ = ell.R_;
            center_ = ell.center_;
        }


        static Ellipsoid FromBbox(const BBox3& bbox) {
            double dx = 0.5 * (bbox[3] - bbox[0]);
            double dy = 0.5 * (bbox[4] - bbox[1]);
            double dz = 0.5 * (bbox[5] - bbox[2]);
            double cx = 0.5 * (bbox[3] + bbox[0]);
            double cy = 0.5 * (bbox[4] + bbox[1]);
            double cz = 0.5 * (bbox[5] + bbox[2]);
            return Ellipsoid(Eigen::Vector3d(dx, dy, dz), Eigen::Matrix3d::Identity(), Eigen::Vector3d(cx, cy, cz));
        }

        static Ellipsoid FromDual(const Eigen::Matrix4d& Q) {
            Eigen::Matrix4d Q_sym = 0.5 * (Q + Q.transpose());
            return Ellipsoid(Q_sym);
        }

        static Ellipsoid FromPrimal(const Eigen::Matrix4d& A) {
            return Ellipsoid(A.inverse());
        }


        Eigen::Matrix4d AsDual() const {
            return Q_;
        }

        Eigen::Matrix4d AsPrimal() const {
            return Q_.inverse();
        }


        inline Eigen::Vector3d GetAxes() const {
            if (has_changed_) {
                decompose();
                has_changed_ = false;
            }
            return axes_;
        }

        inline Eigen::Matrix3d GetOrientation() const {
            if (has_changed_) {
                decompose();
                has_changed_ = false;
            }
            return R_;
        }

        inline Eigen::Vector3d GetCenter() const {
            if (has_changed_) {
                decompose();
                has_changed_ = false;
            }
            return center_;
        }

        Ellipse project(const Eigen::Matrix<double, 3, 4>& P) const {
            Eigen::Matrix3d C = P * Q_ * P.transpose();
            return Ellipse::FromDual(C);
        }

        BBox3 ComputeBbox() const;

        bool IsInside(const Eigen::Vector3d& pt) const;

        Eigen::Matrix<double, Eigen::Dynamic, 3> GeneratePointCloud(int sampling=50) const;

        friend std::ostream& operator <<(std::ostream& os, const Ellipsoid& ell);

         Ellipsoid TransformBySE3(Sophus::SE3d T)
        {
            Ellipsoid ellipse2;
            Eigen::Matrix4d RT = T.matrix();
            Eigen::Matrix3d R = RT.block<3,3>(0,0);

            //ellipse2.axes_ = R * axes_;
//            ellipse2.R_ = R_;
//            ellipse2.axes_ = axes_;
            //ellipse2.center_ = center_ + Eigen::Vector3d (RT(3,0),RT(3,1),RT(3,2));
            ellipse2.Q_ = RT * Q_ * RT.transpose();
            return ellipse2;
        }
    private:
        void decompose() const;

    private:
        Eigen::Matrix4d Q_;
        mutable bool has_changed_ = true;
        mutable Eigen::Vector3d axes_;
        mutable Eigen::Matrix3d R_;
        mutable Eigen::Vector3d center_;

        static Eigen::Matrix<double, Eigen::Dynamic, 3> base_ellipsoid_pts;

    };


}
#endif //MAINLOOP_ELLIPSOID_H
