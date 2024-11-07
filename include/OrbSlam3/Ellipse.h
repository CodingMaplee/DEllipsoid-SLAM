//
// Created by user on 2023/5/17.
//

#ifndef MAINLOOP_ELLIPSE_H
#define MAINLOOP_ELLIPSE_H
#include <iostream>

#include <Eigen/Dense>

#include "Utils.h"


namespace ORB_SLAM3
{

    struct Ellipse
    {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

        Ellipse() {
            C_ = Eigen::Matrix3d::Identity();
            C_(2, 2) = -1;
            has_changed_ = true;
        }


        Ellipse(const Eigen::Matrix3d& C) {
            if ((C.transpose() - C).cwiseAbs().sum() > 1e-3) {
                std::cerr << "Warning: Matrix should be symmetric" << "\n";
            }
            C_ = C;
            C_ /= -C_(2, 2);
            has_changed_ = true;
        }

        Ellipse(const Eigen::Vector2d& axes,
                double angle,
                const Eigen::Vector2d& center);


        static Ellipse FromDual(const Eigen::Matrix3d& C) {
            Eigen::Matrix3d C_sym = 0.5 * (C + C.transpose());
            return Ellipse(C_sym);
        }

        static Ellipse FromPrimal(const Eigen::Matrix3d& A) {
            return Ellipse(A.inverse());
        }


        static Ellipse FromBbox(const BBox2& bbox, double angle=0.0) {
            double w = 0.5 * (bbox[2] - bbox[0]);
            double h = 0.5 * (bbox[3] - bbox[1]);
            double cx = 0.5 * (bbox[2] + bbox[0]);
            double cy = 0.5 * (bbox[3] + bbox[1]);
            return Ellipse(Eigen::Vector2d(w, h), angle, Eigen::Vector2d(cx, cy));
        }

        Eigen::Matrix3d AsDual() const {
            return C_;
        }

        Eigen::Matrix3d AsPrimal() const {
            return C_.inverse();
        }

        Eigen::Vector2d GetAxes() const {
            if (has_changed_) {
                decompose();
                has_changed_ = false;
            }
            return axes_;
        }

        double GetAngle() const {
            if (has_changed_) {
                decompose();
                has_changed_ = false;
            }
            return angle_;
        }

        Eigen::Vector2d GetCenter() const {
            if (has_changed_) {
                decompose();
                has_changed_ = false;
            }
            return center_;
        }

        BBox2 ComputeBbox() const;

        std::tuple<Eigen::Vector2d, Eigen::Matrix2d> AsGaussian() const {
            if (has_changed_) {
                decompose();
                has_changed_ = false;
            }
            Eigen::Matrix2d A_dual;
            A_dual << std::pow(axes_[0], 2), 0.0,
                    0.0, std::pow(axes_[1], 2);
            Eigen::Matrix2d R;
            R << std::cos(angle_), -std::sin(angle_),
                    std::sin(angle_), std::cos(angle_);
            Eigen::Matrix2d cov = R * A_dual * R.transpose();
            Eigen::Vector2d center = center_;

            return {std::tuple<Eigen::Vector2d, Eigen::Matrix2d>(center_, cov)};
        }
        Eigen::Matrix3d ComposePrimalMatrix() const;

        friend std::ostream& operator <<(std::ostream& os, const Ellipse& ell);

    private:
        void decompose() const;


    private:
        Eigen::Matrix3d C_;

        mutable bool has_changed_ = true;
        mutable Eigen::Vector2d axes_ = Eigen::Vector2d(-1, -1);
        mutable double angle_ = 0;
        mutable Eigen::Vector2d center_ = Eigen::Vector2d(-1, -1);


    };


}
#endif //MAINLOOP_ELLIPSE_H
