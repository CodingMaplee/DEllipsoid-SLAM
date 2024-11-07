//
// Created by user on 2023/5/17.
//

#ifndef MAINLOOP_OPTIMIZEROBJECT_H
#define MAINLOOP_OPTIMIZEROBJECT_H
#include "Map.h"
#include "MapPoint.h"
#include "KeyFrame.h"
#include "LoopClosing.h"
#include "Frame.h"
#include "Distance.h"

#include "g2o/core/base_vertex.h"
#include "g2o/core/base_unary_edge.h"
#include "g2o/core/sparse_optimizer.h"
#include "g2o/core/block_solver.h"
#include "g2o/core/solver.h"
#include "g2o/core/optimization_algorithm_levenberg.h"
#include "g2o/solvers/dense/linear_solver_dense.h"
#include "g2o/types/sim3/types_seven_dof_expmap.h"

// #include <ceres/ceres.h>

namespace ORB_SLAM3
{

    Ellipsoid reconstruct_ellipsoid(const std::vector<Ellipse, Eigen::aligned_allocator<Ellipse>>& ellipses,
                                    const std::vector<Matrix34d, Eigen::aligned_allocator<Matrix34d>>& projections);

    struct EllipsoidQuat {
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

        EllipsoidQuat() {}
        explicit EllipsoidQuat(double *data) {
            // storage [ax ay az tx ty tz rx ry rz]
            // qw is computed such that q(qx, qy, qz, qw) is normalized
            axes = Eigen::Vector3d(data);
            se3 = g2o::SE3Quat(Eigen::Matrix<double, 6, 1>(data+3));
        }

        void SetTo(double *data) const {
            for (int i = 0; i < 3; ++i)
                data[i] = axes[i];
            auto rt = se3.log();
            for (int i = 0; i < 6; ++i)
                data[i+3] = rt[i];
        }

        Ellipsoid ToEllipsoid() const {
            return Ellipsoid(axes.cwiseAbs(), se3.rotation().toRotationMatrix(), se3.translation());
        }
        static EllipsoidQuat FromEllipsoid(const Ellipsoid& ell) {
            EllipsoidQuat eq;
            eq.axes = ell.GetAxes();
            eq.se3 = g2o::SE3Quat(Eigen::Quaterniond(ell.GetOrientation()), ell.GetCenter());
            return eq;
        }

        Eigen::Vector3d axes = Eigen::Vector3d::Zero();
        g2o::SE3Quat se3 = g2o::SE3Quat();
    };

    class VertexDynamicTransform: public g2o::BaseVertex<6, Sophus::SE3d>
    {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

        virtual void setToOriginImpl() override {
            _estimate = Sophus::SE3d();
        }

        virtual void oplusImpl(const double *update) override {
            Eigen::Matrix<double, 6, 1> update_eigen;
            update_eigen << update[0], update[1], update[2], update[3], update[4], update[5];
            _estimate = Sophus::SE3d::exp(update_eigen) * _estimate;
        }
        virtual bool read(std::istream&) override {return true;}
        virtual bool write(std::ostream&) const override {return true;}
    };

    class VertexEllipsoidNoRot: public g2o::BaseVertex<6, Eigen::Matrix<double, 6, 1>>
    {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

        virtual void setToOriginImpl() override {
            _estimate.head(3) = Eigen::Vector3d::Zero();
            _estimate.tail(3) = Eigen::Vector3d::Zero();
        }

        virtual void oplusImpl(const double *update) override {
            Eigen::Map<const Eigen::Matrix<double, 6, 1>> u(update);
            _estimate += u;
        }
        virtual bool read(std::istream&) override {return true;}
        virtual bool write(std::ostream&) const override {return true;}


    };


    class VertexEllipsoidQuat: public g2o::BaseVertex<9, EllipsoidQuat>
    {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

        virtual void setToOriginImpl() override {
            _estimate = EllipsoidQuat();
        }

        virtual void oplusImpl(const double *update) override {
            _estimate.axes += Eigen::Map<const Eigen::Vector3d>(update);
            _estimate.se3 = g2o::SE3Quat::exp(Eigen::Map<const Eigen::Matrix<double, 6, 1>>(update + 3)) * _estimate.se3;
        }

        virtual bool read(std::istream&) override {return true;}
        virtual bool write(std::ostream&) const override {return true;}
    };

    class EdgeDynamicEllipsoidProjection: public g2o::BaseUnaryEdge<1, double, VertexDynamicTransform>
    {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

        EdgeDynamicEllipsoidProjection(const Eigen::Matrix<double, 3, 4>& P, const Ellipse& det_ell, const Ellipsoid& ellipsoid)
                : P_(P), det_ell_(det_ell), ellipsoid_(ellipsoid) {
        }
        virtual void computeError() override
        {
            const VertexDynamicTransform* v = static_cast<VertexDynamicTransform*>(_vertices[0]);

            Sophus::SE3d T = v->estimate();
            Ellipsoid ellipsoid2 = ellipsoid_.TransformBySE3(T);

            Ellipse proj = ellipsoid2.project(P_);

            _error[0] = gaussian_wasserstein_2d(det_ell_, proj);
        }

        virtual bool read(std::istream&) override {return true;}
        virtual bool write(std::ostream&) const override {return true;}


    private:
        Eigen::Matrix<double, 3, 4> P_;
        Ellipse det_ell_;
        Ellipsoid ellipsoid_;
    };

class EdgeDynamicPointProjection: public g2o::BaseUnaryEdge<2, Eigen::Vector2d, VertexDynamicTransform>
    {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

    EdgeDynamicPointProjection(const Eigen::Vector3d &pos, const Eigen::Matrix3d &K, const Eigen::Matrix<double, 3, 4> &RT):_pos3d(pos), _K(K), _RT(RT){}
        virtual void computeError() override
        {
            const VertexDynamicTransform* v = static_cast<VertexDynamicTransform*>(_vertices[0]);

            Sophus::SE3d T = v->estimate();
            Eigen::Vector4d pos_tran = T.matrix() * Eigen::Vector4d(_pos3d.x(), _pos3d.y(), _pos3d.z(), 1.0);
            Eigen::Vector3d pos2d = _K * _RT * pos_tran;//Eigen::Vector4d(pos_tran.x(), pos_tran.y(), pos_tran.z(), 1.0);
            _error = _measurement - Eigen::Vector2d (pos2d[0]/pos2d[2], pos2d[1]/pos2d[2]);
        }

        virtual bool read(std::istream&) override {return true;}
        virtual bool write(std::ostream&) const override {return true;}

    private:
        Eigen::Vector3d  _pos3d;
        Eigen::Matrix3d _K;
        Eigen::Matrix<double, 3, 4> _RT;
    };

    class EdgeEllipsoidProjection: public g2o::BaseUnaryEdge<1, double, VertexEllipsoidNoRot>
// class EdgeEllipsoidProjection: public g2o::BaseUnaryEdge<1, double, VertexEllipsoid>
    {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

        EdgeEllipsoidProjection(const Eigen::Matrix<double, 3, 4>& P, const Ellipse& det_ell, const Eigen::Matrix3d& ellipsoid_R)
                : P_(P), det_ell_(det_ell), ellipsoid_R_(ellipsoid_R) {}

        virtual void computeError() override
        {
            const VertexEllipsoidNoRot* v = static_cast<VertexEllipsoidNoRot*>(_vertices[0]);
            Eigen::Matrix<double, 6, 1> ell = v->estimate();
            Ellipsoid ellipsoid(ell.head(3), Eigen::Matrix<double, 3, 3>::Identity(), ell.tail(3));

            // const VertexEllipsoid* v = static_cast<VertexEllipsoid*>(_vertices[0]);
            // EllipsoidQuat e = v->estimate();
            // // Ellipsoid ellipsoid = e.ToEllipsoid();
            // Ellipsoid ellipsoid(e.axes, ellipsoid_R_, e.se3.translation());


            Ellipse proj = ellipsoid.project(P_);

            // auto bb_proj = proj.ComputeBbox();
            // std::cout << "bb_proj: " << bb_proj.transpose() << std::endl;
            // std::cout << "measurement: " << _measurement.transpose() << std::endl;

            // Level-set
            // auto [total, errors, unused1, unused2] = ellipses_sampling_metric_ell(det_ell_, proj, 6, 4, 1.0);
            // for (int i = 0; i < 24; ++i)
            //     _error[i] = errors[i];

            // _error[0] = gaussian_bhattacharrya_2d(det_ell_, proj);
            _error[0] = gaussian_wasserstein_2d(det_ell_, proj);
        }

        virtual bool read(std::istream&) override {return true;}
        virtual bool write(std::ostream&) const override {return true;}

    private:
        Eigen::Matrix<double, 3, 4> P_;
        Ellipse det_ell_;
        Eigen::Matrix<double, 3, 3, Eigen::DontAlign> ellipsoid_R_;
    };

    class EdgeEllipsoidProjectionQuat: public g2o::BaseUnaryEdge<1, double, VertexEllipsoidQuat>
    {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

        EdgeEllipsoidProjectionQuat(const Eigen::Matrix<double, 3, 4>& P, const Ellipse& det_ell, const Eigen::Matrix3d& ellipsoid_R)
                : P_(P), det_ell_(det_ell), ellipsoid_R_(ellipsoid_R) {}

        virtual void computeError() override
        {
            const VertexEllipsoidQuat* v = static_cast<VertexEllipsoidQuat*>(_vertices[0]);
            Ellipsoid ellipsoid = v->estimate().ToEllipsoid();

            Ellipse proj = ellipsoid.project(P_);
            _error[0] = gaussian_bhattacharrya_2d(det_ell_, proj);
            //_error[0] = gaussian_wasserstein_2d(det_ell_, proj);
        }

        virtual bool read(std::istream&) override {return true;}
        virtual bool write(std::ostream&) const override {return true;}

    private:
        Eigen::Matrix<double, 3, 4> P_;
        Ellipse det_ell_;
        Eigen::Matrix<double, 3, 3, Eigen::DontAlign> ellipsoid_R_;
    };


    class EdgeEllipsoidMapPoint: public g2o::BaseUnaryEdge<1, double, VertexEllipsoidQuat>
    {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

        EdgeEllipsoidMapPoint(const Eigen::Vector3d& pt)
                : pt_(pt) {}

        virtual void computeError() override
        {
            const VertexEllipsoidQuat* v = static_cast<VertexEllipsoidQuat*>(_vertices[0]);
            Ellipsoid ellipsoid = v->estimate().ToEllipsoid();
            Eigen::Matrix4d C = ellipsoid.AsPrimal();
            double s = pt_.homogeneous().transpose() * C * pt_.homogeneous();
            _error[0] = s * 0.01;
        }

        virtual bool read(std::istream&) override {return true;}
        virtual bool write(std::ostream&) const override {return true;}

    private:
        Eigen::Vector3d pt_;
    };



    class EdgeEllipsoidProjectionQuatLevelSets: public g2o::BaseUnaryEdge<24, Eigen::Matrix<double, 24, 1>, VertexEllipsoidQuat>
    {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

        EdgeEllipsoidProjectionQuatLevelSets(const Eigen::Matrix<double, 3, 4>& P, const Ellipse& det_ell, const Eigen::Matrix3d& ellipsoid_R)
                : P_(P), det_ell_(det_ell), ellipsoid_R_(ellipsoid_R) {}

        virtual void computeError() override
        {
            const VertexEllipsoidQuat* v = static_cast<VertexEllipsoidQuat*>(_vertices[0]);
            Ellipsoid ellipsoid = v->estimate().ToEllipsoid();

            Ellipse proj = ellipsoid.project(P_);
            std::tuple<double, std::vector<double>, std::vector<double>, std::vector<std::vector<double>>> tu_ =
                    ellipses_sampling_metric_ell(det_ell_, proj, 6, 4, 1.0);
            //auto [total_error, errors, _, __] = ellipses_sampling_metric_ell(det_ell_, proj, 6, 4, 1.0);
            for (int i = 0; i < 24; ++i)
                _error[i] = std::get<1>(tu_)[i];
        }

        virtual bool read(std::istream&) override {return true;}
        virtual bool write(std::ostream&) const override {return true;}

    private:
        Eigen::Matrix<double, 3, 4> P_;
        Ellipse det_ell_;
        Eigen::Matrix<double, 3, 3, Eigen::DontAlign> ellipsoid_R_;
    };

    class EdgeEllipsoidProjectionQuatQBBox: public g2o::BaseUnaryEdge<4, Eigen::Vector4d, VertexEllipsoidQuat>
    {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

        EdgeEllipsoidProjectionQuatQBBox(const Eigen::Matrix<double, 3, 4>& P, const Ellipse& det_ell, const Eigen::Matrix3d& ellipsoid_R,
                                         unsigned int width, unsigned int height)
                : P_(P), det_ell_(det_ell), ellipsoid_R_(ellipsoid_R), w_(width), h_(height) {}

        virtual void computeError() override
        {
            const VertexEllipsoidQuat* v = static_cast<VertexEllipsoidQuat*>(_vertices[0]);
            Ellipsoid ellipsoid = v->estimate().ToEllipsoid();

            Ellipse proj = ellipsoid.project(P_);
            std::tuple<bool, BBox2> tu_1 = find_on_image_bbox(det_ell_, w_, h_);
            //auto [status_det, bb_det] = find_on_image_bbox(det_ell_, w_, h_);
            std::tuple<bool, BBox2> tu_2 = find_on_image_bbox(proj, w_, h_);
            //auto [status_proj, bb_proj] = find_on_image_bbox(proj, w_, h_);
            _error = std::get<1>(tu_1) - std::get<1>(tu_2);
        }

        virtual bool read(std::istream&) override {return true;}
        virtual bool write(std::ostream&) const override {return true;}

    private:
        Eigen::Matrix<double, 3, 4> P_;
        Ellipse det_ell_;
        Eigen::Matrix<double, 3, 3, Eigen::DontAlign> ellipsoid_R_;
        unsigned int w_, h_;
    };



    class EdgeEllipsoidProjectionQuatAlg: public g2o::BaseUnaryEdge<1, double, VertexEllipsoidQuat>
    {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

        EdgeEllipsoidProjectionQuatAlg(const Eigen::Matrix<double, 3, 4>& P, const Ellipse& det_ell, const Eigen::Matrix3d& ellipsoid_R)
                : P_(P), det_ell_(det_ell), ellipsoid_R_(ellipsoid_R) {}

        virtual void computeError() override
        {
            const VertexEllipsoidQuat* v = static_cast<VertexEllipsoidQuat*>(_vertices[0]);
            Ellipsoid ellipsoid = v->estimate().ToEllipsoid();

            Ellipse proj = ellipsoid.project(P_);
            _error[0] = algebraic_distance(det_ell_, proj);
        }

        virtual bool read(std::istream&) override {return true;}
        virtual bool write(std::ostream&) const override {return true;}

    private:
        Eigen::Matrix<double, 3, 4> P_;
        Ellipse det_ell_;
        Eigen::Matrix<double, 3, 3, Eigen::DontAlign> ellipsoid_R_;
    };



    class EdgeEllipsoidProjectionQuatTangency: public g2o::BaseUnaryEdge<4, Eigen::Vector4d, VertexEllipsoidQuat>
    {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

        EdgeEllipsoidProjectionQuatTangency(const Eigen::Matrix<double, 3, 4>& P, const Ellipse& det_ell, const Eigen::Matrix3d& ellipsoid_R,
                                            unsigned int width, unsigned int height)
                : P_(P), det_ell_(det_ell), ellipsoid_R_(ellipsoid_R), w_(width), h_(height) {}

        virtual void computeError() override
        {
            const VertexEllipsoidQuat* v = static_cast<VertexEllipsoidQuat*>(_vertices[0]);
            Ellipsoid ellipsoid = v->estimate().ToEllipsoid();
            BBox2 bb = det_ell_.ComputeBbox();

            Eigen::Vector3d x0(bb[0], bb[1], 1);
            Eigen::Vector3d x1(bb[2], bb[1], 1);
            Eigen::Vector3d x2(bb[0], bb[3], 1);
            Eigen::Vector3d x3(bb[2], bb[3], 1);
            Eigen::Vector3d l01 = x0.cross(x1);
            l01 /= l01[2];
            Eigen::Vector3d l02 = x0.cross(x2);
            l02 /= l02[2];
            Eigen::Vector3d l13 = x1.cross(x3);
            l13 /= l13[2];
            Eigen::Vector3d l23 = x2.cross(x3);
            l23 /= l23[2];
            auto d01 = tangency_segment_ellipsoid(l01, ellipsoid, P_);
            auto d02 = tangency_segment_ellipsoid(l02, ellipsoid, P_);
            auto d13 = tangency_segment_ellipsoid(l13, ellipsoid, P_);
            auto d23 = tangency_segment_ellipsoid(l23, ellipsoid, P_);
            std::cout <<"===========================\n";

            std::cout << d01 << "\n";
            std::cout << d02 << "\n";
            std::cout << d13 << "\n";
            std::cout << d23 << "\n";
            std::cout <<"===========================\n";
            _error[0] = d01;
            _error[1] = d02;
            _error[2] = d13;
            _error[3] = d23;
        }

        virtual bool read(std::istream&) override {return true;}
        virtual bool write(std::ostream&) const override {return true;}

    private:
        Eigen::Matrix<double, 3, 4> P_;
        Ellipse det_ell_;
        Eigen::Matrix<double, 3, 3, Eigen::DontAlign> ellipsoid_R_;
        unsigned int w_, h_;
    };














    class EdgeEllipsoidProjectionBbox: public g2o::BaseUnaryEdge<4, Eigen::Vector4d, VertexEllipsoidNoRot>
    {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

        EdgeEllipsoidProjectionBbox(const Eigen::Matrix<double, 3, 4>& P, const Ellipse& det_ell, const Eigen::Matrix3d& ellipsoid_R)
                : P_(P), det_ell_(det_ell), ellipsoid_R_(ellipsoid_R) {}

        virtual void computeError() override
        {
            const VertexEllipsoidNoRot* v = static_cast<VertexEllipsoidNoRot*>(_vertices[0]);
            Eigen::Matrix<double, 6, 1> ell = v->estimate();
            Ellipsoid ellipsoid(ell.head(3), Eigen::Matrix<double, 3, 3>::Identity(), ell.tail(3));

            Ellipse proj = ellipsoid.project(P_);
            auto bb_proj = proj.ComputeBbox();
            auto bb_det = det_ell_.ComputeBbox();
            _error = bb_det - bb_proj;
        }

        virtual bool read(std::istream&) override {return true;}
        virtual bool write(std::ostream&) const override {return true;}

    private:
        Eigen::Matrix<double, 3, 4> P_;
        Ellipse det_ell_;
        Eigen::Matrix<double, 3, 3, Eigen::DontAlign> ellipsoid_R_;
    };









    class  EdgeSE3ProjectEllipsoidOnlyPose: public g2o::BaseUnaryEdge<1, double, g2o::VertexSE3Expmap>{
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        EdgeSE3ProjectEllipsoidOnlyPose(const Ellipse& ell2d, const Ellipsoid& ell3d, const Eigen::Matrix3d& K) : ellipse_(ell2d), ellipsoid_(ell3d), K_(K) {}

        virtual bool read(std::istream&) override {return true;}
        virtual bool write(std::ostream&) const override {return true;}

        void computeError()  {
            const g2o::VertexSE3Expmap* v1 = static_cast<const g2o::VertexSE3Expmap*>(_vertices[0]);
            g2o::SE3Quat se3_quat = v1->estimate();
            Eigen::Vector3d t = se3_quat.translation();
            Eigen::Matrix3d R = se3_quat.rotation().toRotationMatrix();
            Matrix34d P;
            P << R, t;
            P = K_ * P;

            auto proj = ellipsoid_.project(P);
            // _error[0] = gaussian_bhattacharrya_2d(ellipse_, proj) * 10;
            _error[0] = gaussian_wasserstein_2d(ellipse_, proj);
            // std::cout << "==>" << _error[0] << "\n";
        }


        Ellipse ellipse_;
        Ellipsoid ellipsoid_;
        Eigen::Matrix3d K_;
    };



    class  EdgeSE3ProjectEllipsoid: public g2o::BaseBinaryEdge<1, double, VertexEllipsoidQuat, g2o::VertexSE3Expmap>{
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        EdgeSE3ProjectEllipsoid(const Ellipse& ell2d, const Eigen::Matrix3d& K) : ellipse_(ell2d), K_(K) {}

        virtual bool read(std::istream&) override {return true;}
        virtual bool write(std::ostream&) const override {return true;}

        void computeError()  {
            const VertexEllipsoidQuat* v1 = static_cast<const VertexEllipsoidQuat*>(_vertices[0]);
            const g2o::VertexSE3Expmap* v2 = static_cast<const g2o::VertexSE3Expmap*>(_vertices[1]);

            Ellipsoid ellipsoid = v1->estimate().ToEllipsoid();

            g2o::SE3Quat se3_quat = v2->estimate();
            Eigen::Vector3d t = se3_quat.translation();
            Eigen::Matrix3d R = se3_quat.rotation().toRotationMatrix();
            Matrix34d P;
            P << R, t;
            P = K_ * P;

            Ellipse proj = ellipsoid.project(P);
            _error[0] = gaussian_bhattacharrya_2d(ellipse_, proj) * 10;// * 100;
            // std::cout << "==>" << _error[0] << "\n";
        }

        Ellipse ellipse_;
        Eigen::Matrix3d K_;
    };



    class  EdgeSE3ProjectEllipsoidCenter: public g2o::BaseBinaryEdge<2, Eigen::Vector2d, g2o::VertexPointXYZ, g2o::VertexSE3Expmap>{
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        EdgeSE3ProjectEllipsoidCenter(const Ellipse& ell2d, const Eigen::Matrix3d& K) : ellipse_(ell2d), K_(K) {}

        virtual bool read(std::istream&) override {return true;}
        virtual bool write(std::ostream&) const override {return true;}

        void computeError()  {
            const g2o::VertexPointXYZ* v1 = static_cast<const g2o::VertexPointXYZ*>(_vertices[0]);
            const g2o::VertexSE3Expmap* v2 = static_cast<const g2o::VertexSE3Expmap*>(_vertices[1]);

            // Ellipsoid ellipsoid = v1->estimate().ToEllipsoid();
            Eigen::Vector3d p = v1->estimate();
            g2o::SE3Quat se3_quat = v2->estimate();
            Eigen::Vector3d t = se3_quat.translation();
            Eigen::Matrix3d R = se3_quat.rotation().toRotationMatrix();
            Matrix34d P;
            P << R, t;
            P = K_ * P;

            Eigen::Vector3d p2 = K_ * (R * p + t);
            Eigen::Vector2d c = ellipse_.GetCenter();
            _error[0] = (c[0] - p2[0] / p2[2]) * 1;
            _error[1] = (c[1] - p2[1] / p2[2]) * 1;
            std::cout << "error = " << _error[0] << " " << _error[1] << "\n";
        }

        Ellipse ellipse_;
        Eigen::Matrix3d K_;
    };


// struct EllipsoidProjectionError
// {
//     public:
//     //   EIGEN_MAKE_ALIGNED_OPERATOR_NEW

//     EllipsoidProjectionError(const BBox2& measurement, const Eigen::Matrix<double, 3, 4>& P, const Eigen::Matrix3d& ellipsoid_R, const Eigen::Vector3d& axes)
//     // EllipsoidProjectionError(const Eigen::Vector2d& measurement, const Eigen::Matrix<double, 3, 4>& P, const Eigen::Matrix3d& ellipsoid_R, const Eigen::Vector3d& axes)
//      : measurement_(measurement), P_(P), ellipsoid_R_(ellipsoid_R), ellipsoid_axes_(axes) {}

//     bool operator() (const double* const x, double* residual) const {

//         // Eigen::Matrix<double, 3, 1> v(x);
//         Eigen::Matrix<double, 6, 1> v(x);
//         Ellipsoid ellipsoid(v.head<3>(), ellipsoid_R_, v.tail<3>());
//         // Eigen::Matrix4d Q;
//         // Q << x[0], x[1], x[2], x[3],
//         //      x[1], x[4], x[5], x[6],
//         //      x[2], x[5], x[7], x[8],
//         //      x[3], x[6], x[8], -1;
//         // Ellipsoid ellipsoid(Q);

//         // Ellipsoid ellipsoid(ellipsoid_axes_, ellipsoid_R_, v);
//         Ellipse ell = ellipsoid.project(P_);
//         // residual[0] = measurement_[0] -  0.1;
//         // residual[1] = measurement_[1] - 0.1;
//         // Eigen::Vector2d center = ell.GetCenter();
//         // BBox2 proj = ell.ComputeBbox();
//         // std::cout << ell << "\n";
//         // std:cout << "bbox: " << proj.transpose() << "\n";
//         // std::cout << "residuals: ";
//         // residual[0] = measurement_[0] - center[0];
//         // residual[1] = measurement_[1] - center[1];
//         // return true;
//         // for (int i = 0; i < 4; ++i) {
//         //     residual[i] = measurement_[i] - proj[i];
//         //     // std::cout << measurement_[i] << " - " << proj[i] << " = ";
//         //     // std::cout << residual[i] << " ";
//         // }
//         // std::cout << "\n";


//         auto ell_det = Ellipse::FromBbox(measurement_);

//         // IoU + center
//         // residual[0] = 1.0 - bboxes_iou(measurement_, proj) + (bbox_center(proj) - bbox_center(measurement_)).squaredNorm();

//         // Wass
//         // residual[0] = gaussian_wasserstein_2d(ell_det, ell);
//         residual[0] = gaussian_bhattacharrya_2d(ell_det, ell);

//         // Level-set
//         // auto [total, errors, unused1, unused2] = ellipses_sampling_metric_ell(ell_det, ell, 6, 4, 1.0);
//         // residual[0] = total;
//         // for (int i = 0; i < 24; ++i)
//         //     residual[i] = errors[i];

//         // // Eigen::Matrix3d Q_proj = ell.AsDual();
//         // // Eigen::Matrix3d Q_det = ell_det.AsDual();
//         // // residual[0] = Q_det(0, 0) - Q_proj(0, 0);
//         // // residual[1] = Q_det(0, 1) - Q_proj(0, 1);
//         // // residual[2] = Q_det(0, 2) - Q_proj(0, 2);
//         // // residual[3] = Q_det(1, 1) - Q_proj(1, 1);
//         // // residual[4] = Q_det(1, 2) - Q_proj(1, 2);
//         // // residual[5] = Q_det(2, 2) - Q_proj(2, 2);
//         return true;
//     }

//     Eigen::Matrix<double, 4, 1, Eigen::DontAlign> measurement_;
//     Eigen::Matrix<double, 3, 4, Eigen::DontAlign> P_;
//     Eigen::Matrix<double, 3, 3, Eigen::DontAlign> ellipsoid_R_;
//     Eigen::Matrix<double, 3, 1, Eigen::DontAlign> ellipsoid_axes_;

// };




} //namespace ORB_SLAM
#endif //MAINLOOP_OPTIMIZEROBJECT_H
