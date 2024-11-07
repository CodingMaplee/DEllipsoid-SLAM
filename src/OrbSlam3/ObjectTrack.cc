#include "ObjectTrack.h"
#include "OptimizerObject.h"
#include "ColorManager.h"
#include "MapObject.h"
#include <mutex>
#include <fstream>
#include <iostream>
#include <memory>
#include "Reconstruction.h"

namespace ORB_SLAM3
{
    unsigned int ObjectTrack::factory_id = 0;
    int chair_count = 0;
    ObjectTrack::Ptr ObjectTrack::CreateNewObjectTrack(unsigned int cat, const BBox2& bbox, double score, const Matrix34d& Rt, const cv::Mat &imD, const Eigen::Vector4d& centerPoint_cam, unsigned int frame_idx, Tracking *tracker, KeyFrame *kf) {
        ObjectTrack::Ptr obj = std::make_shared<ObjectTrack>();
        obj->id_ = factory_id++;
        obj->category_id_ = cat;
        obj->bboxes_.reset(max_frames_history);
        obj->Rts_.reset(max_frames_history);
        obj->scores_.reset(max_frames_history);
        obj->bboxes_.push_front(bbox);
        obj->Rts_.push_front(Rt);
        obj->scores_.push_front(score);
        obj->last_obs_score_ = score;

        Eigen::Matrix4d RT = Eigen::Matrix4d::Identity();
        RT.block<3,4>(0,0) = Rt;
        //cout<<"category_id_ "<<cat<<endl;
        //cout<<"CenterPoint3d1:"<<" x="<<centerPoint_cam.x()<<" y="<<centerPoint_cam.y()<<" z="<<centerPoint_cam.z()<<endl;
        Eigen::Vector4d centerPoint4d = RT.inverse() * centerPoint_cam;
        //cout<<"CenterPoint3d2:"<<" x="<<centerPoint4d.x()<<" y="<<centerPoint4d.y()<<" z="<<centerPoint4d.z()<<endl;
        obj->CenterPoint_.push_front(Eigen::Vector3d(centerPoint4d.x(),centerPoint4d.y(),centerPoint4d.z()));

        Eigen::Vector3d centerPoint3d = Eigen::Vector3d(centerPoint4d.x(),centerPoint4d.y(),centerPoint4d.z());
        if (kf) {
            obj->keyframes_bboxes_[kf] = bbox;
            obj->keyframes_scores_[kf] = score;
            obj->keyframes_CenterPoint3d_[kf] = centerPoint3d;
        }
        obj->last_obs_frame_id_ = frame_idx;
        obj->color_ = RandomUniformColorGenerator::Generate();
        cv::Scalar col(0, 0, 0);
        //cv::Scalar bgr(0+obj->id_*10, 0+obj->id_*10,0+obj->id_*10);
        //obj->color_ = bgr;
        if (cat == 1) {
            col = cv::Scalar(0, 0, 0);
        } else if (cat == 2) {
            col = cv::Scalar(0, 255, 0);
        } else if (cat == 3) {
            col = cv::Scalar(0, 0, 255);
        } else if (cat == 4) {
            col = cv::Scalar(255, 255, 0);
        } else if (cat == 5) {
            col = cv::Scalar(0, 255, 255);
        } else if (cat == 6) {
            col = cv::Scalar(130, 0, 220);
        }
        else if (cat == 7) {
            col = cv::Scalar(255, 0, 0);
        }
        else if (cat == 8) {
            col = cv::Scalar(220, 90, 100);
        }

        obj->color_ = col;
        obj->tracker_ = tracker;
        obj->status_ = ObjectTrackStatus::ONLY_2D;
        obj->unc_ = 0.5; // ===> depends on the variance of the reconstruction gaussian curve
        return obj;

    }

    ObjectTrack::~ObjectTrack() {
        // if (map_object_) {
        //     delete map_object_;
        // }
    }




    bool ObjectTrack::ReconstructFromLandmarks(Atlas* map)
    {

        // get current map points
        std::vector<MapPoint*> points = map->GetAllMapPoints();
        Eigen::Matrix<double, 3, Eigen::Dynamic> pts(3, points.size());
        for(size_t j = 0; j < points.size(); j++) {
            Eigen::Vector3f p = points[j]->GetWorldPos();
            pts(0, j) = p.x();
            pts(1, j) = p.y();
            pts(2, j) = p.z();
        }

        std::vector<Matrix34d, Eigen::aligned_allocator<Matrix34d>> Rts = Rts_.to_vector();
        std::vector<BBox2, Eigen::aligned_allocator<BBox2>> bboxes = bboxes_.to_vector();

        std::pair<bool, Ellipsoid> pair_ = ReconstructEllipsoidFromLandmarks(bboxes, Rts, tracker_->GetK(), pts);
        //auto [status, ellipsoid] = ReconstructEllipsoidFromLandmarks(bboxes, Rts, tracker_->GetK(), pts);

        if (!pair_.first)
            return false;
        if (map_object_ ==NULL) cout<<"map_object_==NULL"<<endl;
        if (map_object_) // if object was already reconstructed return
            map_object_->SetEllipsoid(pair_.second);
        else
            map_object_ = std::make_unique<MapObject>(pair_.second, this);

        if (status_ == ObjectTrackStatus::ONLY_2D)
            status_ = ObjectTrackStatus::INITIALIZED;

        return true;
    }


    bool ObjectTrack::ReconstructCrocco(bool use_two_passes)
    {
        if (this->GetAngularDifference() < TO_RAD(10.0)) {
            // std::cerr << "Impossible to triangulate the center: not enough angular difference.\n";
            return false;
        }

        std::vector<Matrix34d, Eigen::aligned_allocator<Matrix34d>> Rts = Rts_.to_vector();
        std::vector<BBox2, Eigen::aligned_allocator<BBox2>> bboxes = bboxes_.to_vector();

        std::pair<bool, Ellipsoid> pair_ = ReconstructEllipsoidCrocco(bboxes, Rts, tracker_->GetK(), use_two_passes);
        //auto [status, ellipsoid] = ReconstructEllipsoidCrocco(bboxes, Rts, tracker_->GetK(), use_two_passes);

        if (!pair_.first)
            return false;

        if (map_object_) // if object was already reconstructed return
            map_object_->SetEllipsoid(pair_.second);
        else
            map_object_ = std::make_unique<MapObject>(pair_.second, this);

        if (status_ == ObjectTrackStatus::ONLY_2D)
            status_ = ObjectTrackStatus::INITIALIZED;

        return true;
    }


    bool ObjectTrack::ReconstructFromCenter(bool use_keyframes)
    {
//        if (this->GetAngularDifference() < TO_RAD(10.0)) {
//            std::cerr << "Impossible to triangulate the center: not enough angular difference.\n";
//            return false;
//        }
        //if (tracker_->)
        std::vector<Matrix34d, Eigen::aligned_allocator<Matrix34d>> Rts;
        std::vector<BBox2, Eigen::aligned_allocator<BBox2>> bboxes;
        std::vector<CenterPoint3d, Eigen::aligned_allocator<CenterPoint3d>> centerPoint3ds;
        if (use_keyframes) {
            cout<<"use_keyframes"<<endl;
            //auto [bbs, poses, _] = this->CopyDetectionsInKeyFrames();
            std::pair<std::vector<BBox2, Eigen::aligned_allocator<BBox2>>, std::vector<Matrix34d, Eigen::aligned_allocator<Matrix34d>>> tu_=
                    this->CopyDetectionsInKeyFrames();
            //auto [bbs, poses, _] = this->CopyDetectionsInKeyFrames();
            centerPoint3ds = this->CopyCenterPoint3dInKeyFrames();
            //std::cout<<centerPoint3ds.size()<<std::endl;
            bboxes = std::move(std::get<0>(tu_));
            //std::cout<<bboxes[bboxes.size()-1]<<std::endl;
            Rts = std::move(std::get<1>(tu_));
            //std::cout<<Rts[Rts.size()-1]<<std::endl;
        } else {
            Rts = Rts_.to_vector();
            bboxes = bboxes_.to_vector();
            centerPoint3ds = CenterPoint_.to_vector();
        }



        std::pair<bool, Ellipsoid> pair_ = ReconstructEllipsoidFromCenters(bboxes, Rts, centerPoint3ds, tracker_->GetK());
        //auto [status, ellipsoid] = ReconstructEllipsoidFromCenters(bboxes, Rts, tracker_->GetK());

        if (!pair_.first)
            return false;

        if (map_object_) // if object was already reconstructed return
            map_object_->SetEllipsoid(pair_.second);
        else
            map_object_ = std::make_unique<MapObject>(pair_.second, this);

        if (status_ == ObjectTrackStatus::ONLY_2D)
            status_ = ObjectTrackStatus::INITIALIZED;

        return true;

    }


    void ObjectTrack::AddDetection(const BBox2& bbox, double score, Eigen::Vector4d centerPoint_cam, const Matrix34d& Rt, unsigned int frame_idx, KeyFrame* kf)
    {
        unique_lock<mutex> lock(mutex_add_detection_);
        bboxes_.push_front(bbox);
        Rts_.push_front(Rt);
        scores_.push_front(score);
        last_obs_frame_id_ = frame_idx;
        last_obs_score_ = score;

        if (kf) {
            Eigen::Matrix<double,4,4> RT;
            RT.block<3,4>(0,0) = Rt;
            RT(3,0) = 0.0;
            RT(3,1) = 0.0;
            RT(3,2) = 0.0;
            RT(3,3) = 1.0;
            Eigen::Vector4d centerPoint4d = RT.inverse() * centerPoint_cam;

            keyframes_bboxes_[kf] = bbox;
            keyframes_scores_[kf] = score;
            keyframes_CenterPoint3d_[kf] = Eigen::Vector3d(centerPoint4d.x(), centerPoint4d.y(), centerPoint4d.z());
            std::unique_lock<std::mutex> lock(mutex_associated_map_points_);
            for (size_t i = 0; i < kf->mvKeys.size(); ++i) {
                MapPoint* p = kf->GetMapPoint(i);
                if (p) {
                    auto kp = kf->mvKeys[i];
                    if (is_inside_bbox(kp.pt.x , kp.pt.y, bbox)) {
                        if (associated_map_points_.find(p) == associated_map_points_.end())
                            associated_map_points_[p] = 1;
                        else
                            associated_map_points_[p]++;
                    }
                }
            }
        }
        if (status_ == ObjectTrackStatus::IN_MAP) {
            // kalman uncertainty update
            double k = unc_ / (unc_ + std::exp(-score));
            unc_ = unc_ * (1.0 - k);
        }


    }
    std::pair<std::vector<BBox2, Eigen::aligned_allocator<BBox2>>, std::vector<Matrix34d, Eigen::aligned_allocator<Matrix34d>>>
    ObjectTrack::CopyDetectionsInKeyFrames()
    {
        unique_lock<mutex> lock(mutex_add_detection_);
        std::vector<BBox2, Eigen::aligned_allocator<BBox2>> bboxes;
        std::vector<Matrix34d, Eigen::aligned_allocator<Matrix34d>> poses;
        std::vector<double> scores;
        if (map_object_) {
            bboxes.reserve(keyframes_bboxes_.size());
            poses.reserve(keyframes_bboxes_.size());
            scores.reserve(keyframes_bboxes_.size());
            int i = 0;
            std::vector<KeyFrame*> to_erase;
            for (auto& it : keyframes_bboxes_) {
                if (it.first->isToBeErased() || it.first->isBad()) {
                    to_erase.push_back(it.first);
                    continue;
                }
                bboxes.push_back(it.second);
                poses.push_back(it.first->GetPose().matrix3x4().cast<double>());
                scores.push_back(keyframes_scores_[it.first]);
                ++i;
            }
            for (auto* pt : to_erase) {
                keyframes_bboxes_.erase(pt);
                keyframes_scores_.erase(pt);
            }
        }
        return {bboxes, poses};
    }


    std::vector<CenterPoint3d, Eigen::aligned_allocator<CenterPoint3d>> ObjectTrack::CopyCenterPoint3dInKeyFrames()
    {
        unique_lock<mutex> lock(mutex_add_detection_);
        std::vector<CenterPoint3d, Eigen::aligned_allocator<CenterPoint3d>> centerPoint3ds;
        if (map_object_) {
            centerPoint3ds.reserve(keyframes_CenterPoint3d_.size());
            std::cout<<"keyframes_CenterPoint3d_:"<<keyframes_CenterPoint3d_.size()<<std::endl;
            int i = 0;
            std::vector<KeyFrame*> to_erase;
            for (auto& it : keyframes_CenterPoint3d_) {
                if (it.first->isToBeErased() || it.first->isBad()) {
                    to_erase.push_back(it.first);
                    continue;
                }
                centerPoint3ds.push_back(keyframes_CenterPoint3d_[it.first]);
                ++i;
            }
            for (auto* pt : to_erase) {
                keyframes_CenterPoint3d_.erase(pt);
            }
        }
        return centerPoint3ds;
    }



    bool ObjectTrack::TrackDynamic(void)
    {
        cout<<"Dynamic ..."<<endl;
        if(map_object_ == nullptr ||map_object_->GetTrack()->GetStatus() != ObjectTrackStatus::IN_MAP){
            cout<<"not in map";
            return  false;
        }
        const Ellipsoid& ellipsoid = map_object_->GetEllipsoid();
        Ellipsoid ellipsoid_ = ellipsoid;
        const Eigen::Matrix3d& K = tracker_->GetK();

        std::cout << "===============================> Start dynamic optimization " << category_id_ << std::endl;
        typedef g2o::BlockSolver<g2o::BlockSolverTraits<6,1>> BlockSolver_6_1;
        typedef g2o::LinearSolverEigen<BlockSolver_6_1::PoseMatrixType> LinearSolverType;
        auto solver = new g2o::OptimizationAlgorithmLevenberg(
                g2o::make_unique<BlockSolver_6_1>(g2o::make_unique<LinearSolverType>()));
        g2o::SparseOptimizer optimizer;
        optimizer.setAlgorithm(solver);
        optimizer.setVerbose(true);
        VertexDynamicTransform* v = new VertexDynamicTransform();
        v->setId(0);
        Sophus::SE3d e = Sophus::SE3d();
        //std::cout << "before dyna opt to  ellipsoid: " << e.matrix() << "\n";
        v->setEstimate(Sophus::SE3d());
        optimizer.addVertex(v);
        auto it_bb = bboxes_.front();
        auto it_Rt = Rts_.front();
        auto it_s = scores_.front();
        Eigen::Matrix<double, 3, 4> P = K * (it_Rt);
        EdgeDynamicEllipsoidProjection *edge_ellipsoid = new EdgeDynamicEllipsoidProjection(P, Ellipse::FromBbox(it_bb), ellipsoid);
        Eigen::Matrix<double, 1, 1> information_matrix = Eigen::Matrix<double, 1, 1>::Identity();
        edge_ellipsoid->setInformation(information_matrix);
        edge_ellipsoid->setId(0);
        edge_ellipsoid->setVertex(0, v);
        optimizer.addEdge(edge_ellipsoid);
        optimizer.initializeOptimization();
        optimizer.optimize(10);
        Sophus::SE3d e2 = v->estimate();
        Ellipsoid ellipsoidUpdate = ellipsoid_.TransformBySE3(e2);
        map_object_->SetEllipsoid(ellipsoidUpdate);
//        std::cout << "after dyna opt to  ellipsoid: " << e2.matrix()<< "\n";
//        fstream f;
//        f.open("../data.txt",ios::out|ios::app);
//        f <<  tracker_->GetCurrentFrameIdx()<<"        ";
//        f << ellipsoidUpdate.GetCenter().x() << "      "<< ellipsoidUpdate.GetCenter().y() << "      "<< ellipsoidUpdate.GetCenter().z() << endl << e2.matrix() << endl;
//        f.close();
        return true;
   }

    void ObjectTrack::OptimizeReconstruction(Atlas *map)
    {
        //optimize
//        if (!map_object_) {
//            std::cerr << "Impossible to optimize ellipsoid. It first requires a initial reconstruction." << std::endl;
//            return ;
//        }
        const Ellipsoid& ellipsoid = map_object_->GetEllipsoid();
        const Eigen::Matrix3d& K = tracker_->GetK();

        //std::cout << "===============================> Start ellipsoid optimization " << id_ << std::endl;
        typedef g2o::BlockSolver<g2o::BlockSolverTraits<6,1>> BlockSolver_6_1;
        typedef g2o::LinearSolverEigen<BlockSolver_6_1::PoseMatrixType> LinearSolverType;

//        BlockSolver_6_1::LinearSolverType *linear_solver = new g2o::LinearSolverDense<BlockSolver_6_1::PoseMatrixType>();
//        auto solver = new g2o::OptimizationAlgorithmLevenberg(
//                new BlockSolver_6_1(linear_solver)
//        );
        auto solver = new g2o::OptimizationAlgorithmLevenberg(
                g2o::make_unique<BlockSolver_6_1>(g2o::make_unique<LinearSolverType>()));

        g2o::SparseOptimizer optimizer;
        optimizer.setAlgorithm(solver);
        optimizer.setVerbose(false);

        VertexEllipsoidNoRot* vertex = new VertexEllipsoidNoRot();
        // VertexEllipsoid* vertex = new VertexEllipsoid();
        vertex->setId(0);
        Eigen::Matrix<double, 6, 1> e;
        e << ellipsoid.GetAxes(), ellipsoid.GetCenter();
        // EllipsoidQuat ellipsoid_quat = EllipsoidQuat::FromEllipsoid(*ellipsoid_);
        //std::cout << "before optim: " << e.transpose() << "\n";;
        // vertex->setEstimate(ellipsoid_quat);
        vertex->setEstimate(e);
        optimizer.addVertex(vertex);

        auto it_bb = bboxes_.begin();
        auto it_Rt = Rts_.begin();
        auto it_s = scores_.begin();
        for (size_t i = 0; i < bboxes_.size() && it_bb != bboxes_.end() && it_Rt != Rts_.end() && it_s != scores_.end(); ++i, ++it_bb, ++it_Rt, ++it_s) {
            Eigen::Matrix<double, 3, 4> P = K * (*it_Rt);
            //std::cout<<"*it_bb : " <<category_id_<<"width:" <<it_bb->z()-it_bb->x()<<" height:" <<it_bb->w()-it_bb->y()<<std::endl;
            EdgeEllipsoidProjection *edge = new EdgeEllipsoidProjection(P, Ellipse::FromBbox(*it_bb), ellipsoid.GetOrientation());
            edge->setId(i);
            edge->setVertex(0, vertex);

            Eigen::Matrix<double, 1, 1> information_matrix = Eigen::Matrix<double, 1, 1>::Identity();
            information_matrix *= *it_s;
            edge->setInformation(information_matrix);
            optimizer.addEdge(edge);
        }
        optimizer.initializeOptimization();
        optimizer.optimize(8);
        Eigen::Matrix<double, 6, 1> ellipsoid_est = vertex->estimate();
        //EllipsoidQuat ellipsoid_quat_est = vertex->estimate();

        //std::cout << "after optim: " << vertex->estimate().transpose() << "\n";

        Ellipsoid new_ellipsoid(ellipsoid_est.head(3), Eigen::Matrix3d::Identity(), ellipsoid_est.tail(3));
        map_object_->SetEllipsoid(new_ellipsoid);
    }

    bool ObjectTrack::CheckProjectIOU(double iou_threshold)
    {
        if(status_ != ObjectTrackStatus::IN_MAP)
            return false;
        const Ellipsoid& ellipsoid = map_object_->GetEllipsoid();
        const Eigen::Matrix3d& K = tracker_->GetK();
        auto it_Rt = Rts_.front();
        Eigen::Matrix<double, 3, 4> P = K * (it_Rt);
        Ellipse ell = ellipsoid.project(P);
        BBox2 proj_bb = ell.ComputeBbox();
        auto it_bb = bboxes_.front();
        double iou = bboxes_iou(it_bb, proj_bb);
        if(iou < iou_threshold)
            return true;
        return false;
    }
    //TODO
    bool ObjectTrack::CheckPreviousIOU(double iou_threshold)
    {

        auto bb = bboxes_.begin();
        auto it_bb = bboxes_.begin();
        ++it_bb;
        auto previous_bb = it_bb;
        //cout << "bb: " << *bb << "previous_bb: " << *previous_bb << endl;
        double iou = bboxes_iou(*bb, *previous_bb);
        if(iou < iou_threshold)
            return true;
        return false;

    }

    bool ObjectTrack::CheckReprojectionIoU(double iou_threshold)
    {
        if (status_ != ObjectTrackStatus::INITIALIZED)
            return false;

        const Ellipsoid& ellipsoid = map_object_->GetEllipsoid();
        const Eigen::Matrix3d& K = tracker_->GetK();

        bool valid = true;
        auto it_bb = bboxes_.begin();
        auto it_Rt = Rts_.begin();
        for (; it_bb != bboxes_.end() && it_Rt != Rts_.end(); ++it_bb, ++it_Rt) {
            Eigen::Matrix<double, 3, 4> P = K * (*it_Rt);
            Ellipse ell = ellipsoid.project(P);
            BBox2 proj_bb = ell.ComputeBbox();
            double iou = bboxes_iou(*it_bb, proj_bb);
            if (iou < iou_threshold)
            {
                valid  = false;
                break;
            }
        }
        return valid;
    }


    double ObjectTrack::CheckReprojectionIoUInKeyFrames(double iou_threshold)
    {
        // called form the local mapping thread
        std::pair<std::vector<BBox2, Eigen::aligned_allocator<BBox2>>, std::vector<Matrix34d, Eigen::aligned_allocator<Matrix34d>>> tu_ =
            this->CopyDetectionsInKeyFrames();
        //auto [bboxes, Rts, scores] = this->CopyDetectionsInKeyFrames();

        const Ellipsoid& ellipsoid = map_object_->GetEllipsoid();
        const Eigen::Matrix3d& K = tracker_->GetK();
        int valid = 0;
        for (size_t i = 0; i < std::get<0>(tu_).size(); ++i)
        {
            Eigen::Matrix<double, 3, 4> P = K * std::get<1>(tu_)[i];
            Ellipse ell = ellipsoid.project(P);
            BBox2 proj_bb = ell.ComputeBbox();
            double iou = bboxes_iou(std::get<0>(tu_)[i], proj_bb);
            if (iou > iou_threshold)
            {
                ++valid;
            }
        }
        double valid_ratio = static_cast<double>(valid) / std::get<0>(tu_).size();
        return valid_ratio;
    }

/// Compute the angle between bearing vectors going through the center
/// of the first and latest bboxes
    double ObjectTrack::GetAngularDifference() const
    {
        Eigen::Vector3d c0 = bbox_center(bboxes_.front()).homogeneous();
        Eigen::Matrix3d K = tracker_->GetK();
        Eigen::Matrix3d K_inv = K.inverse();
        Eigen::Vector3d v0 = K_inv * c0;
        v0.normalize();
        v0 = Rts_.front().block<3, 3>(0, 0).transpose() * v0;
        Eigen::Vector3d c1 = bbox_center(bboxes_.back()).homogeneous();
        Eigen::Vector3d v1 = K_inv * c1;
        v1.normalize();
        v1 = Rts_.back().block<3, 3>(0, 0).transpose() * v1;

        return std::atan2(v0.cross(v1).norm(), v0.dot(v1));
    }


    void ObjectTrack::Merge(ObjectTrack *track)
    {
        std::unique_lock<std::mutex> lock(mutex_add_detection_);

        // std::cout << "second track size = " << track->keyframes_bboxes_.size() << "\n";
        // std::cout << "track size before = " << keyframes_bboxes_.size() << "\n";
        // for (auto kf : track->keyframes_bboxes_) {
        //     if (keyframes_bboxes_.find(kf.first) != keyframes_bboxes_.end()) {
        //         std::cout << "kf already exsits\n";
        //         std::cout << keyframes_bboxes_[kf.first].transpose() << "\n";
        //         std::cout << track->keyframes_bboxes_[kf.first].transpose() << "\n\n";

        //     }
        //     keyframes_bboxes_[kf.first] = kf.second;
        // }

        for (auto kf : track->keyframes_bboxes_) {
            keyframes_bboxes_[kf.first] = kf.second;
            keyframes_scores_[kf.first] = track->keyframes_scores_[kf.first];
            keyframes_CenterPoint3d_[kf.first] = track->keyframes_CenterPoint3d_[kf.first];
        }

        // std::cout << "track size after = " << keyframes_bboxes_.size() << "\n";

        // track->SetIsBad();
    }

    void ObjectTrack::UnMerge(ObjectTrack *track) {
        for (auto kf : track->keyframes_bboxes_) {
            if (keyframes_bboxes_.find(kf.first) != keyframes_bboxes_.end()) {
                keyframes_bboxes_.erase(kf.first);
                keyframes_scores_.erase(kf.first);
                keyframes_CenterPoint3d_.erase(kf.first);
            }
        }
    }


    void ObjectTrack::CleanBadKeyFrames()
    {
        vector<KeyFrame*> to_remove;
        for (auto it : keyframes_bboxes_) {
            if (it.first->isBad() || it.first->isToBeErased()) {
                to_remove.push_back(it.first);
            }
        }
        for (auto* pt : to_remove) {
            keyframes_bboxes_.erase(pt);
            keyframes_scores_.erase(pt);
            keyframes_CenterPoint3d_.erase(pt);
        }
        //std::cout << "Cleaned " << to_remove.size() << " frames.\n";
    }

    bool ObjectTrack::ReconstructFromSamplesEllipsoid()
    {
        // std::ofstream file(std::to_string(id_) + "_reconstruction.txt");
        double dmin = 0.1;
        double dmax = tracker_->GetCurrentMeanDepth() * 2;
        int n_samples = 100;

        const Eigen::Matrix3d& K = tracker_->GetK();
        Eigen::Matrix3d K_inv = K.inverse();

        std::stack<Matrix34d> Rts;
        for (const auto& Rt : Rts_) {
            Rts.push(Rt);
        }

        std::stack<BBox2> bboxes;
        double mean_size = 0.0;
        for (auto const& bb : bboxes_) {
            bboxes.push(bb);
            mean_size = 0.5 * ((bb[2] - bb[0]) / K(0, 0) + (bb[3] - bb[1]) / K(1, 1));
        }
        mean_size /= bboxes_.size();

        auto Rt0 = Rts.top();
        auto bb0 = bboxes.top();
        Rts.pop();
        bboxes.pop();

        Eigen::Matrix3d o0 = Rt0.block<3, 3>(0, 0).transpose();
        Eigen::Vector3d p0 = -o0 * Rt0.block<3, 1>(0, 3);

        std::vector<Ellipsoid, Eigen::aligned_allocator<Ellipsoid>> samples;
        Eigen::Vector3d dir_cam = K_inv * bbox_center(bb0).homogeneous();
        double step = (dmax - dmin) / n_samples;
        for (int i = 0; i < n_samples; ++i) {
            Eigen::Vector3d X_c = (dmin + i * step) * dir_cam;
            double dx = (bb0[2]-bb0[0]) / K(0, 0);
            double dy = (bb0[3]-bb0[1]) / K(1, 1);
            double dz = (dx + dy) * 0.5;
            Eigen::Vector3d X_w = o0 * X_c + p0;
            Eigen::Vector3d axes(dx, dy, dz);
            samples.push_back(Ellipsoid(axes, o0, X_w));
            // samples.push_back(Ellipsoid(axes, Eigen::Matrix3d::Identity(), X));
        }

        vector<double> accu(n_samples, 0.0);

        while (!Rts.empty()) {
            auto Rt = Rts.top();
            auto bb = bboxes.top();
            Rts.pop();
            bboxes.pop();
            Matrix34d P = K * Rt;
            for (int i = 0; i < n_samples; ++i) {
                Ellipse ell = samples[i].project(P);
                double iou = bboxes_iou(ell.ComputeBbox(), bb);
                accu[i] += iou;
            }

            // write in file
            // for (auto a : accu){
            //     file << a << " ";
            // std::cout << a << " ";
            // }
            // file << "\n";
        }

        // file.close();
        // std::cout << "\n";

        int best_idx = std::distance(accu.begin(), std::max_element(accu.begin(), accu.end()));
        Ellipsoid ellipsoid = samples[best_idx];

        if (map_object_)
            map_object_->SetEllipsoid(ellipsoid);
        else
            map_object_ = std::make_unique<MapObject>(ellipsoid, this);

        if (status_ == ObjectTrackStatus::ONLY_2D)
            status_ = ObjectTrackStatus::INITIALIZED;
        return true;
    }


    bool ObjectTrack::ReconstructFromSamplesCenter()
    {
        std::ofstream file(std::to_string(id_) + "_reconstruction.txt");

        double dmin = 0.1;
        double dmax = 10.0;
        int n_samples = 1000;

        const Eigen::Matrix3d& K = tracker_->GetK();
        Eigen::Matrix3d K_inv = K.inverse();

        std::stack<Matrix34d> Rts;
        for (const auto& Rt : Rts_) {
            Rts.push(Rt);
        }

        std::stack<BBox2> bboxes;
        double mean_size = 0.0;
        for (auto const& bb : bboxes_) {
            bboxes.push(bb);
            mean_size = 0.5 * ((bb[2] - bb[0]) / K(0, 0) + (bb[3] - bb[1]) / K(1, 1));
        }
        mean_size /= bboxes_.size();

        auto Rt0 = Rts.top();
        auto bb0 = bboxes.top();
        Rts.pop();
        bboxes.pop();

        Eigen::Matrix3d o0 = Rt0.block<3, 3>(0, 0).transpose();
        Eigen::Vector3d p0 = -o0 * Rt0.block<3, 1>(0, 3);

        std::vector<Eigen::Vector3d> samples;
        // std::vector<Ellipsoid, Eigen::aligned_allocator<Ellipsoid>> samples;
        Eigen::Vector3d dir_cam = K_inv * bbox_center(bb0).homogeneous();
        double step = (dmax - dmin) / n_samples;
        for (int i = 0; i < n_samples; ++i) {
            Eigen::Vector3d X_c = (dmin + i * step) * dir_cam;
            Eigen::Vector3d X_w = o0 * X_c + p0;
            samples.push_back(X_w);
        }

        vector<double> accu(n_samples, 0.0);

        while (!Rts.empty()) {
            auto Rt = Rts.top();
            auto bb = bboxes.top();
            Eigen::Vector2d c = bbox_center(bb);
            Rts.pop();
            bboxes.pop();
            Matrix34d P = K * Rt;
            for (int i = 0; i < n_samples; ++i) {
                Eigen::Vector3d p = P * samples[i].homogeneous();
                p /= p[2];
                accu[i] += (p.head<2>() - c).norm();
            }
        }

        for (auto a : accu){
            file << a << "\n";
            // std::cout << a << " ";
        }
        file.close();
        // std::cout << "\n";

        int best_idx = std::distance(accu.begin(), std::min_element(accu.begin(), accu.end()));
        Eigen::Vector3d center = samples[best_idx];
        double d = (Rt0 * center.homogeneous()).z();
        mean_size *= d;
        Ellipsoid ellipsoid(Eigen::Vector3d(mean_size, mean_size, mean_size), Eigen::Matrix3d::Identity(), center);


        if (map_object_)
            map_object_->SetEllipsoid(ellipsoid);
        else
            map_object_ = std::make_unique<MapObject>(ellipsoid, this);

        if (status_ == ObjectTrackStatus::ONLY_2D)
            status_ = ObjectTrackStatus::INITIALIZED;
        return true;
    }

    void ObjectTrack::TryResetEllipsoidFromMaPoints()
    {
        if (!map_object_)
            return;


        std::unordered_map<MapPoint*, int> tmp;
        Eigen::Vector3d axes = map_object_->GetEllipsoid().GetAxes();
        double size_ratio = std::min(std::min(axes[0], axes[1]), axes[2]) / std::max(std::max(axes[0], axes[1]), axes[2]);
        //std::cout << "Object " << id_ << " size ratio = " << size_ratio << "\n";

        if (size_ratio < 0.01) {
            degenerated_ellipsoid_++;
        }

        if (degenerated_ellipsoid_ > 10) {
//            std::cout << ":!:!!!*********************************************************::!:!:!:!:!:!:!:\n";
//            std::cout << "Reconstruct from map points.\n";
            // Re-initialize ellipsoid using PCA on points
            if (associated_map_points_.size() < 5)
                return; // not enough points

            std::vector<Eigen::Vector3d> good_points;
            for (const auto& mp_cnt : associated_map_points_) {
                if (mp_cnt.second * 2 < keyframes_bboxes_.size()) {
                    Eigen::Vector3f p = mp_cnt.first->GetWorldPos();
                    Eigen::Vector3d pt(p.x(), p.y(), p.z());
                    good_points.push_back(pt);
                }
            }


            //std::cout << "nb points " << good_points.size() << "\n";

            Eigen::MatrixX3d pts(good_points.size(), 3);
            Eigen::Vector3d center(0, 0, 0);
            for (size_t i = 0; i < good_points.size(); ++i) {
                pts.row(i) = good_points[i].transpose();
                center += good_points[i].transpose();
            }
            //std::cout << "points\n" << pts << "\n";
            center /= pts.rows();
            //std::cout << "center = " << center.transpose() << "\n";
            Eigen::MatrixX3d pts_centered = pts.rowwise() - center.transpose();


            Eigen::Matrix3d M = pts_centered.transpose() * pts_centered;
            M /= pts_centered.cols();
            //std::cout << "M\n" << M << "\n";

            Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> solver(M);
            Eigen::Matrix3d R = solver.eigenvectors();
            Eigen::Vector3d s = 0.5 * solver.eigenvalues().cwiseAbs();//.cwiseSqrt();
            if (R.determinant() < 0) {
                R.col(2) *= -1;
            }
            //std::cout << "reconstructed axes = " << s << "\n";

            map_object_->SetEllipsoid(Ellipsoid(s, R, center));
            degenerated_ellipsoid_ = 0;
        }
        /*else {
            auto points = tracker_->GetSystem()->mpMap->GetAllMapPoints();
            Ellipsoid ell = map_object_->GetEllipsoid();
            for (auto* p : points)
            {
                cv::Mat pos = p->GetWorldPos();
                Eigen::Vector3d pt(pos.at<float>(0),pos.at<float>(1),pos.at<float>(2));
                if (ell.IsInside(pt)) {
                }
            }
        }*/
        // std::unique_lock<std::mutex> lock(mutex_associated_map_points_);
    }

    void ObjectTrack::AssociatePointsInsideEllipsoid(Atlas* map)
    {
        if (!map_object_)
            return;

        std::unique_lock<std::mutex> lock(mutex_associated_map_points_);
        associated_map_points_.clear();
        size_t nb_kf = keyframes_bboxes_.size();
        auto points = map->GetAllMapPoints();
        Ellipsoid ell = map_object_->GetEllipsoid();
        for (auto* p : points)
        {
            Eigen::Vector3f pos = p->GetWorldPos();
            Eigen::Vector3d pt(pos.x(),pos.y(),pos.z());
            if (ell.IsInside(pt)) {
                associated_map_points_[p] = nb_kf;
            }
        }
    }


}
