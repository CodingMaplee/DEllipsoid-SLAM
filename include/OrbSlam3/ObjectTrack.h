//
// Created by user on 2023/5/17.
//
#pragma once
#ifndef MAINLOOP_OBJECTTRACK_H
#define MAINLOOP_OBJECTTRACK_H
#include "Utils.h"

#include <memory>
#include <list>
#include <iostream>
#include <Eigen/Dense>
#include<opencv2/core/core.hpp>

#include "g2o/core/base_vertex.h"
#include "g2o/core/base_unary_edge.h"
#include "g2o/core/sparse_optimizer.h"
#include "g2o/core/block_solver.h"
#include "g2o/core/solver.h"
#include "g2o/core/optimization_algorithm_levenberg.h"
#include "g2o/solvers/dense/linear_solver_dense.h"
#include "g2o/types/sim3/types_seven_dof_expmap.h"
# include "g2o/solvers/eigen/linear_solver_eigen.h"

#include "Distance.h"
#include "Ellipse.h"
#include "Ellipsoid.h"
#include "Map.h"
#include "RingBuffer.h"
#include "MapObject.h"
# include "Atlas.h"
static const size_t max_frames_history = 50;

namespace ORB_SLAM3 {
    class Tracking;

    enum class DynamicObjectTrack
    {
        static_object,
        dynamic_object,
    };

    enum class ObjectTrackStatus {
        ONLY_2D,  //1
        INITIALIZED, //2
        IN_MAP, //3
        BAD, //4
//        Dynamic //5
    };

    class ObjectTrack {
    public:
        typedef std::shared_ptr<ObjectTrack> Ptr;
        static unsigned int factory_id;

        static ObjectTrack::Ptr
        CreateNewObjectTrack(unsigned int cat, const BBox2 &bbox, double score, const Matrix34d &Rt, const cv::Mat &imD, const Eigen::Vector4d& centerPoint_cam,
                             unsigned int frame_idx, Tracking *tracker, KeyFrame *kf);

        bool ReconstructFromLandmarks(Atlas *map);

        bool ReconstructCrocco(bool use_two_passes = true);

        bool ReconstructFromCenter(bool use_keyframes = false);

        bool ReconstructFromSamplesEllipsoid();

        bool ReconstructFromSamplesCenter();

        void AddDetection(const BBox2 &bbox, double score, Eigen::Vector4d centerPoint_cam, const Matrix34d &Rt, unsigned int frame_idx, KeyFrame *kf);

        unsigned int GetCategoryId() const {
            return category_id_;
        }

        unsigned int GetId() const {
            return id_;
        }

        // ------- For osmap ------
        void SetId(unsigned int id) {
            id_ = id;
        }

        void SetLastObsFrameId(int frame_id) {
            last_obs_frame_id_ = frame_id;
        }

        void SetColor(const cv::Scalar &color) {
            color_ = color;
        }

        void SetStatus(ObjectTrackStatus status) {
            status_ = status;
        }

        void SetMapObject(MapObject *map_obj) {
            map_object_ = std::unique_ptr<MapObject>(map_obj);
        }
        // ------------------------

        BBox2 GetLastBbox() const {
            return bboxes_.front();
        }

        size_t GetLastObsFrameId() const {
            return last_obs_frame_id_;
        }

        size_t GetNbObservations() const {
            return bboxes_.size();
        }

        size_t GetNbObservationsInKeyFrame() const {
            return keyframes_bboxes_.size();
        }

        cv::Scalar GetColor() const {
            return color_;
        }

        MapObject *GetMapObject() {
            return map_object_.get();
        }

        ~ObjectTrack();


        void OptimizeReconstruction(Atlas *map);

        bool TrackDynamic(void);

        bool CheckReprojectionIoU(double iou_threshold);

        bool CheckProjectIOU(double iou_threshold);

        bool CheckPreviousIOU(double iou_threshold);

        double CheckReprojectionIoUInKeyFrames(double iou_threshold);

        // void OptimizeReconstructionCeres(Map* map);

        // copy detections in keyframes and remove bad keyframes
        std::pair<std::vector<BBox2, Eigen::aligned_allocator<BBox2>>, std::vector<Matrix34d, Eigen::aligned_allocator<Matrix34d>>>
        CopyDetectionsInKeyFrames();

        std::vector<CenterPoint3d, Eigen::aligned_allocator<CenterPoint3d>>
        CopyCenterPoint3dInKeyFrames();
        void CleanBadKeyFrames();

        void RemoveKeyFrame(KeyFrame *kf) {
            unique_lock<mutex> lock(mutex_add_detection_);
            keyframes_bboxes_.erase(kf);
            keyframes_scores_.erase(kf);
            keyframes_CenterPoint3d_.erase(kf);
        }

        bool IsBad() const {
            std::unique_lock<std::mutex> lock(mutex_status_);
            return status_ == ObjectTrackStatus::BAD;
        }
        bool IsDynamic() const {
            std::unique_lock<std::mutex> lock(mutex_status_);
            return dynamic_status_ == DynamicObjectTrack::dynamic_object;
        }


        void SetIsBad() {
            std::unique_lock<std::mutex> lock(mutex_status_);
            status_ = ObjectTrackStatus::BAD;
        }

        void SetIsDynamic() {
            std::unique_lock<std::mutex> lock(mutex_status_);
            dynamic_status_ = DynamicObjectTrack::dynamic_object;
        }
        void SetIsStatic() {
            std::unique_lock<std::mutex> lock(mutex_status_);
            dynamic_status_ = DynamicObjectTrack::static_object;
        }
        void DynamicInitialization(vector<Eigen::Vector2i> mvPoint2d, vector<Eigen::Vector3f> mvPoint3d, int frame_id){
            mvPointReference2d_ = mvPoint2d;
            mvPointReference3d_ = mvPoint3d;
            ReferenceFrameId_ = frame_id;
        }
        ObjectTrackStatus GetStatus() const {
            std::unique_lock<std::mutex> lock(mutex_status_); // REALLY NEEDED ?
            return status_;
        }

        double GetAngularDifference() const;

        void Merge(ObjectTrack *track);

        void UnMerge(ObjectTrack *track);

        void InsertInMap(Atlas *map) {
            map->AddMapObject(map_object_.get());
            status_ = ObjectTrackStatus::IN_MAP;
        }

        void ClearTrackingBuffers() {
            last_obs_frame_id_ = -1;
            bboxes_.clear();
            scores_.clear();
            Rts_.clear();
        }

        double GetLastObsScore() const {
            return last_obs_score_;
        }

        std::pair<std::unordered_map<KeyFrame *, BBox2,
                std::hash<KeyFrame *>,
                std::equal_to<KeyFrame *>,
                Eigen::aligned_allocator<std::pair<KeyFrame const *, BBox2>>>,
                std::unordered_map<KeyFrame *, double>>
        CopyDetectionsMapInKeyFrames() const {
            unique_lock<mutex> lock(mutex_add_detection_);
            return {keyframes_bboxes_, keyframes_scores_};
        }

        void TryResetEllipsoidFromMaPoints();

        void AssociatePointsInsideEllipsoid(Atlas *map);


        std::unordered_map<MapPoint *, int> GetAssociatedMapPoints() const {
            std::unique_lock<std::mutex> lock(mutex_associated_map_points_);
            return associated_map_points_;
        }

        std::unordered_map<MapPoint *, int> GetFilteredAssociatedMapPoints(int threshold) const {
            std::unique_lock<std::mutex> lock(mutex_associated_map_points_);
            int limit = threshold; //  std::max(10.0, threshold * keyframes_bboxes_.size());

            std::unordered_map<MapPoint *, int> filtered;
            for (auto mp_cnt: associated_map_points_) {
                if (mp_cnt.second > limit) {
                    filtered[mp_cnt.first] = mp_cnt.second;
                }
            }

            if (map_object_) {
                const auto &ellipsoid = map_object_->GetEllipsoid();
                for (auto mp_cnt: associated_map_points_) {
                    Eigen::Vector3f p = mp_cnt.first->GetWorldPos();
                    Eigen::Vector3d pos(p.x(), p.y(), p.z());
                    if (ellipsoid.IsInside(pos))
                        filtered[mp_cnt.first] = mp_cnt.second;
                }
            }

            return filtered;
        }


        // protected:
    public:
        unsigned int category_id_;
        unsigned int id_;
        int last_obs_frame_id_ = -1; // no unsigned int: need to ba able to use -1
        double last_obs_score_ = 0.0;
        cv::Scalar color_;
        std::unique_ptr<MapObject> map_object_ = nullptr;
        Tracking *tracker_ = nullptr;

        RingBuffer <Eigen::Matrix4d, Eigen::aligned_allocator<Eigen::Matrix4d>> dynamic_poses_ = RingBuffer <Eigen::Matrix4d, Eigen::aligned_allocator<Eigen::Matrix4d>>(
                max_frames_history); //aligned with bboxes_ when detected dynamic.
        RingBuffer <BBox2, Eigen::aligned_allocator<BBox2>> bboxes_ = RingBuffer<BBox2, Eigen::aligned_allocator<BBox2>>(
                max_frames_history);
        RingBuffer <Matrix34d, Eigen::aligned_allocator<Matrix34d>> Rts_ = RingBuffer<Matrix34d, Eigen::aligned_allocator<Matrix34d>>(
                max_frames_history);
        RingBuffer<double> scores_ = RingBuffer<double>(max_frames_history);
        RingBuffer <CenterPoint3d , Eigen::aligned_allocator<CenterPoint3d>> CenterPoint_ = RingBuffer<CenterPoint3d, Eigen::aligned_allocator<CenterPoint3d>>(
                max_frames_history);
        std::unordered_map<KeyFrame *, BBox2,
                std::hash<KeyFrame *>,
                std::equal_to<KeyFrame *>,
                Eigen::aligned_allocator<std::pair<KeyFrame const *, BBox2>>> keyframes_bboxes_;
        std::unordered_map<KeyFrame *, CenterPoint3d ,
                std::hash<KeyFrame *>,
                std::equal_to<KeyFrame *>,
                Eigen::aligned_allocator<std::pair<KeyFrame const *, CenterPoint3d>>> keyframes_CenterPoint3d_;
        std::unordered_map<KeyFrame *, double> keyframes_scores_;

        mutable std::mutex mutex_add_detection_;
        mutable std::mutex mutex_status_;
        mutable std::mutex mutex_associated_map_points_;
        ObjectTrackStatus status_ = ObjectTrackStatus::ONLY_2D;
        DynamicObjectTrack dynamic_status_ = DynamicObjectTrack::static_object;
        double unc_ = 0.0;
        std::unordered_map<MapPoint *, int> associated_map_points_;
        int degenerated_ellipsoid_ = 0;
        double Dynamic_score = 0;
        vector<Eigen::Vector2i> mvPointReference2d_;
        vector<Eigen::Vector3f> mvPointReference3d_;
        int ReferenceFrameId_ = 0;
    };
}


#endif //MAINLOOP_OBJECTTRACK_H
