//
// Created by user on 2023/5/17.
//

#ifndef MAINLOOP_MAPOBJECT_H
#define MAINLOOP_MAPOBJECT_H
#include "Utils.h"

#include <random>
#include <memory>
#include <list>
#include <iostream>

#include <Eigen/Dense>


#include "Ellipse.h"
#include "Ellipsoid.h"
#include "Map.h"

namespace ORB_SLAM3
{

    class ObjectTrack;

    class MapObject
    {
    public:
        MapObject(const Ellipsoid& ellipsoid, ObjectTrack* track) : ellipsoid_(ellipsoid), object_track_(track) {
        }

        ObjectTrack* GetTrack() const {
            return object_track_;
        }

        const Ellipsoid& GetEllipsoid() const {
            std::unique_lock<std::mutex> lock(mutex_ellipsoid_);
            return ellipsoid_;
        }


        void SetEllipsoid(const Ellipsoid& ell) {
            std::unique_lock<std::mutex> lock(mutex_ellipsoid_);
            ellipsoid_ = ell;
        }

        bool Merge(MapObject* obj);

        void RemoveKeyFrameObservation(KeyFrame* kf);



    protected:
        Ellipsoid ellipsoid_;
        ObjectTrack *object_track_ = nullptr;
        mutable std::mutex mutex_ellipsoid_;

        MapObject() = delete;
    };


}



#endif //MAINLOOP_MAPOBJECT_H
