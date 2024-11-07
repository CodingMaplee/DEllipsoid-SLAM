#include "MapObject.h"
#include "ObjectTrack.h"


namespace ORB_SLAM3
{

    bool MapObject::Merge(MapObject* obj) {
        // do something with the ellipsoid
        // for now just keep the initial one
        // std::unique_lock<std::mutex> lock(mutex_ellipsoid_);

        this->GetTrack()->Merge(obj->GetTrack());
        auto ret = this->GetTrack()->ReconstructFromCenter(true);
        return ret;
        //  auto checked = tr->CheckReprojectionIoU(0.3);

    }

    void MapObject::RemoveKeyFrameObservation(KeyFrame* kf)
    {
        this->object_track_->RemoveKeyFrame(kf);
    }

}
