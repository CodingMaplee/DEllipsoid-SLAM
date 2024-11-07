//
// Created by user on 2023/6/1.
//

#ifndef MAINLOOP_LOCALOBJECTMAPPING_H
#define MAINLOOP_LOCALOBJECTMAPPING_H

#include "KeyFrame.h"
#include "Atlas.h"
#include "LoopClosing.h"
#include "Tracking.h"
#include "KeyFrameDatabase.h"
#include "Settings.h"
#include <mutex>

namespace ORB_SLAM3
{

    class Tracking;
    class LoopClosing;
    class Map;
    class MapPoint;
    class MapObject;
    class Atlas;

    class LocalObjectMapping
    {
    public:
        LocalObjectMapping(Atlas* pMap, Tracking* tracker);

        void SetLoopCloser(LoopClosing* pLoopCloser);

        // Main function
        void Run();

        void InsertKeyFrame(KeyFrame* pKF);

        // Thread Synch
        void RequestStop();
        void RequestReset();
        bool Stop();
        void Release();
        bool isStopped();
        bool stopRequested();
        bool AcceptKeyFrames();
        void SetAcceptKeyFrames(bool flag);
        bool SetNotStop(bool flag);

        void InterruptBA();

        void RequestFinish();
        bool isFinished();

        int KeyframesInQueue(){
            unique_lock<std::mutex> lock(mMutexNewKFs);
            return mlNewKeyFrames.size();
        }

        bool CheckModifiedObjects();
        void InsertModifiedObject(MapObject* obj);
        void OptimizeReconstruction(MapObject *obj);

    protected:

        bool CheckNewKeyFrames();
        void ProcessNewKeyFrame();
        void CreateNewMapPoints();

        void MapPointCulling();
        void SearchInNeighbors();

        void KeyFrameCulling();

        void ResetIfRequested();
        bool mbResetRequested;
        std::mutex mMutexReset;

        bool CheckFinish();
        void SetFinish();
        bool mbFinishRequested;
        bool mbFinished;
        std::mutex mMutexFinish;

        LoopClosing* mpLoopCloser;

        std::list<KeyFrame*> mlNewKeyFrames;

        KeyFrame* mpCurrentKeyFrame;

        std::list<MapPoint*> mlpRecentAddedMapPoints;

        std::mutex mMutexNewKFs;

        bool mbAbortBA;
        bool mbStopped;
        bool mbStopRequested;
        bool mbNotStop;

        std::mutex mMutexStop;

        Atlas* mpMap;
        Tracking* tracker_;
        // bool mbAcceptKeyFrames;
        // std::mutex mMutexAccept;

        std::queue<MapObject*> modified_objects_;
        std::unordered_set<MapObject*> modified_objects_set_;
    };

} //namespace ORB_SLAM
#endif //MAINLOOP_LOCALOBJECTMAPPING_H
