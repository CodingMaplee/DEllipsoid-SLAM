/**
* This file is part of ORB-SLAM3
*
* Copyright (C) 2017-2021 Carlos Campos, Richard Elvira, Juan J. Gómez Rodríguez, José M.M. Montiel and Juan D. Tardós, University of Zaragoza.
* Copyright (C) 2014-2016 Raúl Mur-Artal, José M.M. Montiel and Juan D. Tardós, University of Zaragoza.
*
* ORB-SLAM3 is free software: you can redistribute it and/or modify it under the terms of the GNU General Public
* License as published by the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* ORB-SLAM3 is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even
* the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License along with ORB-SLAM3.
* If not, see <http://www.gnu.org/licenses/>.
*/

#include "MapDrawer.h"
#include "MapPoint.h"
#include "KeyFrame.h"
#include "ColorManager.h"
#include <pangolin/pangolin.h>
#include <mutex>
#include "MapObject.h"
#include "ObjectTrack.h"
#include <unordered_map>
namespace ORB_SLAM3
{


MapDrawer::MapDrawer(Atlas* pAtlas, const string &strSettingPath, Settings* settings):mpAtlas(pAtlas)
{
    use_category_cols_ =  true;
    if(settings){
        newParameterLoader(settings);
    }
    else{
        cv::FileStorage fSettings(strSettingPath, cv::FileStorage::READ);
        bool is_correct = ParseViewerParamFile(fSettings);

        if(!is_correct)
        {
            std::cerr << "**ERROR in the config file, the format is not correct**" << std::endl;
            try
            {
                throw -1;
            }
            catch(exception &e)
            {

            }
        }
    }
}

void MapDrawer::newParameterLoader(Settings *settings) {
    mKeyFrameSize = settings->keyFrameSize();
    mKeyFrameLineWidth = settings->keyFrameLineWidth();
    mGraphLineWidth = settings->graphLineWidth();
    mPointSize = settings->pointSize();
    mCameraSize = settings->cameraSize();
    mCameraLineWidth  = settings->cameraLineWidth();
}

bool MapDrawer::ParseViewerParamFile(cv::FileStorage &fSettings)
{
    bool b_miss_params = false;

    cv::FileNode node = fSettings["Viewer.KeyFrameSize"];
    if(!node.empty())
    {
        mKeyFrameSize = node.real();
    }
    else
    {
        std::cerr << "*Viewer.KeyFrameSize parameter doesn't exist or is not a real number*" << std::endl;
        b_miss_params = true;
    }

    node = fSettings["Viewer.KeyFrameLineWidth"];
    if(!node.empty())
    {
        mKeyFrameLineWidth = node.real();
    }
    else
    {
        std::cerr << "*Viewer.KeyFrameLineWidth parameter doesn't exist or is not a real number*" << std::endl;
        b_miss_params = true;
    }

    node = fSettings["Viewer.GraphLineWidth"];
    if(!node.empty())
    {
        mGraphLineWidth = node.real();
    }
    else
    {
        std::cerr << "*Viewer.GraphLineWidth parameter doesn't exist or is not a real number*" << std::endl;
        b_miss_params = true;
    }

    node = fSettings["Viewer.PointSize"];
    if(!node.empty())
    {
        mPointSize = node.real();
    }
    else
    {
        std::cerr << "*Viewer.PointSize parameter doesn't exist or is not a real number*" << std::endl;
        b_miss_params = true;
    }

    node = fSettings["Viewer.CameraSize"];
    if(!node.empty())
    {
        mCameraSize = node.real();
    }
    else
    {
        std::cerr << "*Viewer.CameraSize parameter doesn't exist or is not a real number*" << std::endl;
        b_miss_params = true;
    }

    node = fSettings["Viewer.CameraLineWidth"];
    if(!node.empty())
    {
        mCameraLineWidth = node.real();
    }
    else
    {
        std::cerr << "*Viewer.CameraLineWidth parameter doesn't exist or is not a real number*" << std::endl;
        b_miss_params = true;
    }

    return !b_miss_params;
}

void MapDrawer::DrawMapPoints()
{
    Map* pActiveMap = mpAtlas->GetCurrentMap();

    if(!pActiveMap)
        return;

    const vector<MapPoint*> &vpMPs = pActiveMap->GetAllMapPoints();
    const vector<MapPoint*> &vpRefMPs = pActiveMap->GetReferenceMapPoints();

    set<MapPoint*> spRefMPs(vpRefMPs.begin(), vpRefMPs.end());

    if(vpMPs.empty())
        return;

    glPointSize(mPointSize);
    glBegin(GL_POINTS);
    glColor3f(0.0,0.0,0.0);

    for(size_t i=0, iend=vpMPs.size(); i<iend;i++)
    {
        if(vpMPs[i]->isBad() || spRefMPs.count(vpMPs[i]))
            continue;
        Eigen::Matrix<float,3,1> pos = vpMPs[i]->GetWorldPos();
        glVertex3f(pos(0),pos(1),pos(2));
    }
    glEnd();

    glPointSize(mPointSize);
    glBegin(GL_POINTS);
    glColor3f(1.0,0.0,0.0);

    for(set<MapPoint*>::iterator sit=spRefMPs.begin(), send=spRefMPs.end(); sit!=send; sit++)
    {
        if((*sit)->isBad())
            continue;
        Eigen::Matrix<float,3,1> pos = (*sit)->GetWorldPos();
        glVertex3f(pos(0),pos(1),pos(2));
    }

    glEnd();
}

void MapDrawer::DrawMapObjects(){

    const std::vector<MapObject*> objects = mpAtlas->GetAllMapObjects();
    glPointSize(mPointSize);
    const auto & color_manager = CategoryColorsManager::GetInstance();
    glLineWidth(2);
    for (auto*obj :objects){
        int c = obj->GetTrack()->GetCategoryId();
        const unsigned char COLORS[31][3] = {
                {255, 255, 255},     {0, 0, 255},     {255, 0, 0},   {0, 255, 0},     {255, 26, 184},  {255, 211, 0},   {0, 131, 246},  {0, 140, 70},
                {167, 96, 61}, {79, 0, 105},    {0, 255, 246}, {61, 123, 140},  {237, 167, 255}, {211, 255, 149}, {184, 79, 255}, {228, 26, 87},
                {131, 131, 0}, {0, 255, 149},   {96, 0, 43},   {246, 131, 17},  {202, 255, 0},   {43, 61, 0},     {0, 52, 193},   {255, 202, 131},
                {0, 43, 96},   {158, 114, 140}, {79, 184, 17}, {158, 193, 255}, {149, 158, 123}, {255, 123, 175}, {158, 8, 0}};

        //cout<<"ddd:  "<<obj->GetTrack()->GetCategoryId()<<std::endl;
        glColor3f(static_cast<double>(COLORS[c][0]) / 255,
                  static_cast<double>(COLORS[c][1]) / 255,
                  static_cast<double>(COLORS[c][2]) / 255);

        const Ellipsoid& ell = obj->GetEllipsoid();
        if(obj->GetTrack()->IsDynamic()){
            Eigen::Vector3d center_ = ell.GetCenter();

            EllipsoidTrajectory.push_back(center_);
            if(EllipsoidTrajectory.size() > 1)
            {
                glBegin(GL_LINE_STRIP);
                for(int i = 0;i<EllipsoidTrajectory.size();i++)
                {
                    glVertex3f(EllipsoidTrajectory[i][0], EllipsoidTrajectory[i][1], EllipsoidTrajectory[i][2]);
                }
                glEnd();
            }


        }
        if (!display_3d_bbox_) {
            auto pts = ell.GeneratePointCloud();
            int i = 0;
            while (i < pts.rows()) {
                glBegin(GL_LINE_STRIP);

                // glColor3f(0.0, 1.0, 0.0);
                // glBegin(GL_POINTS);
                for (int k = 0; k < 50; ++k, ++i){
                    glVertex3f(pts(i, 0), pts(i, 1), pts(i, 2));
                }
                glEnd();
            }
        }
        else {
            Eigen::Vector3d center = ell.GetCenter();
//            cout<<"center:"<<"x= "<<center.x()<<"y= "<<center.y()<<"z= "<<center.z()<<endl;
           Eigen::Vector3d axes = ell.GetAxes();
//            cout<<"axes: "<<"x= "<<axes.x()<<"y= "<<axes.y()<<"z= "<<axes.z()<<endl;
           Eigen::Matrix3d R = ell.GetOrientation();
//            cout<<"R:"<<R<<endl;
            Eigen::Matrix<double, 8, 3> pts;
            pts << -axes[0], -axes[1], -axes[2],
                    axes[0], -axes[1], -axes[2],
                    axes[0],  axes[1], -axes[2],
                    -axes[0],  axes[1], -axes[2],
                    -axes[0], -axes[1],  axes[2],
                    axes[0], -axes[1],  axes[2],
                    axes[0],  axes[1],  axes[2],
                    -axes[0],  axes[1],  axes[2];
            Eigen::Matrix<double, 8, 3> obb = (R * pts.transpose()).transpose();
            obb.rowwise() += center.transpose();
            glBegin(GL_LINE_STRIP);
//            cout<<"obb"<<endl;
//            cout<<obb(0,0)<<" y= "<<obb(0,1)<<" z= "<<obb(0,2)<<endl;
//            cout<<"x= "<<obb(1,0)<<" y= "<<obb(1,1)<<" z= "<<obb(1,2)<<endl;
//            cout<<"x= "<<obb(2,0)<<" y= "<<obb(2,1)<<" z= "<<obb(2,2)<<endl;
//            cout<<"x= "<<obb(3,0)<<" y= "<<obb(3,1)<<" z= "<<obb(3,2)<<endl;
//
//            cout<<"x= "<<obb(0,0)<<" y= "<<obb(0,1)<<" z= "<<obb(0,2)<<endl;
//            cout<<"x= "<<obb(4,0)<<" y= "<<obb(4,1)<<" z= "<<obb(4,2)<<endl;
//            cout<<"x= "<<obb(5,0)<<" y= "<<obb(5,1)<<" z= "<<obb(5,2)<<endl;
//            cout<<"x= "<<obb(1,0)<<" y= "<<obb(1,1)<<" z= "<<obb(1,2)<<endl;
//
//            cout<<"x= "<<obb(5,0)<<" y= "<<obb(5,1)<<" z= "<<obb(5,2)<<endl;
//            cout<<"x= "<<obb(6,0)<<" y= "<<obb(6,1)<<" z= "<<obb(6,2)<<endl;
//            cout<<"x= "<<obb(2,0)<<" y= "<<obb(2,1)<<" z= "<<obb(2,2)<<endl;
//            cout<<"x= "<<obb(6,0)<<" y= "<<obb(6,1)<<" z= "<<obb(6,2)<<endl;
//
//            cout<<"x= "<<obb(7,0)<<" y= "<<obb(7,1)<<" z= "<<obb(7,2)<<endl;
//            cout<<"x= "<<obb(3,0)<<" y= "<<obb(3,1)<<" z= "<<obb(3,2)<<endl;
//            cout<<"x= "<<obb(7,0)<<" y= "<<obb(7,1)<<" z= "<<obb(7,2)<<endl;
//            cout<<"x= "<<obb(4,0)<<" y= "<<obb(4,1)<<" z= "<<obb(4,2)<<endl;

            glVertex3f(obb(0, 0), obb(0, 1), obb(0, 2));
            glVertex3f(obb(1, 0), obb(1, 1), obb(1, 2));
            glVertex3f(obb(2, 0), obb(2, 1), obb(2, 2));
            glVertex3f(obb(3, 0), obb(3, 1), obb(3, 2));

            glVertex3f(obb(0, 0), obb(0, 1), obb(0, 2));
            glVertex3f(obb(4, 0), obb(4, 1), obb(4, 2));
            glVertex3f(obb(5, 0), obb(5, 1), obb(5, 2));
            glVertex3f(obb(1, 0), obb(1, 1), obb(1, 2));

            glVertex3f(obb(5, 0), obb(5, 1), obb(5, 2));
            glVertex3f(obb(6, 0), obb(6, 1), obb(6, 2));
            glVertex3f(obb(2, 0), obb(2, 1), obb(2, 2));
            glVertex3f(obb(6, 0), obb(6, 1), obb(6, 2));

            glVertex3f(obb(7, 0), obb(7, 1), obb(7, 2));
            glVertex3f(obb(3, 0), obb(3, 1), obb(3, 2));
            glVertex3f(obb(7, 0), obb(7, 1), obb(7, 2));
            glVertex3f(obb(4, 0), obb(4, 1), obb(4, 2));
            glEnd();
//            glBegin(GL_LINE_STRIP);
//            //glBegin(GL_LINE);
//            glVertex3f(obb(0, 0), obb(0, 1), obb(0, 2));
//            glVertex3f(obb(1, 0), obb(1, 1), obb(1, 2));
//            glVertex3f(obb(2, 0), obb(2, 1), obb(2, 2));
//            glVertex3f(obb(3, 0), obb(3, 1), obb(3, 2));
//
//            glVertex3f(obb(0, 0), obb(0, 1), obb(0, 2));
//            glVertex3f(obb(4, 0), obb(4, 1), obb(4, 2));
//            glVertex3f(obb(5, 0), obb(5, 1), obb(5, 2));
//            glVertex3f(obb(1, 0), obb(1, 1), obb(1, 2));
//
//            glVertex3f(obb(5, 0), obb(5, 1), obb(5, 2));
//            glVertex3f(obb(6, 0), obb(6, 1), obb(6, 2));
//            glVertex3f(obb(2, 0), obb(2, 1), obb(2, 2));
//            glVertex3f(obb(6, 0), obb(6, 1), obb(6, 2));
//
//            glVertex3f(obb(7, 0), obb(7, 1), obb(7, 2));
//            glVertex3f(obb(3, 0), obb(3, 1), obb(3, 2));
//            glVertex3f(obb(7, 0), obb(7, 1), obb(7, 2));
//            glVertex3f(obb(4, 0), obb(4, 1), obb(4, 2));
//
//            glEnd();
        }
    }
}

void MapDrawer::DrawMapObjectsPoints(double size)
{
    const vector<MapPoint*> &vpMPs = mpAtlas->GetAllMapPoints();
    const vector<MapPoint*> &vpRefMPs = mpAtlas->GetReferenceMapPoints();

    set<MapPoint*> spRefMPs(vpRefMPs.begin(), vpRefMPs.end());

    if(vpMPs.empty())
        return;

    const auto& color_manager = CategoryColorsManager::GetInstance();
    const std::vector<MapObject*> objects = mpAtlas->GetAllMapObjects();
    for (auto* obj : objects) {
        cv::Scalar c;
        if (use_category_cols_) {
            c = color_manager[obj->GetTrack()->GetCategoryId()];
        } else {
            c = obj->GetTrack()->GetColor();
        }
        glColor3f(static_cast<double>(c(2)) / 255,
                  static_cast<double>(c(1)) / 255,
                  static_cast<double>(c(0)) / 255);
        auto assoc_points = obj->GetTrack()->GetFilteredAssociatedMapPoints(10);
        glPointSize(size);
        glBegin(GL_POINTS);
        for (auto pt_cnt : assoc_points) {
            MapPoint* pt = pt_cnt.first;

            if (pt->isBad())
                continue;
            Eigen::Vector3f pos = pt->GetWorldPos();
            glVertex3f(pos.x(),pos.y(),pos.z());
        }
        glEnd();
    }
}

void MapDrawer::DrawKeyFrames(const bool bDrawKF, const bool bDrawGraph, const bool bDrawInertialGraph, const bool bDrawOptLba)
{
    const float &w = mKeyFrameSize;
    const float h = w*0.75;
    const float z = w*0.6;

    Map* pActiveMap = mpAtlas->GetCurrentMap();
    // DEBUG LBA
    std::set<long unsigned int> sOptKFs = pActiveMap->msOptKFs;
    std::set<long unsigned int> sFixedKFs = pActiveMap->msFixedKFs;

    if(!pActiveMap)
        return;

    const vector<KeyFrame*> vpKFs = pActiveMap->GetAllKeyFrames();

    if(bDrawKF)
    {
        for(size_t i=0; i<vpKFs.size(); i++)
        {
            KeyFrame* pKF = vpKFs[i];
            Eigen::Matrix4f Twc = pKF->GetPoseInverse().matrix();
            unsigned int index_color = pKF->mnOriginMapId;

            glPushMatrix();

            glMultMatrixf((GLfloat*)Twc.data());

            if(!pKF->GetParent()) // It is the first KF in the map
            {
                glLineWidth(mKeyFrameLineWidth*5);
                glColor3f(1.0f,0.0f,0.0f);
                glBegin(GL_LINES);
            }
            else
            {
                //cout << "Child KF: " << vpKFs[i]->mnId << endl;
                glLineWidth(mKeyFrameLineWidth);
                if (bDrawOptLba) {
                    if(sOptKFs.find(pKF->mnId) != sOptKFs.end())
                    {
                        glColor3f(0.0f,1.0f,0.0f); // Green -> Opt KFs
                    }
                    else if(sFixedKFs.find(pKF->mnId) != sFixedKFs.end())
                    {
                        glColor3f(1.0f,0.0f,0.0f); // Red -> Fixed KFs
                    }
                    else
                    {
                        glColor3f(0.0f,0.0f,1.0f); // Basic color
                    }
                }
                else
                {
                    glColor3f(0.0f,0.0f,1.0f); // Basic color
                }
                glBegin(GL_LINES);
            }

            glVertex3f(0,0,0);
            glVertex3f(w,h,z);
            glVertex3f(0,0,0);
            glVertex3f(w,-h,z);
            glVertex3f(0,0,0);
            glVertex3f(-w,-h,z);
            glVertex3f(0,0,0);
            glVertex3f(-w,h,z);

            glVertex3f(w,h,z);
            glVertex3f(w,-h,z);

            glVertex3f(-w,h,z);
            glVertex3f(-w,-h,z);

            glVertex3f(-w,h,z);
            glVertex3f(w,h,z);

            glVertex3f(-w,-h,z);
            glVertex3f(w,-h,z);
            glEnd();

            glPopMatrix();

            glEnd();
        }
    }

    if(bDrawGraph)
    {
        glLineWidth(mGraphLineWidth);
        glColor4f(0.0f,1.0f,0.0f,0.6f);
        glBegin(GL_LINES);

        // cout << "-----------------Draw graph-----------------" << endl;
        for(size_t i=0; i<vpKFs.size(); i++)
        {
            // Covisibility Graph
            const vector<KeyFrame*> vCovKFs = vpKFs[i]->GetCovisiblesByWeight(100);
            Eigen::Vector3f Ow = vpKFs[i]->GetCameraCenter();
            if(!vCovKFs.empty())
            {
                for(vector<KeyFrame*>::const_iterator vit=vCovKFs.begin(), vend=vCovKFs.end(); vit!=vend; vit++)
                {
                    if((*vit)->mnId<vpKFs[i]->mnId)
                        continue;
                    Eigen::Vector3f Ow2 = (*vit)->GetCameraCenter();
                    glVertex3f(Ow(0),Ow(1),Ow(2));
                    glVertex3f(Ow2(0),Ow2(1),Ow2(2));
                }
            }

            // Spanning tree
            KeyFrame* pParent = vpKFs[i]->GetParent();
            if(pParent)
            {
                Eigen::Vector3f Owp = pParent->GetCameraCenter();
                glVertex3f(Ow(0),Ow(1),Ow(2));
                glVertex3f(Owp(0),Owp(1),Owp(2));
            }

            // Loops
            set<KeyFrame*> sLoopKFs = vpKFs[i]->GetLoopEdges();
            for(set<KeyFrame*>::iterator sit=sLoopKFs.begin(), send=sLoopKFs.end(); sit!=send; sit++)
            {
                if((*sit)->mnId<vpKFs[i]->mnId)
                    continue;
                Eigen::Vector3f Owl = (*sit)->GetCameraCenter();
                glVertex3f(Ow(0),Ow(1),Ow(2));
                glVertex3f(Owl(0),Owl(1),Owl(2));
            }
        }

        glEnd();
    }

    if(bDrawInertialGraph && pActiveMap->isImuInitialized())
    {
        glLineWidth(mGraphLineWidth);
        glColor4f(1.0f,0.0f,0.0f,0.6f);
        glBegin(GL_LINES);

        //Draw inertial links
        for(size_t i=0; i<vpKFs.size(); i++)
        {
            KeyFrame* pKFi = vpKFs[i];
            Eigen::Vector3f Ow = pKFi->GetCameraCenter();
            KeyFrame* pNext = pKFi->mNextKF;
            if(pNext)
            {
                Eigen::Vector3f Owp = pNext->GetCameraCenter();
                glVertex3f(Ow(0),Ow(1),Ow(2));
                glVertex3f(Owp(0),Owp(1),Owp(2));
            }
        }

        glEnd();
    }

    vector<Map*> vpMaps = mpAtlas->GetAllMaps();

    if(bDrawKF)
    {
        for(Map* pMap : vpMaps)
        {
            if(pMap == pActiveMap)
                continue;
            //pMap->GetAllMapPoints()
            vector<KeyFrame*> vpKFs = pMap->GetAllKeyFrames();

            for(size_t i=0; i<vpKFs.size(); i++)
            {
                KeyFrame* pKF = vpKFs[i];
                Eigen::Matrix4f Twc = pKF->GetPoseInverse().matrix();
                unsigned int index_color = pKF->mnOriginMapId;

                glPushMatrix();

                glMultMatrixf((GLfloat*)Twc.data());

                if(!vpKFs[i]->GetParent()) // It is the first KF in the map
                {
                    glLineWidth(mKeyFrameLineWidth*5);
                    glColor3f(1.0f,0.0f,0.0f);
                    glBegin(GL_LINES);
                }
                else
                {
                    glLineWidth(mKeyFrameLineWidth);
                    glColor3f(mfFrameColors[index_color][0],mfFrameColors[index_color][1],mfFrameColors[index_color][2]);
                    glBegin(GL_LINES);
                }

                glVertex3f(0,0,0);
                glVertex3f(w,h,z);
                glVertex3f(0,0,0);
                glVertex3f(w,-h,z);
                glVertex3f(0,0,0);
                glVertex3f(-w,-h,z);
                glVertex3f(0,0,0);
                glVertex3f(-w,h,z);

                glVertex3f(w,h,z);
                glVertex3f(w,-h,z);

                glVertex3f(-w,h,z);
                glVertex3f(-w,-h,z);

                glVertex3f(-w,h,z);
                glVertex3f(w,h,z);

                glVertex3f(-w,-h,z);
                glVertex3f(w,-h,z);
                glEnd();

                glPopMatrix();
            }
        }
    }
}

void MapDrawer::DrawDistanceEstimation(double depth, cv::Mat &Tcw)
{
    if (Tcw.cols != 4)
        return;
    Eigen::Matrix4d T = cvToEigenMatrix<double, float, 4, 4>(Tcw);
    Eigen::Matrix3d R = T.block<3, 3>(0, 0);
    Eigen::Vector3d t = T.block<3, 1>(0, 3);
    Eigen::Matrix3d o = R.transpose();
    Eigen::Vector3d p = -o * t;
    Eigen::Vector3d e = p + depth * o.col(2);

    glColor3f(0.0, 0.0, 1.0);
    glBegin(GL_LINE_STRIP);
    glVertex3f(p[0], p[1], p[2]);
    glVertex3f(e[0], e[1], e[2]);
    glEnd();
}

void MapDrawer::DrawCurrentCamera(pangolin::OpenGlMatrix &Twc)
{
    const float &w = mCameraSize;
    const float h = w*0.75;
    const float z = w*0.6;

    glPushMatrix();

#ifdef HAVE_GLES
        glMultMatrixf(Twc.m);
#else
        glMultMatrixd(Twc.m);
#endif

    glLineWidth(mCameraLineWidth);
    glColor3f(0.0f,1.0f,0.0f);
    glBegin(GL_LINES);
    glVertex3f(0,0,0);
    glVertex3f(w,h,z);
    glVertex3f(0,0,0);
    glVertex3f(w,-h,z);
    glVertex3f(0,0,0);
    glVertex3f(-w,-h,z);
    glVertex3f(0,0,0);
    glVertex3f(-w,h,z);

    glVertex3f(w,h,z);
    glVertex3f(w,-h,z);

    glVertex3f(-w,h,z);
    glVertex3f(-w,-h,z);

    glVertex3f(-w,h,z);
    glVertex3f(w,h,z);

    glVertex3f(-w,-h,z);
    glVertex3f(w,-h,z);
    glEnd();

    glPopMatrix();
}


void MapDrawer::SetCurrentCameraPose(const Sophus::SE3f &Tcw)
{
    unique_lock<mutex> lock(mMutexCamera);
    mCameraPose = Tcw.inverse();
}

void MapDrawer::GetCurrentOpenGLCameraMatrix(pangolin::OpenGlMatrix &M, pangolin::OpenGlMatrix &MOw)
{
    Eigen::Matrix4f Twc;
    {
        unique_lock<mutex> lock(mMutexCamera);
        Twc = mCameraPose.matrix();
    }

    for (int i = 0; i<4; i++) {
        M.m[4*i] = Twc(0,i);
        M.m[4*i+1] = Twc(1,i);
        M.m[4*i+2] = Twc(2,i);
        M.m[4*i+3] = Twc(3,i);
    }

    MOw.SetIdentity();
    MOw.m[12] = Twc(0,3);
    MOw.m[13] = Twc(1,3);
    MOw.m[14] = Twc(2,3);
}
} //namespace ORB_SLAM
