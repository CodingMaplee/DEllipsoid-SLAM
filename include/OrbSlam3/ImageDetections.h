//
// Created by user on 2023/5/28.
//

#ifndef MAINLOOP_IMAGEDETECTIONS_H
#define MAINLOOP_IMAGEDETECTIONS_H


#include <fstream>
#include <iostream>
#include <memory>
#include <Eigen/Dense>

#include "Utils.h"
namespace ORB_SLAM3
{

    class Detection
    {
    public:
        typedef std::shared_ptr<Detection> Ptr;
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
        Detection(){
            mask = new uchar[640 * 480];
            score=100.0;
        }
//        Detection(unsigned int cat, double det_score, const BBox2 bb)
//                : category_id(cat), score(det_score), bbox(bb) {}

        bool GenCenterPoint(cv::Mat imD,  Eigen::Matrix4f intrinsics)
        {
            Eigen::Vector2d pointCenter2d = bbox_center(bbox);

            int u = (int)pointCenter2d.x();
            int v = (int)pointCenter2d.y();
            float depth = imD.at<float>(v,u);

//            if (depth==0) {
//                bool flag = false;
//                for (int i=-10; i<=10;i++){
//                    for (int j=-10; j<=10; j++){
//                        if (v+i>=0 && v+i<480 && u+j>=0 && u+j<640){
//                            depth = imD.at<float>(v+i, u+j);
//                            if (depth >0){
//                                flag = true;
//                                break;
//                            }
//                        }
//                    }
//                    if (flag == true) break;
//                }
//            }
            Eigen::Vector3d centerPoint3d;
            std::cout<<"category_id: "<<category_id<<" depth: "<<depth<<std::endl;
            depth = depth / 5000.0;
            //&& depth<=2.0
            ///delete wrong depth
           // if (depth>=2.3 && category_id == 6) return false;
            if (depth>0){
               //std::cout<<"category_id: "<<category_id<<" depth: "<<depth<<std::endl;
                centerPoint_cam = Eigen::Vector4d (depth * (u - intrinsics(0,2)) / intrinsics(0,0),
                                   depth * (v - intrinsics(1,2)) / intrinsics(1,1), depth, 1.0);
                std::cout<<"centerPoint_cam:"<<centerPoint_cam.x()<<" "<<centerPoint_cam.y()<<" "<<centerPoint_cam.z()<<std::endl;
                return true;
            }
            else{
                return false;
            }

        }



        friend std::ostream& operator <<(std::ostream& os, const Detection& det);
        unsigned int category_id;
        double score;
        BBox2 bbox;
        uchar* mask;
        Eigen::Vector4d centerPoint_cam;
    };

// using vector_Detection = std::vector<Detection, Eigen::aligned_allocator<Detection>>;

    class ImageDetectionsManager
    {
    public:
        ImageDetectionsManager();
        void SetDetection(std::vector<Detection::Ptr> detection);
        std::vector<Detection::Ptr> get_detections(unsigned int idx) const;

    private:
        //ImageDetectionsManager() = delete;
        std::vector<std::vector<Detection::Ptr>> detections_;

    };

}
#endif //MAINLOOP_IMAGEDETECTIONS_H
