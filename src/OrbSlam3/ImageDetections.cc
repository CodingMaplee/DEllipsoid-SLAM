#include "ImageDetections.h"


namespace ORB_SLAM3
{

std::ostream& operator <<(std::ostream& os, const Detection& det)
{
    os << "Detection:  cat = " << det.category_id << "  score = "
       << det.score << "  bbox = " << det.bbox.transpose();
    return os;
}

ImageDetectionsManager::ImageDetectionsManager()
{

}
void ImageDetectionsManager::SetDetection(std::vector<Detection::Ptr> detection)
{
    detections_.push_back(detection);
}


std::vector<Detection::Ptr> ImageDetectionsManager::get_detections(unsigned int idx) const {
    if (idx < 0 || idx >= detections_.size()) {
        std::cerr << "Warning invalid index: " << idx << std::endl;
        return {};
    }
    return detections_[idx];
}

}
