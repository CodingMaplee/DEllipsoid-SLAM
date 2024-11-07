#include "ColorManager.h"


namespace ORB_SLAM3
{

std::random_device RandomUniformColorGenerator::rd("default");
std::mt19937 RandomUniformColorGenerator::gen(RandomUniformColorGenerator::rd());
std::uniform_int_distribution<int> RandomUniformColorGenerator::distr = std::uniform_int_distribution<int>(0, 255);


CategoryColorsManager *CategoryColorsManager::instance = nullptr;


CategoryColorsManager::CategoryColorsManager()
    : colors_(std::vector<cv::Scalar>(nb_category_colors)) {
    for (int i = 0; i < nb_category_colors; ++i) {
        colors_[i] = RandomUniformColorGenerator::Generate();
    }
}

}
