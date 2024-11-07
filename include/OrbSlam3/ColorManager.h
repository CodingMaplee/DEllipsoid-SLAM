//
// Created by user on 2023/5/17.
//

#ifndef MAINLOOP_COLORMANAGER_H
#define MAINLOOP_COLORMANAGER_H
#include <random>
#include <iostream>

#include <opencv2/core.hpp>
namespace ORB_SLAM3
{



    class RandomUniformColorGenerator
    {
    public:
        static cv::Scalar Generate() {
            cv::Scalar color;
            color(0) = RandomUniformColorGenerator::distr(RandomUniformColorGenerator::gen);
            color(1) = RandomUniformColorGenerator::distr(RandomUniformColorGenerator::gen);
            color(2) = RandomUniformColorGenerator::distr(RandomUniformColorGenerator::gen);
            return color;
        }

    private:
        static std::random_device rd;
        static std::mt19937 gen;
        static std::uniform_int_distribution<int> distr;
    };

    const int nb_category_colors = 500;


    class CategoryColorsManager
    {
    public:
        static const CategoryColorsManager& GetInstance() {
            if (!instance) {
                instance = new CategoryColorsManager();
            }
            return *instance;
        }

        static void FreeInstance() {
            if (instance)
                delete instance;
        }

        const cv::Scalar& operator[](size_t idx) const {
            return colors_[idx];
        }

    private:
        static CategoryColorsManager *instance;

        std::vector<cv::Scalar> colors_;

        CategoryColorsManager();
    };


}


#endif //MAINLOOP_COLORMANAGER_H
