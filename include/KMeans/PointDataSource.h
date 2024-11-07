//
// Created by user on 2023/2/27.
//

#ifndef MAINLOOP_POINTDATASOURCE_H
#define MAINLOOP_POINTDATASOURCE_H
#include"Point3.h"

namespace kmeans {

    template<typename T>
    class PointDataSource
    {
    public:
        PointDataSource(const Point3<T>* ptr, size_t count) :m_ptr(ptr), m_count(count) {

        }

        PointDataSource(const std::vector<Point3<T>>& points) :m_ptr(&points[0]), m_count(points.size()) {

        }

        PointDataSource() :m_ptr(nullptr), m_count(0) {

        }

        PointDataSource& operator= (const PointDataSource& other) = default;

        ~PointDataSource() {

        }

        size_t size() const {
            return m_count;
        }

        const Point3<T>& operator[](size_t index) const{
            return m_ptr[index];
        }

        const Point3<T>* begin() const {
            return m_ptr;
        }

        const Point3<T>* end() const {
            return m_ptr + m_count;
        }

    private:
        const Point3<T>* m_ptr;
        size_t m_count;
    };
}
#endif //MAINLOOP_POINTDATASOURCE_H
