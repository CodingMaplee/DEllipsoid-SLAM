//
// Created by user on 2023/2/27.
//

#ifndef MAINLOOP_POINT3_H
#define MAINLOOP_POINT3_H
#include<math.h>

namespace kmeans {

    template<typename T>
    class Point3
    {
    public:
        T x, y, z;

        Point3() = default;
        Point3(T x, T y, T z) :x(x), y(y), z(z) {}
        ~Point3() {}

        Point3 operator- (const Point3& other) {
            return Point3(x - other.x, y - other.y, z - other.z);
        }

        Point3 operator+ (const Point3& other) {
            return Point3(x + other.x, y + other.y, z + other.z);
        }

        T getSquaredDistanceTo(const Point3& other) {
            const T dx = x - other.x;
            const T dy = y - other.y;
            const T dz = z - other.z;
            return dx*dx + dy*dy + dz*dz;
        }

    private:

    };

}
#endif //MAINLOOP_POINT3_H
