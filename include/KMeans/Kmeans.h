//
// Created by user on 2023/2/27.
//

#ifndef MAINLOOP_KMEANS_H
#define MAINLOOP_KMEANS_H
#include<vector>
#include"Point3.h"
#include"PointDataSource.h"

namespace kmeans {

    template<typename T>
    class  Kmeans
    {
    private:
        int k;
        PointDataSource<T> pointCloud;

        Point3<T> UpdateCenterPoint(const std::vector<size_t>& clusterPoints) {
            T x = 0, y = 0, z = 0;
            for (size_t i = 0; i < clusterPoints.size(); i++)
            {
                x += pointCloud[clusterPoints[i]].x;
                y += pointCloud[clusterPoints[i]].y;
                z += pointCloud[clusterPoints[i]].z;
            }

            return Point3<T>(x / clusterPoints.size(), y / clusterPoints.size(), z / clusterPoints.size());
        }

    public:

        Kmeans() = default;

        Kmeans(int k,const std::vector<Point3<T>>& pointCloud) :k(k),pointCloud(pointCloud) {

        }

        ~Kmeans()  {

        }

        std::vector<std::vector<size_t>> GetClusterPoint() {
            std::vector<std::vector<size_t>> cluster;
            cluster.resize(k);

            for (int i = 0; i < k; i++)
            {
                size_t index = rand() % pointCloud.size();
                cluster[i].push_back(index);
            }

            std::vector<Point3<T>> clusterCenter,oldClusterCenter;
            clusterCenter.resize(k);
            for (int i = 0; i < k; i++)
            {
                clusterCenter[i] = pointCloud[cluster[i][0]];
            }

            while (true)
            {
                for (size_t i = 0; i < pointCloud.size(); i++)
                {
                    Point3<T> point = pointCloud[i];
                    T distance = INFINITY, temp = distance;
                    size_t maxDistIndex = -1;
                    for (int j = 0; j < k; j++)
                    {
                        distance = point.getSquaredDistanceTo(clusterCenter[j]);
                        if (distance<temp)
                        {
                            temp = distance;
                            maxDistIndex = j;
                        }
                    }
                    cluster[maxDistIndex].push_back(i);
                }

                oldClusterCenter = clusterCenter;
                int unUpdateflag = 0;
                for (int i = 0; i < k; i++)
                {
                    clusterCenter[i] = UpdateCenterPoint(cluster[i]);
                    if (clusterCenter[i].getSquaredDistanceTo(oldClusterCenter[i])<0.09)
                    {
                        unUpdateflag++;
                    }
                }

                if (unUpdateflag == k)
                {
                    break;
                }

                for (int i = 0; i < k; i++)
                {
                    cluster[i].clear();
                }
            }
            return cluster;

        }

    };
}

#endif //MAINLOOP_KMEANS_H
