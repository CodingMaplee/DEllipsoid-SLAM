#include "Utils.h"

#include <fstream>
#include <iomanip>


namespace ORB_SLAM3
{

    void writeOBJ(const std::string& filename, const Eigen::Matrix<double, Eigen::Dynamic, 3>& pts,
                  const Eigen::Matrix<int, Eigen::Dynamic, 3>& colors)
    {
        std::ofstream f(filename);
        f << std::fixed;
        bool with_colors = pts.rows() == colors.rows();
        for (int j = 0; j < pts.rows(); ++j) {
            f << "v " << std::setprecision(7) << " "<< pts(j, 0)
              << " " << pts(j, 1)
              << " " << pts(j, 2);
            if (with_colors) {
                f << " " << colors(j, 0) << " " << colors(j, 1) << " " << colors(j, 2);
            }
            f << "\n";
        }
        f.close();
    }

}
