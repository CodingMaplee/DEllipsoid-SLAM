//
// Created by user on 2023/7/13.
//

#ifndef MAINLOOP_P3P_H
#define MAINLOOP_P3P_H
#include <Eigen/Dense>
#include <math.h>
#include <stdlib.h>
namespace monocular_pose_estimator
{

/**
 * The P3P class that solves the Perspective from Three Points (P3P) problem.
 *
 * The absolute pose of a camera is calculated using three 3D-to-2D correspondences using an implementation of Laurent Kneip's P3P algorithm.
 *
 * \note The Eigen linear algebra library is used instead of the TooN library that Laurent Kneip used in his original implementation.
 *
 * \author Karl Schwabe
 * \author Laurent Kneip (original author) (http://www.laurentkneip.de)
 *
 * \cite Kneip:2011 Reference: A Novel Parametrization of the Perspective-Three-Point Problem for a Direct Computation of Absolute Camera Position and Orientation. DOI: 10.1109/CVPR.2011.5995464
 *
 *
 */
    class P3P
    {
    public:

        /**
         * Solves the P3P problem.
         *
         * Computes the four possible poses for the P3P problem, given three 3D-to-2D point correspondences.
         *
         * For further details on how the algorithm works, see \cite Kneip:2011.
         *
         * \param feature_vectors a 3x3 matrix containing the unit vectors that point from the centre of projection of the camera to the world points used in the P3P algorithm. Each column of the matrix represents a unit vector.
         * \param world_points the 3D coordinates of the world points that are used in the P3P algorithm
         * \param solutions (output) the four solutions to the P3P problem. It is passed as a vector of 4 elements, where each element is a 3x4 matrix that contains the solutions of the form: \n
         *                             [ [3x4] (3x3 rotation matrix and 3x1 position vector (solution1));\n
         *                               [3x4] (3x3 rotation matrix and 3x1 position vector (solution2));\n
         *                               [3x4] (3x3 rotation matrix and 3x1 position vector (solution3));\n
         *                               [3x4] (3x3 rotation matrix and 3x1 position vector (solution4)) ]\n
         *                             The obtained orientation matrices are defined as transforming points from the camera to the world frame
         *
         * \returns
         * - \b 0 if executed correctly
         * - \b -1 if the world points are colinear and it was unable to solve the P3P problem
         *
         */
        static int computePoses(const Eigen::Matrix3d & feature_vectors, const Eigen::Matrix3d & world_points,
                                Eigen::Matrix<Eigen::Matrix<double, 3, 4>, 4, 1> & solutions);

        /**
         * Solves a quartic equation.
         *
         * Computes the solution to the quartic equation \f$a_4 \cdot x^4 + a_3 \cdot x^3 + a_2 \cdot x^2 + a_1 \cdot x + a_0 = 0\f$ using Ferrari's closed form solution for finding the roots of a fourth order polynomial.
         *
         * For further details on how the algorithm works, see \cite Kneip:2011.
         *
         * \param factors vector of the coefficients of \f$x\f$, listed in decending powers of \f$x\f$, i.e., factors(0)&nbsp;=&nbsp;\f$a_4\f$, factors(1)&nbsp;=&nbsp;\f$a_3\f$, factors(2)&nbsp;=&nbsp;\f$a_2\f$, factors(3)&nbsp;=&nbsp;\f$a_1\f$, and factors(4)&nbsp;=&nbsp;\f$a_0\f$
         * \param real_roots (output) vector containing the four solutions to the quartic equation
         *
         * \returns
         * \b 0 if executed correctly
         *
         */
        static int solveQuartic(const Eigen::Matrix<double, 5, 1> & factors, Eigen::Matrix<double, 4, 1> & real_roots);
    };

}
#endif //MAINLOOP_P3P_H
