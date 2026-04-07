#pragma once
#include <Eigen/Dense>

namespace learnSPH
{
    namespace constitutive
    {
        struct Config
        {
            double mu;
            double lambda;
            bool granular = false;
            double cohesion = 0.0;
        };
        struct ConstraintResult
        {
            double C; // constraint value: sqrt(2*psi(F^E))
            Eigen::Matrix3d dCdF; // gradient of C w.r.t. F^E
        };

        ConstraintResult evaluateStVKHencky(
                const Eigen::Matrix3d& U,
                const Eigen::Vector3d& S,
                const Eigen::Matrix3d& V,
                const Config& config);

    }
}