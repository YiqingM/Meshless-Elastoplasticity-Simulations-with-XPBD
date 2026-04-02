#include "constitutive.h"
#include <cmath>

learnSPH::constitutive::ConstraintResult learnSPH::constitutive::evaluateStVKHencky(
                const Eigen::Matrix3d& U,
                const Eigen::Vector3d& S,
                const Eigen::Matrix3d& V,
                double mu,
                double lambda)
{
    // calculate Hencky strain
    const Eigen::Vector3d henckystrain = S.array().log();

    // calculate trace of Hencky strain and trace of squared Hencky strain
    const double trace_hencky = henckystrain.sum();
    const double trace_hencky_squared = henckystrain.squaredNorm();

    // calculate psi
    const double psi = mu * trace_hencky_squared + 0.5 * lambda * trace_hencky * trace_hencky;

    // calculate constraint value C
    const double C = std::sqrt(2.0 * psi);

    if (C < 1e-8) {
        // Avoid division by zero, return zero gradient
        return {0.0, Eigen::Matrix3d::Zero()};
    }

    // calculate gradient dC/dF^E
    Eigen::Matrix3d dPsidS = Eigen::Matrix3d::Zero();
    for (size_t i = 0; i < 3; i++)
    {
        dPsidS(i, i) = (2.0 * mu * henckystrain[i] + lambda * trace_hencky) / S[i];
    }

    // dC/dF = P / C
    Eigen::Matrix3d dCdF = (U * dPsidS * V.transpose()) / C;

    return {C, dCdF};
}