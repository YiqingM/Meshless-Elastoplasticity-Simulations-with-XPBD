#include "constitutive.h"
#include <cmath>

learnSPH::constitutive::ConstraintResult learnSPH::constitutive::evaluateStVKHencky(
                const Eigen::Matrix3d& U,
                const Eigen::Vector3d& S,
                const Eigen::Matrix3d& V,
                Config const& config)
{
    const Eigen::Vector3d strain = S.array().log();
    const double trace = strain.sum();

    double psi;
    Eigen::Matrix3d dPsidS = Eigen::Matrix3d::Zero();

    if (config.granular && trace >= 3.0 * config.cohesion)
    {
        const Eigen::Vector3d strain_dev = strain - (trace / 3.0) * Eigen::Vector3d::Ones();
        psi = config.mu * strain_dev.squaredNorm();

        for (int i = 0; i < 3; i++)
            dPsidS(i, i) = 2.0 * config.mu * strain_dev[i] / S[i];
    }
    else
    {
        psi = config.mu * strain.squaredNorm() + 0.5 * config.lambda * trace * trace;

        for (int i = 0; i < 3; i++)
            dPsidS(i, i) = (2.0 * config.mu * strain[i] + config.lambda * trace) / S[i];
    }

    const double C = std::sqrt(2.0 * psi);

    if (C < 1e-8)
        return {0.0, Eigen::Matrix3d::Zero()};

    Eigen::Matrix3d dCdF = (U * dPsidS * V.transpose()) / C;
    return {C, dCdF};
}