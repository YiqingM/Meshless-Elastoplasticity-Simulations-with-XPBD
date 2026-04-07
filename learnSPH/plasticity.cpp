#include "plasticity.h"
#include <cmath>

learnSPH::plasticity::PlasticityResult learnSPH::plasticity::vonMisesReturnMapping(
    const Eigen::Vector3d& S_trial,
    double mu,
    double ksai,
    double q)
{
    // compute the hencky strain
    const Eigen::Vector3d strain = S_trial.array().log();
    const Eigen::Vector3d strain_dev = strain - strain.mean() * Eigen::Vector3d::Ones();

    // trial yield condition
    double deltaGamma = (strain_dev.norm() - q / (2.0 * mu)) / (1.0 + ksai);

    // elastic case
    if (deltaGamma < 1e-12)
    {
        // elastic case, no plastic correction
        return {S_trial, 0.0};
    }

    // plastic case
    const Eigen::Vector3d strain_corrected = strain - deltaGamma * strain_dev.normalized();
    const Eigen::Vector3d S_corrected = strain_corrected.array().exp();
    return {S_corrected, deltaGamma};
}

learnSPH::plasticity::PlasticityResult
learnSPH::plasticity::druckerPragerReturnMapping(
    const Eigen::Vector3d& S_trial,
    double mu,
    double lambda,
    double alpha,
    double cohesion)
{
    // hencky strain
    const Eigen::Vector3d strain = S_trial.array().log();
    const Eigen::Vector3d strain_dev = strain - strain.mean() * Eigen::Vector3d::Ones();
    const double trace_strain = strain.sum();

    // yield condition
    double deltaGamma = 0.0;

    if (trace_strain > 3.0 * cohesion)
    {
        // tensile case
        return {Eigen::Vector3d::Constant(std::exp(cohesion)), 1.0};
    }
    else
    {
        // compressed case
        double yield_stress = - alpha * (3.0 * lambda + 2.0 * mu) * (trace_strain - 3.0 * cohesion);
        deltaGamma = strain_dev.norm() - yield_stress / (2.0 * mu);
    }

    if (deltaGamma < 1e-12)
    {
        // elastic case, no plastic correction
        return {S_trial, 0.0};
    }
    else
    {
        // plastic case
        const Eigen::Vector3d strain_corrected = strain - deltaGamma * strain_dev.normalized();
        const Eigen::Vector3d S_corrected = strain_corrected.array().exp();
        return {S_corrected, deltaGamma};
    }
}