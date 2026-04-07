#pragma once
#include <Eigen/Dense>

namespace learnSPH
{
    namespace plasticity
    {
        struct PlasticityResult
        {
            Eigen::Vector3d S; // singular values of F^E after plastic projection
            double deltaGamma; // plastic correction amount
        };

        PlasticityResult vonMisesReturnMapping(
            const Eigen::Vector3d& S_trial, // singular values of trial elastic deformation gradient F^E_trial
            double mu,
            double ksai, // hardening parameter
            double q); // q is the evolving yield stress
        
        PlasticityResult druckerPragerReturnMapping(
            const Eigen::Vector3d& S_trial, // singular values of trial elastic deformation gradient F^E_trial
            double mu,
            double lambda,
            double alpha, // friction angle parameter
            double cohesion,
            double diff_log_J); // cohesion parameter to provide chuncky behavior
    }
}