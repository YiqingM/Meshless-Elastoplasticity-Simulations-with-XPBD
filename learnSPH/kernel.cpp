#include "kernel.h"

double learnSPH::kernel::h     = 0.0;
double learnSPH::kernel::sigma = 0.0;

void learnSPH::kernel::setSmoothingLength(double smoothing_length)
{
    assert(smoothing_length > 0.0);
    h     = smoothing_length;
    sigma = 21.0 / (16.0 * PI * h * h * h);
}

double learnSPH::kernel::Wendland_W(const Eigen::Vector3d& x)
{
    const double r = x.norm();
    const double q = r / h;
    if (q >= 2.0) return 0.0;
 
    const double s = 1.0 - 0.5 * q;
    return sigma * (s * s * s * s) * (1.0 + 2.0 * q);
}

Eigen::Vector3d learnSPH::kernel::Wendland_gradW(const Eigen::Vector3d& x)
{
    const double r = x.norm();
    if (r < 1e-15) return Eigen::Vector3d::Zero();
 
    const double q = r / h;
    if (q >= 2.0) return Eigen::Vector3d::Zero();
 
    const double s = 1.0 - 0.5 * q;
    const double dWdq = -5.0 * q * (s * s * s);
    return (sigma * dWdq / (r * h)) * x;
}
 

