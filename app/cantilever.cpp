// Cantilever beam check (thesis Sec. 4.1 / Fig. 4.3 / Fig. 4.24).
// Usage: learnSPH_cantilever [out_dir] [steps] [dt] [nu] [pc_coeff] [E]
// g = 9.81 is hard-coded in the solver. Beam 0.45 x 0.1 x 0.1 m, rho = 100,
// 3 kinematic layers (m_inv = 0) at x < 0 clamp it, beam starts 0.5 m above ground.
#include "../learnSPH/xpbi.h"
#include <vector>
#include <string>
#include <cmath>
#include <cstdlib>
#include <Eigen/Dense>

int main(int argc, char** argv)
{
    std::string out = argc > 1 ? argv[1] : "../res/cantilever/";
    int    steps    = argc > 2 ? std::atoi(argv[2]) : 60000;
    double dt       = argc > 3 ? std::atof(argv[3]) : 1e-5;
    double nu       = argc > 4 ? std::atof(argv[4]) : 0.3;
    double pc_coeff = argc > 5 ? std::atof(argv[5]) : 0.75;
    double E        = argc > 6 ? std::atof(argv[6]) : 1e4;

    const double r = 0.005, dx = 2.0 * r;
    const double L = 0.45, W = 0.1, H = 0.1;
    const int n_fixed = 3;
    const double density = 100.0;
    const double V = dx * dx * dx;

    std::vector<Eigen::Vector3d> pos, vel;
    std::vector<double> minv;
    int nl = int(std::round(L / dx)), nw = int(std::round(W / dx)), nh = int(std::round(H / dx));
    for (int i = -n_fixed; i < nl; ++i)
        for (int j = 0; j < nw; ++j)
            for (int k = 0; k < nh; ++k)
            {
                pos.push_back({(i + 0.5) * dx, (j + 0.5) * dx, 0.5 + (k + 0.5) * dx});
                vel.push_back(Eigen::Vector3d::Zero());
                minv.push_back(i < 0 ? 0.0 : 1.0 / (density * V));
            }

    double lambda = E * nu / ((1 + nu) * (1 - 2 * nu));
    double mu = E / (2 * (1 + nu));

    pbd::XPBISolver solver;
    solver.initialize(pos, minv, vel, r,
                      steps, 10, dt, 60.0, V, mu, lambda,
                      pbd::PlasticityModel::None, {},
                      out, false,
                      2.4, pc_coeff, 0.0);
    solver.run();
    return 0;
}