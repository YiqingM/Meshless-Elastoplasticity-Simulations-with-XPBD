#include "../learnSPH/xpbi.h"
#include <vector>
#include <string>
#include <cmath>
#include <Eigen/Dense>

int main(int argc, char** argv)
{
    std::string out = argc > 1 ? argv[1] : "../res/sand_collapse/";
    int steps       = argc > 2 ? std::atoi(argv[2]) : 15000;

    const double r  = 0.005;
    const double dx = 2.0 * r;
    const double W = 0.15, H = 0.30;    // column, aspect ratio 2
    const double density = 1.0;

    std::vector<Eigen::Vector3d> pos, vel;
    int nw = int(std::round(W / dx)), nh = int(std::round(H / dx));
    for (int i = 0; i < nw; ++i)
        for (int j = 0; j < nw; ++j)
            for (int k = 0; k < nh; ++k)
            {
                pos.push_back({(i + 0.5) * dx - W / 2, (j + 0.5) * dx - W / 2, (k + 0.5) * dx});
                vel.push_back(Eigen::Vector3d::Zero());
            }
    double V = dx * dx * dx;
    std::vector<double> minv(pos.size(), 1.0 / (density * V));

    double E = 400.0, nu = 0.4;
    double lambda = E * nu / ((1 + nu) * (1 - 2 * nu));
    double mu = E / (2 * (1 + nu));
    double phi = 30.0 * M_PI / 180.0;

    pbd::XPBISolver solver;
    solver.initialize(pos, minv, vel, r,
                      steps, 5, 1e-4, 60.0, V, mu, lambda,
                      pbd::PlasticityModel::DruckerPrager, {phi, 0.0},
                      out, true,
                      2.4,    // h = 2.4 r
                      0.75,   // d_min = 0.75 h
                      0.0);   // hard position constraint
    solver.run();
    return 0;
}