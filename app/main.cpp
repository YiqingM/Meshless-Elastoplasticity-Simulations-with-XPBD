#include "../learnSPH/xpbi.h"
#include "../learnSPH/io.h"
#include <vector>
#include <Eigen/Dense>
#include <iostream>

int main()
{
    // Box of sand: 1m x 1m x 1m
    double particle_radius = 0.005;
    double sampling_dist = 2.0 * particle_radius;
    double density = 1.0;

    int ln = static_cast<int>(1.0 / sampling_dist);
    int wn = static_cast<int>(1.0 / sampling_dist);
    int hn = static_cast<int>(1.0 / sampling_dist);

    std::vector<Eigen::Vector3d> positions;
    std::vector<Eigen::Vector3d> velocities;

    for (int i = 0; i < ln; ++i)
        for (int j = 0; j < wn; ++j)
            for (int k = 0; k < hn; ++k)
            {
                positions.push_back({i * sampling_dist, j * sampling_dist, k * sampling_dist + 2.0 * particle_radius});
                velocities.push_back(Eigen::Vector3d::Zero());
            }

    double particle_volume = 1.0 * 1.0 * 1.0 / positions.size();
    std::vector<double> mass_inverses(positions.size(), 1.0 / (density * particle_volume));

    // Material
    double E = 400.0;
    double nu = 0.4;
    double lambda = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu));
    double mu = E / (2.0 * (1.0 + nu));

    // Drucker-Prager
    double friction_deg = 30.0;
    double friction_rad = friction_deg * M_PI / 180.0;
    double cohesion = 0.00;

    double smoothing_ratio = 2.4;
    double pc_coeff = smoothing_ratio * particle_radius - 0.75 * smoothing_ratio * particle_radius;
    double pc_alpha = 0.0; // hard constraint

    pbd::XPBISolver solver;
    solver.initialize(
        positions, mass_inverses, velocities,
        particle_radius,
        24000,      // sim iterations
        5,         // solver iterations
        0.0001,     // dt
        60.0,       // fps
        particle_volume, mu, lambda,
        pbd::PlasticityModel::DruckerPrager,
        {friction_rad, cohesion},
        "../res/dp_sand_largescale/",
        true,       // CFL
        smoothing_ratio, pc_coeff, pc_alpha
    );

    solver.run();
    return 0;
}