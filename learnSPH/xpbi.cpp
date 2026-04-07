#include "xpbi.h"
#include "io.h"
#include "kernel.h"
#include "constitutive.h"
#include "plasticity.h"
#include <iostream>
#include <filesystem>
#include <iomanip>
#include <cmath>

namespace pbd
{
    void XPBISolver::initialize(
        const std::vector<Eigen::Vector3d>& positions,
        const std::vector<double>& mass_inverses,
        const std::vector<Eigen::Vector3d>& velocities,
        double particle_radius,
        int sim_iterations,
        int solver_iterations,
        double dt,
        double fps,
        double particle_volume,
        double mu,
        double lambda,
        PlasticityModel plasticity_model,
        const std::vector<double>& plasticity_params,
        const std::string& output_dir,
        bool use_cfl,
        double smoothing_length_ratio,
        double position_correction_coeff,
        double position_correction_alpha)
    {
        // Particle state
        m_positions      = positions;
        m_velocities     = velocities;
        m_mass_inverses  = mass_inverses;
        const int n = static_cast<int>(positions.size());

        // Material
        m_mu     = mu;
        m_lambda = lambda;
        m_particle_volume_0 = particle_volume;
        m_particle_volumes.assign(n, particle_volume);
 
        // Simulation parameters
        m_particle_radius = particle_radius;
        m_smoothing_length = smoothing_length_ratio * particle_radius;
        m_compact_support  = 2.0 * m_smoothing_length;
        m_position_correction_dist = position_correction_coeff;
        m_max_dt = dt;
        m_dt     = dt;
        m_sim_iterations    = sim_iterations;
        m_solver_iterations = solver_iterations;
        m_fps        = fps;
        m_frame_time = 1.0 / fps;
        m_use_cfl    = use_cfl;

        // Compliance
        m_alpha_tilde_elastic   = 1.0 / (m_particle_volume_0 * dt * dt);
        if (position_correction_alpha < particle_volume)
            m_alpha_tilde_collision = 0.0;  // hard constraint
        else
            m_alpha_tilde_collision = 1.0 / (position_correction_alpha * dt * dt);
    
        // Kernel
        learnSPH::kernel::setSmoothingLength(m_smoothing_length);

        // Neighborhood search
        m_nsearch.set_radius(m_compact_support);
        m_nsearch.add_point_set(m_positions.front().data(), n, true, true);
    
        // Deformation gradient
        m_F.assign(n, Eigen::Matrix3d::Identity());
        m_F_trial.assign(n, Eigen::Matrix3d::Identity());

        // SPH correction
        m_L.assign(n, Eigen::Matrix3d::Identity());
        m_corrected_gradW.resize(n);
    
        // Constraint lambdas
        m_lambda_elastic.assign(n, 0.0);
        m_lambda_ground.assign(n, 0.0);
        m_lambda_position_correction.resize(n);
        m_lambda_pc_offset.resize(n, 0);

        // Plasticity
        m_plasticity_model = plasticity_model;
        m_is_yielded.assign(n, false);
    
        if (plasticity_model == PlasticityModel::VonMises)
        {
            double yield_stress = (plasticity_params.size() > 0) ? plasticity_params[0] : 0.0;
            m_vm_ksai = (plasticity_params.size() > 1) ? plasticity_params[1] : 0.0;
            m_vm_q.assign(n, yield_stress);
            m_constitutive_config = {m_mu, m_lambda, 
                                        m_plasticity_model == PlasticityModel::VonMises,
                                        0.0};
        }
        else if (plasticity_model == PlasticityModel::DruckerPrager)
        {
            if (plasticity_params.size() >= 2)
            {
                double friction_angle = plasticity_params[0];
                m_dp_cohesion = plasticity_params[1];
                m_dp_alpha = std::sqrt(2.0 / 3.0) *
                            (2.0 * std::sin(friction_angle) /
                            (3.0 - std::sin(friction_angle)));
                m_constitutive_config = {m_mu, m_lambda, 
                                            m_plasticity_model == PlasticityModel::DruckerPrager,
                                            m_dp_cohesion};
            }
        }

        // Output directory
        namespace fs = std::filesystem;
        fs::path outdir(output_dir);
        if (!fs::exists(outdir))
            fs::create_directories(outdir);
        m_output_dir = outdir.string();
    
        // Log
        std::cout << "XPBI initialized: " << n << " particles, "
                << "h=" << m_smoothing_length << ", "
                << "dt=" << m_dt << std::endl;
    }

    // ════════════════════════════════════════════════════════════════
    // Main simulation loop
    // ════════════════════════════════════════════════════════════════
    void XPBISolver::run()
    {
        for (int i = 0; i < m_sim_iterations; ++i)
        {
            step();
            m_ct += m_dt;
    
            // Write frame output
            if (m_ct >= m_ct_frame)
            {
                learnSPH::write_particles_to_vtk(
                    m_output_dir + "frame_" + std::to_string(m_frame) + ".vtk",
                    m_positions);
    
                m_ct_frame += m_frame_time;
                m_frame++;
            }
            if (i % 100 == 0)
            std::cout << "step " << i << "/" << m_sim_iterations 
                    << ", t=" << m_ct << std::endl;
        }
    }

    // ════════════════════════════════════════════════════════════════
    // Single timestep: t^n → t^{n+1}
    // ════════════════════════════════════════════════════════════════
    
    void XPBISolver::step()
    {
        // CFL adaptive timestep
        if (m_use_cfl)
            computeCFL();
    
        // Neighbor search at x^n
        m_nsearch.find_neighbors();

        buildColoredBuckets();
    
        // Kernel gradient correction: L_p
        computeKernelGradientCorrection();
    
        // Predict velocities: v ← v + Δt M⁻¹ f_ext
        predictVelocities();
    
        // Clear lambdas for this timestep
        clearLambdas();
 
        // Gauss-Seidel iterations
        for (int iter = 0; iter < m_solver_iterations; ++iter)
        {
            solveElasticConstraints();
            solvePositionCorrection();
            if (m_has_triangle_boundary)
                solveTriangleContact();
        }
    
        // XSPH velocity smoothing
        smoothVelocities();
 
        // Update F^{n+1} with return mapping
        updateDeformationGradient();
    
        // Advect: x^{n+1} = x^n + Δt v^{n+1}
        advect();
    }

    // ════════════════════════════════════════════════════════════════
    // Pipeline steps
    // ════════════════════════════════════════════════════════════════
    void XPBISolver::buildColoredBuckets()
    {
        // assign particles to colored cells for parallel GS
        const int n = static_cast<int>(m_positions.size());
        const double grid_size = 1.1 * m_compact_support;

        // Bounding box
        Eigen::Vector3d min_pos = m_positions[0];
        Eigen::Vector3d max_pos = m_positions[0];
        for (auto const& pos : m_positions)
        {
            min_pos = min_pos.cwiseMin(pos);
            max_pos = max_pos.cwiseMax(pos);
        }

        Eigen::Vector3i cell_range = ((max_pos - min_pos) / grid_size).array().ceil().cast<int>();
        cell_range = cell_range.array() + 1;
        int cx = cell_range[0];
        int cy = cell_range[1];
        int cz = cell_range[2];
        int total_cells = cx * cy * cz;

        m_colored_cells.assign(total_cells, {});
        m_colored_buckets.assign(N_COLORS, {});
        std::vector<char> cell_used(total_cells, 0);

        for (int i = 0; i < n; ++i)
        {
            Eigen::Vector3d rel = m_positions[i] - min_pos;
            int gx = static_cast<int>(rel[0] / grid_size);
            int gy = static_cast<int>(rel[1] / grid_size);
            int gz = static_cast<int>(rel[2] / grid_size);

            int color_id = (gx % 3) * 9 + (gy % 3) * 3 + (gz % 3);
            int cell_id  = gx * cy * cz + gy * cz + gz;

            m_colored_cells[cell_id].push_back(i);
            if (!cell_used[cell_id])
            {
                cell_used[cell_id] = 1;
                m_colored_buckets[color_id].push_back(cell_id);
            }
        }
    }


    void XPBISolver::computeKernelGradientCorrection()
    {
        auto const& d = m_nsearch.point_set(0);
        const int n = static_cast<int>(m_positions.size());

        // Update current volumes: V = V_0 * det(F)
        #pragma omp parallel for
        for (int i = 0; i < n; ++i)
            m_particle_volumes[i] = m_particle_volume_0 * m_F[i].determinant();

        // Compute L matrix and corrected gradients (Eq. 10)
        #pragma omp parallel for
        for (int i = 0; i < n; ++i)
        {
            const int n_neighbors = d.n_neighbors(0, i);

            // Accumulate L = Σ_j V_j ∇W_ij ⊗ (-r_ij)
            Eigen::Matrix3d L_raw = Eigen::Matrix3d::Zero();
            std::vector<Eigen::Vector3d> Wendland_gradWs(n_neighbors);
            for (int j = 0; j < n_neighbors; ++j)
            {
                int n_id = d.neighbor(0, i, j);
                Eigen::Vector3d r_ij = m_positions[i] - m_positions[n_id];
                Wendland_gradWs[j] = learnSPH::kernel::Wendland_gradW(r_ij);
                L_raw += m_particle_volumes[n_id] * (Wendland_gradWs[j] * (-r_ij).transpose());
            }

            // Pseudo-inverse via SVD with clamped singular values
            Eigen::JacobiSVD<Eigen::Matrix3d> svd(L_raw, Eigen::ComputeFullU | Eigen::ComputeFullV);
            Eigen::Vector3d S = svd.singularValues();
            Eigen::Matrix3d S_inv = Eigen::Matrix3d::Zero();
            for (int k = 0; k < 3; ++k)
                S_inv(k, k) = (S[k] > 1e-2) ? (1.0 / S[k]) : 1.0;
            m_L[i] = svd.matrixV() * S_inv * svd.matrixU().transpose();

            m_corrected_gradW[i].resize(n_neighbors);
            for (int j = 0; j < n_neighbors; ++j)
            {
                int n_id = d.neighbor(0, i, j);
                m_corrected_gradW[i][j] = m_particle_volumes[n_id] * m_L[i] * Wendland_gradWs[j];
            }
        }
    }
    
    void XPBISolver::predictVelocities()
    {
        //v_i += dt * m_inv_i * f_ext (gravity)
        const int n = static_cast<int>(m_positions.size());
        #pragma omp parallel for
        for (int i = 0; i < n; ++i)
        {
            if (m_mass_inverses[i] == 0.0) continue; // kinematic particles handling
            m_velocities[i] += m_dt * m_gravity;
        }
    }

    void XPBISolver::clearLambdas()
    {
        auto const& d = m_nsearch.point_set(0);
        const int n = d.n_points();

        std::fill(m_lambda_elastic.begin(), m_lambda_elastic.end(), 0.0);
        std::fill(m_lambda_ground.begin(), m_lambda_ground.end(), 0.0);
        m_is_yielded.assign(n, false);

        // Position correction lambdas: one per neighbor pair
        m_lambda_position_correction.resize(n);
        for (int i = 0; i < n; ++i)
            m_lambda_position_correction[i].assign(d.n_neighbors(0, i), 0.0);

        // Triangle lambdas
        if (m_has_triangle_boundary)
        {
            m_tri_lambdas.resize(n);
            for (int i = 0; i < n; ++i)
                m_tri_lambdas[i].assign(m_tri_faces.size(), 0.0);
        }
    }
 
    void XPBISolver::solveElasticConstraints()
    {
        auto const& d = m_nsearch.point_set(0);
        // colored GS loop
        for (int c = 0; c < N_COLORS; ++c)
        {
            auto const& bucket = m_colored_buckets[c];

            #pragma omp parallel for
            for (int cell_id = 0; cell_id < static_cast<int>(bucket.size()); ++cell_id)
            {
                // sequentially solve particles in the same cell to avoid data race
                auto const& cell_particles = m_colored_cells[bucket[cell_id]];

                //   for each particle:
                for (int p = 0; p < static_cast<int>(cell_particles.size()); ++p)
                {
                    int i = cell_particles[p];
                    if (m_mass_inverses[i] == 0.0) continue; // kinematic particles handling
                    //     1. compute velocity gradient, F_trial = (I + dt * ∇v) * F^n
                    Eigen::Matrix3d vel_grad = Eigen::Matrix3d::Zero();
                    Eigen::Vector3d v_i = m_velocities[i];
                    for (int j = 0; j < d.n_neighbors(0, i); ++j)
                    {
                        int n_id = d.neighbor(0, i, j);
                        Eigen::Vector3d v_ji = m_velocities[n_id] - v_i;
                        vel_grad += v_ji * m_corrected_gradW[i][j].transpose();
                    }
                    //     2. SVD of F_trial
                    m_F_trial[i] = (Eigen::Matrix3d::Identity() + m_dt * vel_grad) * m_F[i];
                    Eigen::JacobiSVD<Eigen::Matrix3d> svd(m_F_trial[i], Eigen::ComputeFullU | Eigen::ComputeFullV);
                    Eigen::Matrix3d U = svd.matrixU();
                    Eigen::Vector3d S = svd.singularValues();
                    Eigen::Matrix3d V = svd.matrixV();

                    //     3. plasticity return mapping (if enabled)
                    if (m_plasticity_model == PlasticityModel::VonMises)
                    {
                        auto pr = learnSPH::plasticity::vonMisesReturnMapping(
                            S, m_mu, m_vm_ksai, m_vm_q[i]);
                        // if plasticity exists
                        if (pr.deltaGamma > 0.0)
                        {
                            S = pr.S;
                            // convert back to F_E
                            m_F_trial[i] = U * S.asDiagonal() * V.transpose();
                            m_is_yielded[i] = true;
                        }
                    }
                    else if (m_plasticity_model == PlasticityModel::DruckerPrager)
                    {
                        auto pr = learnSPH::plasticity::druckerPragerReturnMapping(
                            S, m_mu, m_lambda, m_dp_alpha, m_dp_cohesion);
                        // if plasticity exists
                        if (pr.deltaGamma > 0.0)
                        {
                            S = pr.S;
                            // convert back to F_E
                            m_F_trial[i] = U * S.asDiagonal() * V.transpose();
                            m_is_yielded[i] = true;
                        }
                    }
                    //     4. constitutive: evaluate C and dC/dF
                    auto cr = learnSPH::constitutive::evaluateStVKHencky(U, S, V, m_constitutive_config);
                    if (cr.C <= 0.0) continue;

                    //     5. compute Δλ, update velocities
                    double numerator = -cr.C - m_alpha_tilde_elastic * m_lambda_elastic[i];
                    double denominator = m_alpha_tilde_elastic;

                    // ∇_{x_b} C = dC/dF · F^T · corrected_gradW  (Eq. 12)
                    Eigen::Vector3d gradC_i = Eigen::Vector3d::Zero();
                    std::vector<Eigen::Vector3d> grad_neighbors(d.n_neighbors(0, i), Eigen::Vector3d::Zero());
                    
                    for (int j = 0; j < d.n_neighbors(0, i); ++j)
                    {
                        int n_id = d.neighbor(0, i, j);
                        Eigen::Vector3d g = cr.dCdF * m_F[i].transpose() * m_corrected_gradW[i][j];
                        grad_neighbors[j] = g;
                        gradC_i -= g;                                         // Eq. 13
                        denominator += m_mass_inverses[n_id] * g.squaredNorm();
                    }
                    denominator += m_mass_inverses[i] * gradC_i.squaredNorm();

                    double delta_lambda = numerator / denominator;
                    m_lambda_elastic[i] += delta_lambda;                     // Eq. 19

                    // ── 6. Velocity update (Eq. 18) ──
                    m_velocities[i] += m_mass_inverses[i] * gradC_i * delta_lambda / m_dt;

                    for (int j = 0; j < d.n_neighbors(0, i); ++j)
                    {
                        int n_id = d.neighbor(0, i, j);
                        m_velocities[n_id] += m_mass_inverses[n_id] * grad_neighbors[j] * delta_lambda / m_dt;
                    }
                }
             }
        }
    }
    
    void XPBISolver::solvePositionCorrection()
    {
        // prevent particle penetration (ground + inter-particle)
        auto const& d = m_nsearch.point_set(0);
        const int n = static_cast<int>(m_positions.size());
        const Eigen::Vector3d ground_normal{0.0, 0.0, 1.0};

        // ── Part 1: Inter-particle position correction (colored GS) ──
        for (int c = 0; c < N_COLORS; ++c)
        {
            auto const& bucket = m_colored_buckets[c];

            #pragma omp parallel for
            for (int cell_id = 0; cell_id < static_cast<int>(bucket.size()); ++cell_id)
            {
                auto const& cell_particles = m_colored_cells[bucket[cell_id]];

                for (int p = 0; p < static_cast<int>(cell_particles.size()); ++p)
                {
                    int i = cell_particles[p];

                    for (int j = 0; j < d.n_neighbors(0, i); ++j)
                    {
                        int n_id = d.neighbor(0, i, j);
                        if (n_id < i) continue; // avoid double counting 
                        if (m_mass_inverses[i] == 0.0 && m_mass_inverses[n_id] == 0.0) continue;

                        // Predict positions
                        Eigen::Vector3d x_i = m_positions[i];
                        Eigen::Vector3d x_j = m_positions[n_id];
                        if (m_mass_inverses[i]    != 0.0) x_i += m_velocities[i]    * m_dt;
                        if (m_mass_inverses[n_id] != 0.0) x_j += m_velocities[n_id] * m_dt;

                        Eigen::Vector3d r_ij = x_i - x_j;
                        double dist = r_ij.norm();

                        Eigen::Vector3d grad;
                        if (dist > 1e-12)
                            grad = r_ij / dist;
                        else {
                            grad = Eigen::Vector3d(1.0, 0.0, 0.0);
                            dist = 1e-12;
                        }

                        double constraint = dist - m_position_correction_dist;
                        if (constraint >= 0.0) continue;

                        double denom = m_mass_inverses[i] + m_mass_inverses[n_id] + m_alpha_tilde_collision;
                        double delta_lambda = (-constraint - m_alpha_tilde_collision * m_lambda_position_correction[i][j]) / denom;
                        m_lambda_position_correction[i][j] += delta_lambda;

                        if (m_mass_inverses[i] != 0.0)
                            m_velocities[i] += m_mass_inverses[i] * grad * delta_lambda / m_dt;
                        if (m_mass_inverses[n_id] != 0.0)
                            m_velocities[n_id] -= m_mass_inverses[n_id] * grad * delta_lambda / m_dt;
                    }
                }
            }
        }

        // ── Part 2: Ground contact with Coulomb friction ──
        #pragma omp parallel for
        for (int i = 0; i < n; ++i)
        {
            if (m_mass_inverses[i] == 0.0) continue;

            Eigen::Vector3d x_pred = m_positions[i] + m_velocities[i] * m_dt;
            if (x_pred[2] >= 0.0) continue;

            // Hard constraint: push above ground
            double delta_lambda = -x_pred[2] / m_mass_inverses[i];
            m_lambda_ground[i] += delta_lambda;

            Eigen::Vector3d v_before = m_velocities[i];
            m_velocities[i] += m_mass_inverses[i] * ground_normal * delta_lambda / m_dt;
            Eigen::Vector3d v_after = m_velocities[i];

            // Coulomb friction
            Eigen::Vector3d impulse = (v_after - v_before) / m_mass_inverses[i];
            Eigen::Vector3d vt = v_after - v_after.dot(ground_normal) * ground_normal;

            if (vt.norm() > 1e-12 && impulse.dot(ground_normal) > 0.0)
            {
                double max_friction = m_ground_friction * impulse.norm() * m_mass_inverses[i];
                if (vt.norm() <= max_friction)
                    m_velocities[i] -= vt;  // static friction
                else
                    m_velocities[i] -= max_friction * vt.normalized();  // dynamic friction
            }
        }
    }
    
    void XPBISolver::smoothVelocities()
    {
        // XSPH smoothing
        auto const& d = m_nsearch.point_set(0);
        const int n = d.n_points();
        constexpr double xsph_coeff = 0.01;

        // Accumulate smoothing bias
        std::vector<Eigen::Vector3d> bias(n, Eigen::Vector3d::Zero());

        #pragma omp parallel for
        for (int i = 0; i < n; ++i)
        {
            if (m_mass_inverses[i] == 0.0) continue;

            for (int j = 0; j < d.n_neighbors(0, i); ++j)
            {
                int n_id = d.neighbor(0, i, j);
                Eigen::Vector3d r_ij = m_positions[i] - m_positions[n_id];
                Eigen::Vector3d v_ji = m_velocities[n_id] - m_velocities[i];
                bias[i] += m_particle_volumes[n_id] * v_ji * learnSPH::kernel::Wendland_W(r_ij);
            }
        }

        // Apply
        #pragma omp parallel for
        for (int i = 0; i < n; ++i)
        {
            if (m_mass_inverses[i] == 0.0) continue;
            m_velocities[i] += xsph_coeff * bias[i];
        }
    }
 
    void XPBISolver::updateDeformationGradient()
    {
        // F^{n+1} = Z((I + dt ∇v^{n+1}) F^n)
        //   also update m_vm_q if yielded
        auto const& d = m_nsearch.point_set(0);
        const int n = d.n_points();

        #pragma omp parallel for
        for (int i = 0; i < n; ++i)
        {
            // Velocity gradient with post-XSPH velocities
            Eigen::Matrix3d vel_grad = Eigen::Matrix3d::Zero();
            for (int j = 0; j < d.n_neighbors(0, i); ++j)
            {
                int n_id = d.neighbor(0, i, j);
                Eigen::Vector3d v_ji = m_velocities[n_id] - m_velocities[i];
                vel_grad += v_ji * m_corrected_gradW[i][j].transpose();
            }

            // F^{n+1} = (I + Δt ∇v^{n+1}) F^n   (Eq. 22)
            Eigen::Matrix3d F_new = (Eigen::Matrix3d::Identity() + vel_grad * m_dt) * m_F[i];

            // Return mapping on the stored F (Eq. 22: Z applied)
            if (m_plasticity_model == PlasticityModel::VonMises)
            {
                Eigen::JacobiSVD<Eigen::Matrix3d> svd(F_new, Eigen::ComputeFullU | Eigen::ComputeFullV);
                auto pr = learnSPH::plasticity::vonMisesReturnMapping(
                    svd.singularValues(), m_mu, m_vm_ksai, m_vm_q[i]);
                if (pr.deltaGamma > 0.0)
                {
                    F_new = svd.matrixU() * pr.S.asDiagonal() * svd.matrixV().transpose();
                    m_vm_q[i] += 2.0 * m_mu * m_vm_ksai * pr.deltaGamma; // linear hardening updates yield stress
                }
            }
            else if (m_plasticity_model == PlasticityModel::DruckerPrager)
            {
                Eigen::JacobiSVD<Eigen::Matrix3d> svd(F_new, Eigen::ComputeFullU | Eigen::ComputeFullV);
                auto pr = learnSPH::plasticity::druckerPragerReturnMapping(
                    svd.singularValues(), m_mu, m_lambda, m_dp_alpha, m_dp_cohesion);
                if (pr.deltaGamma > 0.0)
                    F_new = svd.matrixU() * pr.S.asDiagonal() * svd.matrixV().transpose();
            }

            m_F[i] = F_new;
        }
    }
    
    void XPBISolver::advect()
    {
        // x^{n+1} = x^n + dt * v^{n+1}
        const int n = static_cast<int>(m_positions.size());

        #pragma omp parallel for
        for (int i = 0; i < n; ++i)
        {
            if (m_mass_inverses[i] == 0.0) continue;
            m_positions[i] += m_velocities[i] * m_dt;
        }
    }
    
    void XPBISolver::computeCFL()
    {
        // TODO: adaptive dt based on max velocity
        double v_max = 1e-6;
        for (auto const& v : m_velocities)
            v_max = std::max(v_max, v.norm());

        double dt_new = std::min(m_max_dt, 0.4 * 2.0 * m_particle_radius / v_max);

        // Recompute compliance with new dt
        double ratio_sq = (m_dt * m_dt) / (dt_new * dt_new);
        m_alpha_tilde_elastic   *= ratio_sq;
        m_alpha_tilde_collision *= ratio_sq;

        m_dt = dt_new;
    }

 
    void XPBISolver::solveTriangleContact()
    {
        // TODO: triangle mesh boundary collision
    }
 
    void XPBISolver::addTriangleBoundaryMesh(
        const std::vector<Eigen::Vector3d>& vertices,
        const std::vector<std::array<int, 3>>& triangles)
    {
        m_tri_vertices = vertices;
        m_tri_faces    = triangles;
        m_has_triangle_boundary = true;
    }
 
    Eigen::Vector3d XPBISolver::closestPointOnTriangle(
        const Eigen::Vector3d& p,
        const Eigen::Vector3d& a,
        const Eigen::Vector3d& b,
        const Eigen::Vector3d& c,
        bool& is_outside)
    {
        // TODO: copy from original
        is_outside = true;
        return p;
    }
}
