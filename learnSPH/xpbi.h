#pragma once
#include <CompactNSearch/CompactNSearch>
#include <Eigen/Dense>
#include <vector>
#include <string>
#include <array>
#include <unordered_set>
#include <unordered_map>
#include <fstream>
#include "constitutive.h"

namespace pbd
{
    using ConstitutiveConfig = learnSPH::constitutive::Config;

    enum class PlasticityModel
    {
        None,
        VonMises,
        DruckerPrager
    };

    class XPBISolver
    {
        public:
            XPBISolver() = default;
            ~XPBISolver() = default;

        void initialize(
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
            PlasticityModel plasticity_model = PlasticityModel::None,
            const std::vector<double>& plasticity_params = {},
            const std::string& output_dir = "./",
            bool use_cfl = false,
            double smoothing_length_ratio = 2.4,
            double position_correction_coeff = 0.25,
            double position_correction_alpha = 1.0
        );

        void run();

        void addTriangleBoundaryMesh(
            const std::vector<Eigen::Vector3d>& vertices,
            const std::vector<std::array<int, 3>>& triangles
        );

        private:
            // ── Pipeline steps ──
            void step();
            void computeKernelGradientCorrection();
            void predictVelocities();
            void solveElasticConstraints();
            void solvePositionCorrection();
            void solveTriangleContact();
            void smoothVelocities();        // XSPH
            void updateDeformationGradient();
            void advect();
            void computeCFL();
            void clearLambdas();

            // ── Colored Gauss-Seidel ──
            void buildColoredBuckets();
            static constexpr int N_COLORS = 27;  // 3^d in 3D
            std::vector<std::vector<int>> m_colored_buckets;
            std::vector<std::vector<int>> m_colored_cells;

            // ── Neighborhood search ──
            CompactNSearch::NeighborhoodSearch m_nsearch{1.0};

            // ── Particle state ──
            std::vector<Eigen::Vector3d> m_positions;
            std::vector<Eigen::Vector3d> m_initial_positions;
            std::vector<Eigen::Vector3d> m_velocities;
            std::vector<double>          m_mass_inverses;
            
            // ── Deformation gradient ──
            std::vector<Eigen::Matrix3d> m_F;       // elastic deformation gradient (state at t^n)
            std::vector<Eigen::Matrix3d> m_F_trial; // trial F within GS iterations (state at t^n + 1)

            // ── SPH correction ──
            std::vector<Eigen::Matrix3d> m_L;       // kernel gradient correction matrix
            std::vector<std::vector<Eigen::Vector3d>> m_corrected_gradW; // L·∇W per particle per neighbor
    
            // ── Constraint lambdas ──
            std::vector<double> m_lambda_elastic;
            std::vector<double> m_lambda_ground;
            std::vector<std::vector<double>> m_lambda_position_correction;
            std::vector<int>    m_lambda_pc_offset; // neighbor offset for position correction

            // ── Material parameters ──
            double m_mu;
            double m_lambda;
            double m_particle_volume_0; // initial particle voluem
            std::vector<double> m_particle_volumes;
            ConstitutiveConfig m_constitutive_config;

            // ── Plasticity ──
            PlasticityModel m_plasticity_model = PlasticityModel::None;
            // Von Mises
            double m_vm_ksai = 0.0;
            std::vector<double> m_vm_q;   // hardening variable per particle
            // Drucker-Prager
            double m_dp_alpha = 0.0;      // precomputed from friction angle
            double m_dp_cohesion = 0.0;
            // Yield tracking
            std::vector<bool> m_is_yielded;
 
            // ── Simulation parameters ──
            double m_particle_radius;
            double m_smoothing_length;
            double m_compact_support;
            double m_dt;
            double m_max_dt;
            int    m_sim_iterations;
            int    m_solver_iterations;
            double m_alpha_tilde_elastic;
            double m_alpha_tilde_collision;
            double m_position_correction_dist;
            bool   m_use_cfl = false;
            Eigen::Vector3d m_gravity{0.0, 0.0, -9.81};
            double m_ground_friction = 0.5;

            // ── Time / output ──
            double m_ct = 0.0;            // current simulation time
            double m_fps;
            double m_frame_time;
            double m_ct_frame = 0.0;      // time accumulated within current frame
            int    m_frame = 0;
            std::string m_output_dir;

            // ── Triangle boundary mesh ──
            bool m_has_triangle_boundary = false;
            std::vector<Eigen::Vector3d>      m_tri_vertices;
            std::vector<std::array<int, 3>>   m_tri_faces;
            std::vector<std::vector<double>>  m_tri_lambdas;

            static Eigen::Vector3d closestPointOnTriangle(
            const Eigen::Vector3d& p,
            const Eigen::Vector3d& a,
            const Eigen::Vector3d& b,
            const Eigen::Vector3d& c,
            bool& is_outside);
    };
}