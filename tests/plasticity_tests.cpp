#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include <Eigen/Dense>
#include <cmath>

#include "../learnSPH/plasticity.h"
#include "../learnSPH/constitutive.h"

using Catch::Approx;
using learnSPH::plasticity::druckerPragerReturnMapping;
using learnSPH::constitutive::evaluateStVKHencky;
using learnSPH::constitutive::Config;

namespace
{
    const double E  = 400.0;
    const double nu = 0.4;
    const double lambda = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu));
    const double mu     = E / (2.0 * (1.0 + nu));
    const double phi    = 30.0 * M_PI / 180.0;
    const double alpha  = std::sqrt(2.0 / 3.0) * (2.0 * std::sin(phi) / (3.0 - std::sin(phi)));

    Eigen::Vector3d hencky(const Eigen::Vector3d& S) { return S.array().log(); }
    Eigen::Vector3d dev(const Eigen::Vector3d& e)    { return e - e.mean() * Eigen::Vector3d::Ones(); }

    // DP yield function in Hencky strain space (thesis Eq. 3.82):
    //   f = |dev eps| + alpha (3 lambda + 2 mu)/(2 mu) (tr eps - 3c);  f<0 elastic, f=0 on cone
    double yieldFn(const Eigen::Vector3d& e, double c)
    {
        return dev(e).norm() + alpha * (3.0 * lambda + 2.0 * mu) / (2.0 * mu) * (e.sum() - 3.0 * c);
    }
}

TEST_CASE("DP: elastic trial state is returned unchanged", "[dp]")
{
    Eigen::Vector3d S(0.98, 0.985, 0.99);
    REQUIRE(yieldFn(hencky(S), 0.0) < 0.0);

    auto pr = druckerPragerReturnMapping(S, mu, lambda, alpha, 0.0);
    CHECK(pr.deltaGamma == 0.0);
    CHECK((pr.S - S).norm() == Approx(0.0).margin(1e-14));
}

TEST_CASE("DP: compressive trial outside the cone is projected exactly onto the cone", "[dp]")
{
    Eigen::Vector3d S(1.06, 1.0, 0.93);
    Eigen::Vector3d e = hencky(S);
    REQUIRE(e.sum() < 0.0);
    REQUIRE(yieldFn(e, 0.0) > 0.0);

    auto pr = druckerPragerReturnMapping(S, mu, lambda, alpha, 0.0);
    Eigen::Vector3d e2 = hencky(pr.S);

    CHECK(pr.deltaGamma > 0.0);
    CHECK(yieldFn(e2, 0.0) == Approx(0.0).margin(1e-10));              // on the yield surface
    CHECK(e2.sum() == Approx(e.sum()).margin(1e-12));                   // volume preserved
    CHECK(dev(e2).normalized().dot(dev(e).normalized()) == Approx(1.0)); // deviatoric return

    Eigen::Matrix3d I = Eigen::Matrix3d::Identity();
    Config cfg{mu, lambda, true, 0.0};
    auto cr = evaluateStVKHencky(I, pr.S, I, cfg);
    CHECK(cr.C > 0.0);                                                   // sand resists compression
}

TEST_CASE("DP: tensile trial (tr eps > 0) keeps its volume and yields ZERO constraint", "[dp]")
{
    Eigen::Vector3d S(1.10, 1.04, 0.97);
    Eigen::Vector3d e = hencky(S);
    REQUIRE(e.sum() > 0.0);

    auto pr = druckerPragerReturnMapping(S, mu, lambda, alpha, 0.0);
    Eigen::Vector3d e2 = hencky(pr.S);

    CHECK(pr.deltaGamma == Approx(dev(e).norm()));
    CHECK(dev(e2).norm() == Approx(0.0).margin(1e-12));   // no deviatoric part left
    CHECK(e2.sum() == Approx(e.sum()).margin(1e-12));     // ECI volume-preserving (Eq. 3.80)

    Eigen::Matrix3d I = Eigen::Matrix3d::Identity();

    // correct pairing: extrapolated StVK -> psi = mu|dev eps|^2 = 0 -> C = 0 (free separation)
    Config granular{mu, lambda, true, 0.0};
    CHECK(evaluateStVKHencky(I, pr.S, I, granular).C == 0.0);

    // wrong pairing: plain StVK keeps 0.5 lambda tr^2 + mu|eps|^2 -> C > 0 -> cohesive "sand"
    Config plain{mu, lambda, false, 0.0};
    CHECK(evaluateStVKHencky(I, pr.S, I, plain).C > 0.1);
}

TEST_CASE("DP: cohesion shifts the cone tip to tr eps = 3c", "[dp]")
{
    const double c = 0.01;
    Eigen::Vector3d S(1.05, 1.00, 0.98);
    Eigen::Vector3d e = hencky(S);
    REQUIRE(e.sum() < 3.0 * c);
    REQUIRE(yieldFn(e, c) > 0.0);

    auto pr = druckerPragerReturnMapping(S, mu, lambda, alpha, c);
    Eigen::Vector3d e2 = hencky(pr.S);
    CHECK(yieldFn(e2, c) == Approx(0.0).margin(1e-10));
    CHECK(e2.sum() == Approx(e.sum()).margin(1e-12));

    Eigen::Vector3d S_t(1.03, 1.02, 1.01);          // beyond the shifted tip
    auto pr_t = druckerPragerReturnMapping(S_t, mu, lambda, alpha, c);
    CHECK(dev(hencky(pr_t.S)).norm() == Approx(0.0).margin(1e-12));
    CHECK(hencky(pr_t.S).sum() == Approx(hencky(S_t).sum()).margin(1e-12));
}
