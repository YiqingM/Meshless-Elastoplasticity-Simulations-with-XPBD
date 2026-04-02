#pragma once
#include <cassert>
#include <Eigen/Dense>

namespace learnSPH
{
	namespace kernel
	{
		constexpr double PI = 3.14159265358979323846;
		
		// Smoothing length and precomputed normalization constant.
    	// Must be initialized once via setSmoothingLength() before any W/gradW call.
		extern double h;
    	extern double sigma;

		// Initialize kernel parameters. Call once before simulation.
    	void setSmoothingLength(double smoothing_length);

		double Wendland_W(const Eigen::Vector3d& x);

		Eigen::Vector3d Wendland_gradW(const Eigen::Vector3d& x);
	};
};