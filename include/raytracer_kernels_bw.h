#ifndef RAYTRACER_KERNELS_H
#define RAYTRACER_KERNELS_H
#include <curand_kernel.h>



#ifdef RTE_RRTMGP_SINGLE_PRECISION
using Float = float;
constexpr int block_size = 512;
constexpr int grid_size = 256;
#else
using Float = double;
constexpr int block_size = 512;
constexpr int grid_size = 256;
#endif
using Int = unsigned long long;
constexpr int ngrid_h = 90;
constexpr int ngrid_v = 71;
constexpr Float k_null_gas_min = Float(1.e-3);


struct Optics_ext
{
    Float gas;
    Float cloud;
};


struct Optics_scat
{
    Float ssa;
    Float asy;
};

__global__
void ray_tracer_kernel_bw(
        const Int photons_to_shoot,
        const Float* __restrict__ k_null_grid,
        Float* __restrict__ camera_count,
        int* __restrict__ counter,
        const int cam_nx, const int cam_ny,
        const Optics_ext* __restrict__ k_ext, const Optics_scat* __restrict__ ssa_asy,
        const Float surface_albedo,
        const Float diffuse_faction,
        const Float x_size, const Float y_size, const Float z_size,
        const Float dx_grid, const Float dy_grid, const Float dz_grid,
        const Float dir_x, const Float dir_y, const Float dir_z,
        const int itot, const int jtot, const int ktot,
        curandDirectionVectors32_t* qrng_vectors, unsigned int* qrng_constants);
#endif
