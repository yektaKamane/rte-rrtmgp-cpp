#ifndef RAYTRACER_KERNEL_H
#define RAYTRACER_KERNEL_H


#ifdef RTE_RRTMGP_SINGLE_PRECISION
using Float = float;
#else
using Float = double;
#endif
using Int = unsigned long long;


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
void ray_tracer_kernel(
        const Int photons_to_shoot,
        const Float* __restrict__ k_null_grid,
        Float* __restrict__ toa_down_count,
        Float* __restrict__ toa_up_count,
        Float* __restrict__ surface_down_direct_count,
        Float* __restrict__ surface_down_diffuse_count,
        Float* __restrict__ surface_up_count,
        Float* __restrict__ atmos_direct_count,
        Float* __restrict__ atmos_diffuse_count,
        const Optics_ext* __restrict__ k_ext, const Optics_scat* __restrict__ ssa_asy,
        const Float k_ext_null_cld, const Float k_ext_null_gas,
        const Float surface_albedo,
        const Float x_size, const Float y_size, const Float z_size,
        const Float dx_grid, const Float dy_grid, const Float dz_grid,
        const Float dir_x, const Float dir_y, const Float dir_z,
        const int itot, const int jtot, const int ktot,
        curandDirectionVectors32_t* qrng_vectors, unsigned int* qrng_constants,
        const Float* __restrict__ cloud_dims);
#endif
