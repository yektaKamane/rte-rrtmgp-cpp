#ifndef RAYTRACER_H
#define RAYTRACER_H

#include <memory>
#include "Types.h"
#include <curand_kernel.h>
#include "raytracer_kernels.h"
#include "Optical_props.h"
// Forward declarations.
template<typename, int> class Array_gpu;
template<typename TF> class Optical_props_gpu;
template<typename> class Optical_props_arry_gpu;

#ifdef __CUDACC__
template<typename TF>
class Raytracer_gpu
{
    public:
        Raytracer_gpu();

        void trace_rays(
                const Int photons_to_shoot,
                const int n_col_x, const int n_col_y, const int n_lay,
                const TF dx_grid, const TF dy_grid, const TF dz_grid,
                const Optical_props_2str_gpu<TF>& optical_props,
                const Optical_props_2str_gpu<TF>& cloud_optical_props,
                const TF surface_albedo,
                const TF zenith_angle,
                const TF azimuth_angle,
                const TF flux_toa_dir,
                const TF flux_toa_dif,
                Array_gpu<TF,2>& flux_toa_up,
                Array_gpu<TF,2>& flux_sfc_dir,
                Array_gpu<TF,2>& flux_sfc_dif,
                Array_gpu<TF,2>& flux_sfc_up,
                Array_gpu<TF,3>& flux_abs_dir,
                Array_gpu<TF,3>& flux_abs_dif);

    private:
        curandDirectionVectors32_t* qrng_vectors_gpu;
        unsigned int* qrng_constants_gpu;

};
#endif

#endif
