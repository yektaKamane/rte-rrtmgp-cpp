#ifndef RAYTRACER_H
#define RAYTRACER_H

#include <memory>
#include "Types.h"
#include <curand_kernel.h>
#include "raytracer_kernels_bw.h"
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
                const int n_col_x, const int n_col_y, const int n_z, const int n_lay,
                const TF dx_grid, const TF dy_grid, const TF dz_grid,
                const Array_gpu<TF,1>& z_lev,
                const Optical_props_2str_gpu<TF>& optical_props,
                const Optical_props_2str_gpu<TF>& cloud_optical_props,
                const Array_gpu<TF,2>& surface_albedo,
                const TF zenith_angle,
                const TF azimuth_angle,
                const Array_gpu<TF,1>& toa_src,
                const TF toa_factor,
                const TF rayleigh,
                const Array_gpu<TF,2>& col_dry,
                const Array_gpu<TF,2>& vmr_h2o,
                Array_gpu<TF,2>& flux_camera);

        void add_xyz_camera(
                const int cam_nx, const int cam_ny,
                const Array_gpu<TF,1>& xyz_factor,
                const Array_gpu<TF,2>& flux_camera,
                Array_gpu<TF,3>& XYZ);
        
        void normalize_xyz_camera(
                const int cam_nx, const int cam_ny,
                const TF total_source,
                Array_gpu<TF,3>& XYZ);
                
    private:
        curandDirectionVectors32_t* qrng_vectors_gpu;
        unsigned int* qrng_constants_gpu;

};
#endif

#endif
