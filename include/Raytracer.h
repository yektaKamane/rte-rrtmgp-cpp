#ifndef RAYTRACING_H
#define RAYTRACING_H

#include <memory>
#include "Types.h"
#include <curand_kernel.h>

// Forward declarations.
//template<typename, int> class Array;
template<typename, int> class Array_gpu;
//template<typename> class Optical_props_arry;
//template<typename> class Optical_props_arry_gpu;
//template<typename> class Fluxes_broadband;

#ifdef __CUDACC__
template<typename TF>
class Raytracer_gpu
{
    public:
        Raytracer_gpu();
//        void raytracer(
//                const std::unique_ptr<Optical_props_arry_gpu<TF>>& optical_props,
//                const BOOL_TYPE top_at_1,
//                const Array_gpu<TF,1>& mu0,
//                const Array_gpu<TF,1>& inc_flux_dir,
//                const Array_gpu<TF,2>& sfc_alb_dir,
//                const Array_gpu<TF,2>& sfc_alb_dif,
//                const Array_gpu<TF,1>& inc_flux_dif,
//                Array_gpu<TF,2>& gpt_flux_up,
//                Array_gpu<TF,2>& gpt_flux_dn,
//                Array_gpu<TF,2>& gpt_flux_dir);

    private:
        curandDirectionVectors32_t* qrng_vectors_gpu;
        unsigned int* qrng_constants_gpu;

};
#endif

#endif
