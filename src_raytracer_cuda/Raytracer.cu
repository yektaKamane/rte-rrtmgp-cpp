#include "Raytracer.h"
#include "Array.h"
#include "raytracer_kernels.h"

namespace
{
    inline void gpu_assert(cudaError_t code, const char *file, int line, bool abort=true)
    {
        if (code != cudaSuccess)
        {
            fprintf(stderr,"CUDA_SAFE_CALL: %s %s %d\n", cudaGetErrorString(code), file, line);
            if (abort) exit(code);
        }
    }
    
    
    template<typename T>
    T* allocate_gpu(const int length)
    {
        T* data_ptr = Tools_gpu::allocate_gpu<T>(length);
    
        return data_ptr;
    }
    template<typename T>
    void copy_to_gpu(T* gpu_data, const T* cpu_data, const int length)
    {
        cuda_safe_call(cudaMemcpy(gpu_data, cpu_data, length*sizeof(T), cudaMemcpyHostToDevice));
    }
    
    
    template<typename T>
    void copy_from_gpu(T* cpu_data, const T* gpu_data, const int length)
    {
        cuda_safe_call(cudaMemcpy(cpu_data, gpu_data, length*sizeof(T), cudaMemcpyDeviceToHost));
    }
}

template<typename TF>
Raytracer_gpu<TF>::Raytracer_gpu()
{
    curandDirectionVectors32_t* qrng_vectors;
    curandGetDirectionVectors32(
                &qrng_vectors,
                CURAND_SCRAMBLED_DIRECTION_VECTORS_32_JOEKUO6);
    unsigned int* qrng_constants;
    curandGetScrambleConstants32(&qrng_constants);

    this->qrng_vectors_gpu = allocate_gpu<curandDirectionVectors32_t>(2);
    this->qrng_constants_gpu = allocate_gpu<unsigned int>(2);
    
    copy_to_gpu(qrng_vectors_gpu, qrng_vectors, 2);
    copy_to_gpu(qrng_constants_gpu, qrng_constants, 2);

}

//template<typename TF>
//void Rte_sw_gpu<TF>::rte_sw(
//        const std::unique_ptr<Optical_props_arry_gpu<TF>>& optical_props,
//        const BOOL_TYPE top_at_1,
//        const Array_gpu<TF,1>& mu0,
//        const Array_gpu<TF,1>& inc_flux_dir,
//        const Array_gpu<TF,2>& sfc_alb_dir,
//        const Array_gpu<TF,2>& sfc_alb_dif,
//        const Array_gpu<TF,1>& inc_flux_dif,
//        Array_gpu<TF,2>& gpt_flux_up,
//        Array_gpu<TF,2>& gpt_flux_dn,
//        Array_gpu<TF,2>& gpt_flux_dir)
//{
//    const int ncol = optical_props->get_ncol();
//    const int nlay = optical_props->get_nlay();
//    const int ngpt = optical_props->get_ngpt();
//
// //   expand_and_transpose(optical_props, sfc_alb_dir, sfc_alb_dir_gpt);
// //   expand_and_transpose(optical_props, sfc_alb_dif, sfc_alb_dif_gpt);
//
//    // Upper boundary condition. At this stage, flux_dn contains the diffuse radiation only.
//    rte_kernel_launcher_cuda::apply_BC(ncol, nlay, ngpt, top_at_1, inc_flux_dir, mu0, gpt_flux_dir);
//    if (inc_flux_dif.size() == 0)
//        rte_kernel_launcher_cuda::apply_BC(ncol, nlay, ngpt, top_at_1, gpt_flux_dn);
//    else
//        rte_kernel_launcher_cuda::apply_BC(ncol, nlay, ngpt, top_at_1, inc_flux_dif, gpt_flux_dn);
//
//    // Run the radiative transfer solver
//    // CvH: only two-stream solutions, I skipped the sw_solver_noscat.
//    rte_kernel_launcher_cuda::sw_solver_2stream(
//            ncol, nlay, ngpt, top_at_1,
//            optical_props->get_tau(),
//            optical_props->get_ssa(),
//            optical_props->get_g  (),
//            mu0,
//            sfc_alb_dir, sfc_alb_dif,
//            gpt_flux_up, gpt_flux_dn, gpt_flux_dir,
//            sw_solver_2stream_map);
//
//    // CvH: The original fortran code had a call to the reduce here.
//    // fluxes->reduce(gpt_flux_up, gpt_flux_dn, gpt_flux_dir, optical_props, top_at_1);
//}

#ifdef RTE_RRTMGP_SINGLE_PRECISION
template class Raytracer_gpu<float>;
#else
template class Raytracer_gpu<double>;
#endif
