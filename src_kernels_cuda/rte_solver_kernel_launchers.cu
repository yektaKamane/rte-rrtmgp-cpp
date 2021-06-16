#include <chrono>

#include "rte_kernel_launcher_cuda.h"
#include "tools_gpu.h"
#include "Array.h"
#include <iomanip>


namespace
{
    #include "rte_solver_kernels.cu"
}


namespace rte_kernel_launcher_cuda
{
    template<typename TF>
    void apply_BC(const int ncol, const int nlay, const int ngpt, const BOOL_TYPE top_at_1,
                  const Array_gpu<TF,2>& inc_flux_dir, const Array_gpu<TF,1>& mu0, Array_gpu<TF,3>& gpt_flux_dir)
    {
        const int block_col = 32;
        const int block_gpt = 32;

        const int grid_col = ncol/block_col + (ncol%block_col > 0);
        const int grid_gpt = ngpt/block_gpt + (ngpt%block_gpt > 0);

        dim3 grid_gpu(grid_col, grid_gpt);
        dim3 block_gpu(block_col, block_gpt);
        apply_BC_kernel<<<grid_gpu, block_gpu>>>(ncol, nlay, ngpt, top_at_1, inc_flux_dir.ptr(), mu0.ptr(), gpt_flux_dir.ptr());

    }


    template<typename TF>
    void apply_BC(const int ncol, const int nlay, const int ngpt, const BOOL_TYPE top_at_1, Array_gpu<TF,3>& gpt_flux_dn)
    {
        const int block_col = 32;
        const int block_gpt = 32;

        const int grid_col = ncol/block_col + (ncol%block_col > 0);
        const int grid_gpt = ngpt/block_gpt + (ngpt%block_gpt > 0);

        dim3 grid_gpu(grid_col, grid_gpt);
        dim3 block_gpu(block_col, block_gpt);
        apply_BC_kernel<<<grid_gpu, block_gpu>>>(ncol, nlay, ngpt, top_at_1, gpt_flux_dn.ptr());
    }


    template<typename TF>
    void apply_BC(const int ncol, const int nlay, const int ngpt, const BOOL_TYPE top_at_1, const Array_gpu<TF,2>& inc_flux_dif, Array_gpu<TF,3>& gpt_flux_dn)
    {
        const int block_col = 32;
        const int block_gpt = 32;

        const int grid_col = ncol/block_col + (ncol%block_col > 0);
        const int grid_gpt = ngpt/block_gpt + (ngpt%block_gpt > 0);

        dim3 grid_gpu(grid_col, grid_gpt);
        dim3 block_gpu(block_col, block_gpt);
        apply_BC_kernel<<<grid_gpu, block_gpu>>>(ncol, nlay, ngpt, top_at_1, inc_flux_dif.ptr(), gpt_flux_dn.ptr());
    }


    template<typename TF>
    void lw_solver_noscat_gaussquad(const int ncol, const int nlay, const int ngpt, const BOOL_TYPE top_at_1, const int nmus,
                                    const Array_gpu<TF,2>& ds, const Array_gpu<TF,2>& weights, const Array_gpu<TF,3>& tau, const Array_gpu<TF,3> lay_source,
                                    const Array_gpu<TF,3>& lev_source_inc, const Array_gpu<TF,3>& lev_source_dec, const Array_gpu<TF,2>& sfc_emis,
                                    const Array_gpu<TF,2>& sfc_src, Array_gpu<TF,3>& flux_up, Array_gpu<TF,3>& flux_dn,
                                    const Array_gpu<TF,2>& sfc_src_jac, Array_gpu<TF,3>& flux_up_jac)
    {
        TF eps = std::numeric_limits<TF>::epsilon();

        const int flx_size = flux_dn.size();
        const int opt_size = tau.size();
        const int sfc_size = sfc_src.size();

        TF* source_sfc = Tools_gpu::allocate_gpu<TF>(sfc_size);
        TF* source_sfc_jac = Tools_gpu::allocate_gpu<TF>(sfc_size);
        TF* sfc_albedo = Tools_gpu::allocate_gpu<TF>(sfc_size);
        TF* tau_loc = Tools_gpu::allocate_gpu<TF>(opt_size);
        TF* trans = Tools_gpu::allocate_gpu<TF>(opt_size);
        TF* source_dn = Tools_gpu::allocate_gpu<TF>(opt_size);
        TF* source_up = Tools_gpu::allocate_gpu<TF>(opt_size);
        TF* radn_dn = Tools_gpu::allocate_gpu<TF>(flx_size);
        TF* radn_up = Tools_gpu::allocate_gpu<TF>(flx_size);
        TF* radn_up_jac = Tools_gpu::allocate_gpu<TF>(flx_size);

        const int block_col2d = 32;
        const int block_gpt2d = 8;

        const int grid_col2d = ncol/block_col2d + (ncol%block_col2d > 0);
        const int grid_gpt2d = ngpt/block_gpt2d + (ngpt%block_gpt2d > 0);

        dim3 grid_gpu2d(grid_col2d, grid_gpt2d);
        dim3 block_gpu2d(block_col2d, block_gpt2d);

        /*
        // Running some permutations of block sizes.
        {
            std::cout << "TUNING lw_solver_noscat_gaussquad_kernel" << std::endl;
            std::vector<std::pair<int, int>> col_gpt_combis;
            std::vector<int> cols{ 1, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512};
            std::vector<int> gpts{ 1, 2, 4, 8, 16, 32, 64, 128};
            for (const int igpt : gpts)
                for (const int icol : cols)
                    col_gpt_combis.emplace_back(icol, igpt);

            // Create tmp arrays to write output to.
            Array_gpu<TF,3> flux_up_tmp{flux_up}, flux_dn_tmp{flux_dn}, flux_up_jac_tmp{flux_up_jac};

            for (const auto& p : col_gpt_combis)
            {
                std::cout << "(" << p.first << ", " << p.second << "): ";

                const int block_col2d = p.first;
                const int block_gpt2d = p.second;

                const int grid_col2d = ncol/block_col2d + (ncol%block_col2d > 0);
                const int grid_gpt2d = ngpt/block_gpt2d + (ngpt%block_gpt2d > 0);

                dim3 grid_gpu2d(grid_col2d, grid_gpt2d);
                dim3 block_gpu2d(block_col2d, block_gpt2d);

                cudaEvent_t start;
                cudaEvent_t stop;
                cudaEventCreate(&start);
                cudaEventCreate(&stop);

                cudaEventRecord(start, 0);
                lw_solver_noscat_gaussquad_kernel<<<grid_gpu2d, block_gpu2d>>>(
                        ncol, nlay, ngpt, eps, top_at_1, nmus, ds.ptr(), weights.ptr(), tau.ptr(), lay_source.ptr(),
                        lev_source_inc.ptr(), lev_source_dec.ptr(), sfc_emis.ptr(), sfc_src.ptr(), radn_up,
                        radn_dn, sfc_src_jac.ptr(), radn_up_jac, tau_loc, trans, source_dn, source_up,
                        source_sfc, sfc_albedo, source_sfc_jac, flux_up_tmp.ptr(), flux_dn_tmp.ptr(), flux_up_jac_tmp.ptr());
                cudaEventRecord(stop, 0);
                cudaEventSynchronize(stop);
                float duration = 0.f;
                cudaEventElapsedTime(&duration, start, stop);

                std::cout << std::setprecision(10) << duration << " (ns), check: " << flux_up_tmp({ncol, nlay+1, ngpt}) << ", ";

                // Check whether kernel has succeeded;
                cudaError err = cudaGetLastError();
                if (err != cudaSuccess)
                    std::cout << cudaGetErrorString(err) << std::endl;
                else
                    std::cout << std::endl;
            }

            std::cout << "STOP TUNING lw_solver_noscat_gaussquad_kernel" << std::endl;
        }
        // End of performance tuning.
        */

        lw_solver_noscat_gaussquad_kernel<<<grid_gpu2d, block_gpu2d>>>(
                ncol, nlay, ngpt, eps, top_at_1, nmus, ds.ptr(), weights.ptr(), tau.ptr(), lay_source.ptr(),
                lev_source_inc.ptr(), lev_source_dec.ptr(), sfc_emis.ptr(), sfc_src.ptr(), radn_up,
                radn_dn, sfc_src_jac.ptr(), radn_up_jac, tau_loc, trans, source_dn, source_up,
                source_sfc, sfc_albedo, source_sfc_jac, flux_up.ptr(), flux_dn.ptr(), flux_up_jac.ptr());

        Tools_gpu::free_gpu(tau_loc);
        Tools_gpu::free_gpu(radn_up);
        Tools_gpu::free_gpu(radn_up_jac);
        Tools_gpu::free_gpu(radn_dn);
        Tools_gpu::free_gpu(trans);
        Tools_gpu::free_gpu(source_dn);
        Tools_gpu::free_gpu(source_up);
        Tools_gpu::free_gpu(source_sfc);
        Tools_gpu::free_gpu(source_sfc_jac);
        Tools_gpu::free_gpu(sfc_albedo);
    }


    template<typename TF>
    void sw_solver_2stream(const int ncol, const int nlay, const int ngpt, const BOOL_TYPE top_at_1,
                           const Array_gpu<TF,3>& tau, const Array_gpu<TF,3>& ssa, const Array_gpu<TF,3>& g,
                           const Array_gpu<TF,1>& mu0, const Array_gpu<TF,2>& sfc_alb_dir, const Array_gpu<TF,2>& sfc_alb_dif,
                           Array_gpu<TF,3>& flux_up, Array_gpu<TF,3>& flux_dn, Array_gpu<TF,3>& flux_dir)
    {
        const int opt_size = tau.size();
        const int alb_size = sfc_alb_dir.size();
        const int flx_size = flux_up.size();

        TF* r_dif = Tools_gpu::allocate_gpu<TF>(opt_size);
        TF* t_dif = Tools_gpu::allocate_gpu<TF>(opt_size);
        TF* r_dir = Tools_gpu::allocate_gpu<TF>(opt_size);
        TF* t_dir = Tools_gpu::allocate_gpu<TF>(opt_size);
        TF* t_noscat = Tools_gpu::allocate_gpu<TF>(opt_size);
        TF* source_up = Tools_gpu::allocate_gpu<TF>(opt_size);
        TF* source_dn = Tools_gpu::allocate_gpu<TF>(opt_size);
        TF* source_sfc = Tools_gpu::allocate_gpu<TF>(alb_size);
        TF* albedo = Tools_gpu::allocate_gpu<TF>(flx_size);
        TF* src = Tools_gpu::allocate_gpu<TF>(flx_size);
        TF* denom = Tools_gpu::allocate_gpu<TF>(opt_size);

        const int block_col3d = 16;
        const int block_lay3d = 16;
        const int block_gpt3d = 1;

        const int grid_col3d = ncol/block_col3d + (ncol%block_col3d > 0);
        const int grid_lay3d = nlay/block_lay3d + (nlay%block_lay3d > 0);
        const int grid_gpt3d = ngpt/block_gpt3d + (ngpt%block_gpt3d > 0);

        dim3 grid_gpu3d(grid_col3d, grid_lay3d, grid_gpt3d);
        dim3 block_gpu3d(block_col3d, block_lay3d, block_gpt3d);

        TF tmin = std::numeric_limits<TF>::epsilon();
        sw_2stream_kernel<<<grid_gpu3d, block_gpu3d>>>(
                ncol, nlay, ngpt, tmin, tau.ptr(), ssa.ptr(), g.ptr(), mu0.ptr(), r_dif, t_dif, r_dir, t_dir, t_noscat);

        const int block_col2d = 32;
        const int block_gpt2d = 4;

        const int grid_col2d = ncol/block_col2d + (ncol%block_col2d > 0);
        const int grid_gpt2d = ngpt/block_gpt2d + (ngpt%block_gpt2d > 0);

        dim3 grid_gpu2d(grid_col2d, grid_gpt2d);
        dim3 block_gpu2d(block_col2d, block_gpt2d);

        sw_source_adding_kernel<<<grid_gpu2d, block_gpu2d>>>(
                ncol, nlay, ngpt, top_at_1, sfc_alb_dir.ptr(), sfc_alb_dif.ptr(), r_dif, t_dif, r_dir, t_dir, t_noscat,
                flux_up.ptr(), flux_dn.ptr(), flux_dir.ptr(), source_up, source_dn, source_sfc, albedo, src, denom);

        Tools_gpu::free_gpu(r_dif);
        Tools_gpu::free_gpu(t_dif);
        Tools_gpu::free_gpu(r_dir);
        Tools_gpu::free_gpu(t_dir);
        Tools_gpu::free_gpu(t_noscat);
        Tools_gpu::free_gpu(source_up);
        Tools_gpu::free_gpu(source_dn);
        Tools_gpu::free_gpu(source_sfc);
        Tools_gpu::free_gpu(albedo);
        Tools_gpu::free_gpu(src);
        Tools_gpu::free_gpu(denom);
    }
}


#ifdef RTE_RRTMGP_SINGLE_PRECISION
template void rte_kernel_launcher_cuda::apply_BC(const int, const int, const int, const BOOL_TYPE,
                  const Array_gpu<float,2>&, const Array_gpu<float,1>&, Array_gpu<float,3>&);
template void rte_kernel_launcher_cuda::apply_BC(const int, const int, const int, const BOOL_TYPE, Array_gpu<float,3>&);
template void rte_kernel_launcher_cuda::apply_BC(const int, const int, const int, const BOOL_TYPE,
                  const Array_gpu<float,2>&, Array_gpu<float,3>&);

template void rte_kernel_launcher_cuda::sw_solver_2stream<float>(
            const int, const int, const int, const BOOL_TYPE,
            const Array_gpu<float,3>&, const Array_gpu<float,3>&, const Array_gpu<float,3>&,
            const Array_gpu<float,1>&, const Array_gpu<float,2>&, const Array_gpu<float,2>&,
            Array_gpu<float,3>&, Array_gpu<float,3>&, Array_gpu<float,3>&);

template void rte_kernel_launcher_cuda::lw_solver_noscat_gaussquad<float>(
            const int ncol, const int nlay, const int ngpt, const BOOL_TYPE top_at_1, const int nmus,
            const Array_gpu<float,2>& ds, const Array_gpu<float,2>& weights, const Array_gpu<float,3>& tau, const Array_gpu<float,3> lay_source,
            const Array_gpu<float,3>& lev_source_inc, const Array_gpu<float,3>& lev_source_dec, const Array_gpu<float,2>& sfc_emis,
            const Array_gpu<float,2>& sfc_src, Array_gpu<float,3>& flux_dn, Array_gpu<float,3>& flux_up,
            const Array_gpu<float,2>& sfc_src_jac, Array_gpu<float,3>& flux_up_jac);
#else
template void rte_kernel_launcher_cuda::apply_BC(const int, const int, const int, const BOOL_TYPE,
                  const Array_gpu<double,2>&, const Array_gpu<double,1>&, Array_gpu<double,3>&);
template void rte_kernel_launcher_cuda::apply_BC(const int, const int, const int, const BOOL_TYPE, Array_gpu<double,3>&);
template void rte_kernel_launcher_cuda::apply_BC(const int, const int, const int, const BOOL_TYPE,
                  const Array_gpu<double,2>&, Array_gpu<double,3>&);

template void rte_kernel_launcher_cuda::sw_solver_2stream<double>(
            const int, const int, const int, const BOOL_TYPE,
            const Array_gpu<double,3>&, const Array_gpu<double,3>&, const Array_gpu<double,3>&,
            const Array_gpu<double,1>&, const Array_gpu<double,2>&, const Array_gpu<double,2>&,
            Array_gpu<double,3>&, Array_gpu<double,3>&, Array_gpu<double,3>&);

template void rte_kernel_launcher_cuda::lw_solver_noscat_gaussquad<double>(
            const int ncol, const int nlay, const int ngpt, const BOOL_TYPE top_at_1, const int nmus,
            const Array_gpu<double,2>& ds, const Array_gpu<double,2>& weights, const Array_gpu<double,3>& tau, const Array_gpu<double,3> lay_source,
            const Array_gpu<double,3>& lev_source_inc, const Array_gpu<double,3>& lev_source_dec, const Array_gpu<double,2>& sfc_emis,
            const Array_gpu<double,2>& sfc_src, Array_gpu<double,3>& flux_up, Array_gpu<double,3>& flux_dn,
            const Array_gpu<double,2>& sfc_src_jac,Array_gpu<double,3>& flux_up_jac);
#endif