#include "Raytracer_bw.h"
#include "Array.h"
#include <curand_kernel.h>
#include "rrtmgp_kernel_launcher_cuda_rt.h"
#include "raytracer_kernels_bw.h"
#include "Optical_props_rt.h"
namespace
{
    __global__
    void normalize_xyz_camera_kernel(
            const int cam_nx, const int cam_ny,
            const Float total_source,
            Float* __restrict__ XYZ)
    {
        const int ix = blockIdx.x*blockDim.x + threadIdx.x;
        const int iy = blockIdx.y*blockDim.y + threadIdx.y;
        if ( ( ix < cam_nx) && ( iy < cam_ny) )
        {
            for (int i=0; i<3; ++i)
            {
                const int idx_out = ix + iy*cam_nx + i*cam_nx*cam_ny;
                XYZ[idx_out] /= total_source;
            }
        }
    }

    __global__
    void add_camera_kernel(
            const int cam_nx, const int cam_ny,
            const Float* __restrict__ flux_camera,
            Float* __restrict__ radiance)
    {
        const int ix = blockIdx.x*blockDim.x + threadIdx.x;
        const int iy = blockIdx.y*blockDim.y + threadIdx.y;
        if ( ( ix < cam_nx) && ( iy < cam_ny) )
        {
            const int idx = ix + iy*cam_nx;
            radiance[idx] += flux_camera[idx];
        }
    }

    __global__
    void add_xyz_camera_kernel(
            const int cam_nx, const int cam_ny,
            const Float* __restrict__ xyz_factor,
            const Float* __restrict__ flux_camera,
            Float* __restrict__ XYZ)
    {
        const int ix = blockIdx.x*blockDim.x + threadIdx.x;
        const int iy = blockIdx.y*blockDim.y + threadIdx.y;
        if ( ( ix < cam_nx) && ( iy < cam_ny) )
        {
            const int idx_in = ix + iy*cam_nx;
            for (int i=0; i<3; ++i)
            {
                const int idx_out = ix + iy*cam_nx + i*cam_nx*cam_ny;
                XYZ[idx_out] += xyz_factor[i] * flux_camera[idx_in];
            }
        }
    }

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

    __global__
    void create_knull_grid(
            const int ncol_x, const int ncol_y, const int nz, const Float k_ext_null_min,
            const Float* __restrict__ k_ext, Grid_knull* __restrict__ k_null_grid)
    {
        const int grid_x = blockIdx.x*blockDim.x + threadIdx.x;
        const int grid_y = blockIdx.y*blockDim.y + threadIdx.y;
        const int grid_z = blockIdx.z*blockDim.z + threadIdx.z;
        if ( ( grid_x < ngrid_x) && ( grid_y < ngrid_y) && ( grid_z < ngrid_z))
        {
            const Float fx = Float(ncol_x) / Float(ngrid_x);
            const Float fy = Float(ncol_y) / Float(ngrid_y);
            const Float fz = Float(nz) / Float(ngrid_z);

            const int x0 = grid_x*fx;
            const int x1 = floor((grid_x+1)*fx);
            const int y0 = grid_y*fy;
            const int y1 = floor((grid_y+1)*fy);
            const int z0 = grid_z*fz;
            const int z1 = floor((grid_z+1)*fz);

            const int ijk_grid = grid_x +grid_y*ngrid_x + grid_z*ngrid_y*ngrid_x;
            Float k_null_min = Float(1e15); // just a ridicilously high value
            Float k_null_max = Float(0.);

            for (int k=z0; k<z1; ++k)
                for (int j=y0; j<y1; ++j)
                    for (int i=x0; i<x1; ++i)
                    {
                        const int ijk_in = i + j*ncol_x + k*ncol_x*ncol_y;
                        k_null_min = min(k_null_min, k_ext[ijk_in]);
                        k_null_max = max(k_null_max, k_ext[ijk_in]);
                    }
            if (k_null_min == k_null_max) k_null_min = k_null_max * Float(0.99);
            k_null_grid[ijk_grid].k_min = k_null_min;
            k_null_grid[ijk_grid].k_max = k_null_max;
        }
    }


    __global__
    void bundles_optical_props(
            const int ncol_x, const int ncol_y, const int nz, const int nlay, const Float dz_grid,
            const Float* __restrict__ tau_tot, const Float* __restrict__ ssa_tot,
            const Float* __restrict__ tau_cld, const Float* __restrict__ ssa_cld, const Float* __restrict__ asy_cld,
            const Float* __restrict__ tau_aer, const Float* __restrict__ ssa_aer, const Float* __restrict__ asy_aer,
            const Float rayleigh,
            const Float* __restrict__ col_dry, const Float* __restrict__ vmr_h2o,
            Float* __restrict__ k_ext, Optics_scat* __restrict__ scat_asy)
    {
        const int icol_x = blockIdx.x*blockDim.x + threadIdx.x;
        const int icol_y = blockIdx.y*blockDim.y + threadIdx.y;
        const int iz = blockIdx.z*blockDim.z + threadIdx.z;
        if ( ( icol_x < ncol_x) && ( icol_y < ncol_y) && ( iz < nz))
        {
            const int idx = icol_x + icol_y*ncol_x + iz*ncol_y*ncol_x;
            const Float ksca_gas = rayleigh * (1 + vmr_h2o[idx]) * col_dry[idx] / dz_grid;
            const Float kext_cld = tau_cld[idx] / dz_grid;
            const Float kext_aer = tau_aer[idx] / dz_grid;
            const Float ksca_cld = kext_cld * ssa_cld[idx];
            const Float ksca_aer = kext_aer * ssa_aer[idx];
            const Float kext_tot_old = tau_tot[idx] / dz_grid;
            const Float kext_gas_old = kext_tot_old - kext_cld - kext_aer;
            const Float kabs_gas = kext_gas_old - (kext_tot_old * ssa_tot[idx] - ksca_cld - ksca_aer);
            const Float kext_gas = kabs_gas + ksca_gas;

            k_ext[idx] = tau_tot[idx] / dz_grid;
            scat_asy[idx].k_sca_gas = ksca_gas;
            scat_asy[idx].k_sca_cld = ksca_cld;
            scat_asy[idx].k_sca_aer = ksca_aer;
            scat_asy[idx].asy_cld = asy_cld[idx];
            scat_asy[idx].asy_aer = asy_aer[idx];
        }

    }

    __global__
    void bundles_optical_props_bb(
            const int ncol_x, const int ncol_y, const int nz, const int nlay, const Float dz_grid,
            const Float* __restrict__ tau_tot, const Float* __restrict__ ssa_tot,
            const Float* __restrict__ tau_cld, const Float* __restrict__ ssa_cld, const Float* __restrict__ asy_cld,
            const Float* __restrict__ tau_aer, const Float* __restrict__ ssa_aer, const Float* __restrict__ asy_aer,
            Float* __restrict__ k_ext, Optics_scat* __restrict__ scat_asy)
     {
        const int icol_x = blockIdx.x*blockDim.x + threadIdx.x;
        const int icol_y = blockIdx.y*blockDim.y + threadIdx.y;
        const int iz = blockIdx.z*blockDim.z + threadIdx.z;

        if ( (icol_x < ncol_x) && (icol_y < ncol_y) && (iz < nz) )
        {
            const int idx = icol_x + icol_y*ncol_x + iz*ncol_y*ncol_x;
            const Float kext_cld = tau_cld[idx] / dz_grid;
            const Float kext_aer = tau_aer[idx] / dz_grid;
            const Float kext_tot = tau_tot[idx] / dz_grid;
            const Float ksca_cld = kext_cld * ssa_cld[idx];
            const Float ksca_aer = kext_aer * ssa_aer[idx];

            k_ext[idx] = tau_tot[idx] / dz_grid;
            scat_asy[idx].k_sca_gas = kext_tot*ssa_tot[idx] - ksca_cld - ksca_aer;
            scat_asy[idx].k_sca_cld = ksca_cld;
            scat_asy[idx].k_sca_aer = ksca_aer;
            scat_asy[idx].asy_cld = asy_cld[idx];
            scat_asy[idx].asy_aer = asy_aer[idx];
        }
    }

    __global__
    void background_profile(
            const int ncol_x, const int ncol_y, const int nz, const int nbg,
            const Float* __restrict__ z_lev,
            const Float* __restrict__ tau_tot, const Float* __restrict__ ssa_tot,
            const Float* __restrict__ tau_cld, const Float* __restrict__ ssa_cld, const Float* __restrict__ asy_cld,
            const Float* __restrict__ tau_aer, const Float* __restrict__ ssa_aer, const Float* __restrict__ asy_aer,
            const Float rayleigh,
            const Float* __restrict__ col_dry, const Float* __restrict__ vmr_h2o,
            Float* __restrict__ k_ext_bg, Optics_scat* __restrict__ scat_asy_bg, Float* __restrict__ z_lev_bg)
    {
        const int i = blockIdx.x * blockDim.x + threadIdx.x;
        if ( i < nbg)
        {
            const int idx = (i+nz)*ncol_y*ncol_x;
            const Float dz = abs(z_lev[i+nz+1] - z_lev[i+nz]);

            const Float ksca_gas = rayleigh * (1 + vmr_h2o[idx]) * col_dry[idx] / dz;
            const Float kext_cld = tau_cld[idx] / dz;
            const Float kext_aer = tau_aer[idx] / dz;
            const Float ksca_cld = kext_cld * ssa_cld[idx];
            const Float ksca_aer = kext_aer * ssa_aer[idx];
            const Float kext_tot_old = tau_tot[idx] / dz;
            const Float kext_gas_old = kext_tot_old - kext_cld  - kext_aer;
            const Float kabs_gas = kext_gas_old - (kext_tot_old * ssa_tot[idx] - ksca_cld - ksca_aer);
            const Float kext_gas = kabs_gas + ksca_gas;

            k_ext_bg[i] = tau_tot[idx] / dz;
            scat_asy_bg[i].k_sca_gas = ksca_gas;
            scat_asy_bg[i].k_sca_cld = ksca_cld;
            scat_asy_bg[i].k_sca_aer = ksca_aer;
            scat_asy_bg[i].asy_cld = asy_cld[idx];
            scat_asy_bg[i].asy_aer = asy_aer[idx];

            z_lev_bg[i] = z_lev[i + nz];
            if (i == nbg-1) z_lev_bg[i + 1] = z_lev[i + nz + 1];
        }
    }

    __global__
    void background_profile_bb(
            const int ncol_x, const int ncol_y, const int nz, const int nbg,
            const Float* __restrict__ z_lev,
            const Float* __restrict__ tau_tot, const Float* __restrict__ ssa_tot,
            const Float* __restrict__ tau_cld, const Float* __restrict__ ssa_cld, const Float* __restrict__ asy_cld,
            const Float* __restrict__ tau_aer, const Float* __restrict__ ssa_aer, const Float* __restrict__ asy_aer,
            Float* __restrict__ k_ext_bg, Optics_scat* __restrict__ scat_asy_bg, Float* __restrict__ z_lev_bg)
    {
        const int i = blockIdx.x * blockDim.x + threadIdx.x;
        if ( i < nbg)
        {
            const int idx = (i+nz)*ncol_y*ncol_x;
            const Float dz = abs(z_lev[i+nz+1] - z_lev[i+nz]);

            const Float kext_cld = tau_cld[idx] / dz;
            const Float kext_aer = tau_aer[idx] / dz;
            const Float kext_tot = tau_tot[idx] / dz;

            const Float ksca_cld = kext_cld * ssa_cld[idx];
            const Float ksca_aer = kext_aer * ssa_aer[idx];
            const Float ksca_gas = kext_tot * ssa_tot[idx] - ksca_cld - ksca_aer;

            k_ext_bg[i] = kext_tot;
            scat_asy_bg[i].k_sca_gas = ksca_gas;
            scat_asy_bg[i].k_sca_cld = ksca_cld;
            scat_asy_bg[i].k_sca_aer = ksca_aer;
            scat_asy_bg[i].asy_cld = asy_cld[idx];
            scat_asy_bg[i].asy_aer = asy_aer[idx];

            z_lev_bg[i] = z_lev[i + nz];
            if (i == nbg-1) z_lev_bg[i + 1] = z_lev[i + nz + 1];
        }
    }

    __global__
    void count_to_flux_2d(
            const int cam_nx, const int cam_ny, const Float photons_per_col, const Float toa_src, const Float toa_factor,
            const Float* __restrict__ count, Float* __restrict__ flux)
    {
        const int ix = blockIdx.x*blockDim.x + threadIdx.x;
        const int iy = blockIdx.y*blockDim.y + threadIdx.y;

        if ( ( ix < cam_nx) && ( iy < cam_ny) )
        {
            const int idx = ix + iy*cam_nx;
            const Float flux_per_ray = toa_src * toa_factor / photons_per_col;
            flux[idx] = count[idx] * flux_per_ray;
        }
    }

    __global__
    void add_to_flux_2d(
            const int cam_nx, const int cam_ny, const Float photons_per_col, const Float toa_src, const Float toa_factor,
            const Float* __restrict__ count, Float* __restrict__ flux)
    {
        const int ix = blockIdx.x*blockDim.x + threadIdx.x;
        const int iy = blockIdx.y*blockDim.y + threadIdx.y;

        if ( ( ix < cam_nx) && ( iy < cam_ny) )
        {
            const int idx = ix + iy*cam_nx;
            const Float flux_per_ray = toa_src * toa_factor / photons_per_col;
            flux[idx] += count[idx] * flux_per_ray;
        }
    }


}


Raytracer_bw::Raytracer_bw()
{
}

void Raytracer_bw::add_camera(
    const int cam_nx, const int cam_ny,
    const Array_gpu<Float,2>& flux_camera,
    Array_gpu<Float,2>& radiance)
{
    const int block_x = 8;
    const int block_y = 8;

    const int grid_x  = cam_nx/block_x + (cam_nx%block_x > 0);
    const int grid_y  = cam_ny/block_y + (cam_ny%block_y > 0);

    dim3 grid(grid_x, grid_y);
    dim3 block(block_x, block_y);

    add_camera_kernel<<<grid, block>>>(
            cam_nx, cam_ny,
            flux_camera.ptr(),
            radiance.ptr());
}

void Raytracer_bw::add_xyz_camera(
    const int cam_nx, const int cam_ny,
    const Array_gpu<Float,1>& xyz_factor,
    const Array_gpu<Float,2>& flux_camera,
    Array_gpu<Float,3>& XYZ)
{
    const int block_x = 8;
    const int block_y = 8;

    const int grid_x  = cam_nx/block_x + (cam_nx%block_x > 0);
    const int grid_y  = cam_ny/block_y + (cam_ny%block_y > 0);

    dim3 grid(grid_x, grid_y);
    dim3 block(block_x, block_y);

    add_xyz_camera_kernel<<<grid, block>>>(
            cam_nx, cam_ny,
            xyz_factor.ptr(),
            flux_camera.ptr(),
            XYZ.ptr());
}


void Raytracer_bw::normalize_xyz_camera(
    const int cam_nx, const int cam_ny,
    const Float total_source,
    Array_gpu<Float,3>& XYZ)
{
    const int block_x = 8;
    const int block_y = 8;

    const int grid_x  = cam_nx/block_x + (cam_nx%block_x > 0);
    const int grid_y  = cam_ny/block_y + (cam_ny%block_y > 0);

    dim3 grid(grid_x, grid_y);
    dim3 block(block_x, block_y);

    normalize_xyz_camera_kernel<<<grid, block>>>(
            cam_nx, cam_ny,
            total_source,
            XYZ.ptr());

}


void Raytracer_bw::trace_rays(
        const Int photons_to_shoot,
        const int ncol_x, const int ncol_y, const int nz, const int nlay,
        const Float dx_grid, const Float dy_grid, const Float dz_grid,
        const Array_gpu<Float,1>& z_lev,
        const Array_gpu<Float,2>& tau_total,
        const Array_gpu<Float,2>& ssa_total,
        const Array_gpu<Float,2>& tau_cloud,
        const Array_gpu<Float,2>& ssa_cloud,
        const Array_gpu<Float,2>& asy_cloud,
        const Array_gpu<Float,2>& tau_aeros,
        const Array_gpu<Float,2>& ssa_aeros,
        const Array_gpu<Float,2>& asy_aeros,
        const Array_gpu<Float,2>& surface_albedo,
        const Array_gpu<Float,1>& land_use_map,
        const Float zenith_angle,
        const Float azimuth_angle,
        const Float toa_src,
        const Float toa_factor,
        const Float rayleigh,
        const Array_gpu<Float,2>& col_dry,
        const Array_gpu<Float,2>& vmr_h2o,
        const Array_gpu<Float,1>& cam_data,
        Array_gpu<Float,2>& flux_camera)
{
    const Float mu = std::abs(std::cos(zenith_angle));

    // set of block and grid dimensions used in data processing kernels - requires some proper tuning later
    const int block_col_x = 8;
    const int block_col_y = 8;
    const int block_z = 4;

    const int grid_col_x  = ncol_x/block_col_x + (ncol_x%block_col_x > 0);
    const int grid_col_y  = ncol_y/block_col_y + (ncol_y%block_col_y > 0);
    const int grid_z  = nz/block_z + (nz%block_z > 0);

    dim3 grid_2d(grid_col_x, grid_col_y);
    dim3 block_2d(block_col_x, block_col_y);
    dim3 grid_3d(grid_col_x, grid_col_y, grid_z);
    dim3 block_3d(block_col_x, block_col_y, block_z);

    // bundle optical properties in struct
    Array_gpu<Float,3> k_ext({ncol_x, ncol_y, nz});
    Array_gpu<Optics_scat,3> ssa_asy({ncol_x, ncol_y, nz});

    bundles_optical_props<<<grid_3d, block_3d>>>(
            ncol_x, ncol_y, nz, nlay, dz_grid,
            tau_total.ptr(), ssa_total.ptr(),
            tau_cloud.ptr(), ssa_cloud.ptr(), asy_cloud.ptr(),
            tau_aeros.ptr(), ssa_aeros.ptr(), asy_aeros.ptr(),
            rayleigh, col_dry.ptr(), vmr_h2o.ptr(), k_ext.ptr(), ssa_asy.ptr());

    // create k_null_grid
    const int block_kn_x = 2;
    const int block_kn_y = 2;
    const int block_kn_z = 4;

    const int grid_kn_x  = ngrid_x/block_kn_x + (ngrid_x%block_kn_x > 0);
    const int grid_kn_y  = ngrid_y/block_kn_y + (ngrid_y%block_kn_y > 0);
    const int grid_kn_z  = ngrid_z/block_kn_z + (ngrid_z%block_kn_z > 0);

    dim3 grid_kn(grid_kn_x, grid_kn_y, grid_kn_z);
    dim3 block_kn(block_kn_x, block_kn_y, block_kn_z);

    Array_gpu<Grid_knull,3> k_null_grid({ngrid_x, ngrid_y, ngrid_z});
    const Float k_ext_null_min = Float(1e-3);

    create_knull_grid<<<grid_kn, block_kn>>>(
            ncol_x, ncol_y, nz, k_ext_null_min,
            k_ext.ptr(), k_null_grid.ptr());

    // TOA-TOD profile (at x=0, y=0)
    const int nbg = nlay-nz;
    Array_gpu<Float,1> k_ext_bg({nbg});
    Array_gpu<Optics_scat,1> ssa_asy_bg({nbg});
    Array_gpu<Float,1> z_lev_bg({nbg+1});

    const int block_1d_z = 16;
    const int grid_1d_z  = nbg/block_1d_z + (nbg%block_1d_z > 0);
    dim3 grid_1d(grid_1d_z);
    dim3 block_1d(block_1d_z);

    background_profile<<<grid_1d, block_1d>>>(
            ncol_x, ncol_y, nz, nbg, z_lev.ptr(),
            tau_total.ptr(), ssa_total.ptr(),
            tau_cloud.ptr(), ssa_cloud.ptr(), asy_cloud.ptr(),
            tau_aeros.ptr(), ssa_aeros.ptr(), asy_aeros.ptr(),
            rayleigh, col_dry.ptr(), vmr_h2o.ptr(),
            k_ext_bg.ptr(), ssa_asy_bg.ptr(), z_lev_bg.ptr());

    // initialise output arrays and set to 0
    const int cam_nx = flux_camera.dim(1);
    const int cam_ny = flux_camera.dim(2);

    Array_gpu<Float,2> camera_count({cam_nx, cam_ny});
    Array_gpu<Float,2> shot_count({cam_nx, cam_ny});
    Array_gpu<int,1> counter({1});

    rrtmgp_kernel_launcher_cuda_rt::zero_array(cam_nx, cam_ny, camera_count.ptr());
    rrtmgp_kernel_launcher_cuda_rt::zero_array(cam_nx, cam_ny, shot_count.ptr());
    rrtmgp_kernel_launcher_cuda_rt::zero_array(1, counter.ptr());

    // domain sizes
    const Float x_size = ncol_x * dx_grid;
    const Float y_size = ncol_y * dy_grid;
    const Float z_size = nz * dz_grid;

    // direction of direct rays
    const Float dir_x = -std::sin(zenith_angle) * std::sin(azimuth_angle);
    const Float dir_y = -std::sin(zenith_angle) * std::cos(azimuth_angle);
    const Float dir_z = -std::cos(zenith_angle);

    dim3 grid{grid_size}, block{block_size};
    Int photons_per_thread = photons_to_shoot / (grid_size * block_size);

    ray_tracer_kernel_bw<<<grid, block, nbg*sizeof(Float)>>>(
            photons_per_thread, k_null_grid.ptr(),
            camera_count.ptr(),
            shot_count.ptr(),
            counter.ptr(),
            cam_nx, cam_ny, cam_data.ptr(),
            k_ext.ptr(), ssa_asy.ptr(),
            k_ext_bg.ptr(), ssa_asy_bg.ptr(),
            z_lev_bg.ptr(),
            surface_albedo.ptr(),
            land_use_map.ptr(),
            mu,
            x_size, y_size, z_size,
            dx_grid, dy_grid, dz_grid,
            ngrid_x, ngrid_y, ngrid_z,
            dir_x, dir_y, dir_z,
            ncol_x, ncol_y, nz, nbg);

    //// convert counts to fluxes
    const int block_cam_x = 8;
    const int block_cam_y = 8;

    const int grid_cam_x  = cam_nx/block_cam_x + (cam_nx%block_cam_x > 0);
    const int grid_cam_y  = cam_ny/block_cam_y + (cam_ny%block_cam_y > 0);

    dim3 grid_cam(grid_cam_x, grid_cam_y);
    dim3 block_cam(block_cam_x, block_cam_y);

    const Float photons_per_col = Float(photons_to_shoot) / (cam_nx * cam_ny);
    count_to_flux_2d<<<grid_cam, block_cam>>>(
            cam_nx, cam_ny, photons_per_col,
            toa_src,
            toa_factor,
            camera_count.ptr(),
            flux_camera.ptr());

}

void Raytracer_bw::trace_rays_bb(
        const Int photons_to_shoot,
        const int ncol_x, const int ncol_y, const int nz, const int nlay,
        const Float dx_grid, const Float dy_grid, const Float dz_grid,
        const Array_gpu<Float,1>& z_lev,
        const Array_gpu<Float,2>& tau_total,
        const Array_gpu<Float,2>& ssa_total,
        const Array_gpu<Float,2>& tau_cloud,
        const Array_gpu<Float,2>& ssa_cloud,
        const Array_gpu<Float,2>& asy_cloud,
        const Array_gpu<Float,2>& tau_aeros,
        const Array_gpu<Float,2>& ssa_aeros,
        const Array_gpu<Float,2>& asy_aeros,
        const Array_gpu<Float,2>& surface_albedo,
        const Array_gpu<Float,1>& land_use_map,
        const Float zenith_angle,
        const Float azimuth_angle,
        const Float toa_src,
        const Array_gpu<Float,1>& cam_data,
        Array_gpu<Float,2>& flux_camera)
{
    const Float mu = std::abs(std::cos(zenith_angle));

    // set of block and grid dimensions used in data processing kernels - requires some proper tuning later
    const int block_col_x = 8;
    const int block_col_y = 8;
    const int block_z = 4;

    const int grid_col_x  = ncol_x/block_col_x + (ncol_x%block_col_x > 0);
    const int grid_col_y  = ncol_y/block_col_y + (ncol_y%block_col_y > 0);
    const int grid_z  = nz/block_z + (nz%block_z > 0);

    dim3 grid_2d(grid_col_x, grid_col_y);
    dim3 block_2d(block_col_x, block_col_y);
    dim3 grid_3d(grid_col_x, grid_col_y, grid_z);
    dim3 block_3d(block_col_x, block_col_y, block_z);

    // bundle optical properties in struct
    Array_gpu<Float,3> k_ext({ncol_x, ncol_y, nz});
    Array_gpu<Optics_scat,3> ssa_asy({ncol_x, ncol_y, nz});

    bundles_optical_props_bb<<<grid_3d, block_3d>>>(
            ncol_x, ncol_y, nz, nlay, dz_grid,
            tau_total.ptr(), ssa_total.ptr(),
            tau_cloud.ptr(), ssa_cloud.ptr(), asy_cloud.ptr(),
            tau_aeros.ptr(), ssa_aeros.ptr(), asy_aeros.ptr(),
            k_ext.ptr(), ssa_asy.ptr());

    // create k_null_grid
    const int block_kn_x = 2;
    const int block_kn_y = 2;
    const int block_kn_z = 4;

    const int grid_kn_x  = ngrid_x/block_kn_x + (ngrid_x%block_kn_x > 0);
    const int grid_kn_y  = ngrid_y/block_kn_y + (ngrid_y%block_kn_y > 0);
    const int grid_kn_z  = ngrid_z/block_kn_z + (ngrid_z%block_kn_z > 0);

    dim3 grid_kn(grid_kn_x, grid_kn_y, grid_kn_z);
    dim3 block_kn(block_kn_x, block_kn_y, block_kn_z);

    Array_gpu<Grid_knull,3> k_null_grid({ngrid_x, ngrid_y, ngrid_z});
    const Float k_ext_null_min = Float(1e-3);

    create_knull_grid<<<grid_kn, block_kn>>>(
            ncol_x, ncol_y, nz, k_ext_null_min,
            k_ext.ptr(), k_null_grid.ptr());

    // TOA-TOD profile (at x=0, y=0)
    const int nbg = nlay-nz;
    Array_gpu<Float,1> k_ext_bg({nbg});
    Array_gpu<Optics_scat,1> ssa_asy_bg({nbg});
    Array_gpu<Float,1> z_lev_bg({nbg+1});

    const int block_1d_z = 16;
    const int grid_1d_z  = nbg/block_1d_z + (nbg%block_1d_z > 0);
    dim3 grid_1d(grid_1d_z);
    dim3 block_1d(block_1d_z);

    background_profile_bb<<<grid_1d, block_1d>>>(
            ncol_x, ncol_y, nz, nbg, z_lev.ptr(),
            tau_total.ptr(), ssa_total.ptr(),
            tau_cloud.ptr(), ssa_cloud.ptr(), asy_cloud.ptr(),
            tau_aeros.ptr(), ssa_aeros.ptr(), asy_aeros.ptr(),
            k_ext_bg.ptr(), ssa_asy_bg.ptr(), z_lev_bg.ptr());

    // initialise output arrays and set to 0
    const int cam_nx = flux_camera.dim(1);
    const int cam_ny = flux_camera.dim(2);

    Array_gpu<Float,2> camera_count({cam_nx, cam_ny});
    Array_gpu<Float,2> shot_count({cam_nx, cam_ny});
    Array_gpu<int,1> counter({1});

    rrtmgp_kernel_launcher_cuda_rt::zero_array(cam_nx, cam_ny, camera_count.ptr());
    rrtmgp_kernel_launcher_cuda_rt::zero_array(cam_nx, cam_ny, shot_count.ptr());
    rrtmgp_kernel_launcher_cuda_rt::zero_array(1, counter.ptr());

    // domain sizes
    const Float x_size = ncol_x * dx_grid;
    const Float y_size = ncol_y * dy_grid;
    const Float z_size = nz * dz_grid;

    // direction of direct rays
    const Float dir_x = -std::sin(zenith_angle) * std::sin(azimuth_angle);
    const Float dir_y = -std::sin(zenith_angle) * std::cos(azimuth_angle);
    const Float dir_z = -std::cos(zenith_angle);

    dim3 grid{grid_size}, block{block_size};
    Int photons_per_thread = photons_to_shoot / (grid_size * block_size);

    ray_tracer_kernel_bw<<<grid, block, nbg*sizeof(Float)>>>(
            photons_per_thread, k_null_grid.ptr(),
            camera_count.ptr(),
            shot_count.ptr(),
            counter.ptr(),
            cam_nx, cam_ny, cam_data.ptr(),
            k_ext.ptr(), ssa_asy.ptr(),
            k_ext_bg.ptr(), ssa_asy_bg.ptr(),
            z_lev_bg.ptr(),
            surface_albedo.ptr(),
            land_use_map.ptr(),
            mu,
            x_size, y_size, z_size,
            dx_grid, dy_grid, dz_grid,
            ngrid_x, ngrid_y, ngrid_z,
            dir_x, dir_y, dir_z,
            ncol_x, ncol_y, nz, nbg);

    //// convert counts to fluxes
    const int block_cam_x = 8;
    const int block_cam_y = 8;

    const int grid_cam_x  = cam_nx/block_cam_x + (cam_nx%block_cam_x > 0);
    const int grid_cam_y  = cam_ny/block_cam_y + (cam_ny%block_cam_y > 0);

    dim3 grid_cam(grid_cam_x, grid_cam_y);
    dim3 block_cam(block_cam_x, block_cam_y);

    const Float photons_per_col = Float(photons_to_shoot) / (cam_nx * cam_ny);

    count_to_flux_2d<<<grid_cam, block_cam>>>(
            cam_nx, cam_ny, photons_per_col,
            toa_src,
            Float(1.),
            camera_count.ptr(),
            flux_camera.ptr());

}

