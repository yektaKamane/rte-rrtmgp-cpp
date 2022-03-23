#include "raytracer_kernels_bw.h"
#include "Raytracer.h"
#include <iomanip>
#include <iostream>
#include <vector>
#include <chrono>
#include <limits>
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

void run_ray_tracer(const Int n_photons)
{
    // Workload per thread
    const Int photons_per_thread = n_photons / (grid_size * block_size);
    std::cout << "Shooting " << n_photons << " photons (" << photons_per_thread << " per thread) " << std::endl;

    //// DEFINE INPUT ////
    // Grid properties.
    const Float dx_grid = 10000.;
    const Float dy_grid = 10000.;
    const Float dz_grid = 500.; // 30.;

    const int itot = 4;//240;
    const int jtot = 4;//240;
    const int ktot = 142;//284;

    const int icam=1024;
    const int jcam=1024;

    const Float x_size = itot*dx_grid;
    const Float y_size = jtot*dy_grid;
    const Float z_size = ktot*dz_grid;

    // Radiation properties.
    const Float zenith_angle = 50.*(M_PI/180.); // 50
    const Float azimuth_angle = 200.*(M_PI/180.); // 200
    const Float dir_x = -std::sin(zenith_angle) * std::cos(azimuth_angle);
    const Float dir_y = -std::sin(zenith_angle) * std::sin(azimuth_angle);
    const Float dir_z = -std::cos(zenith_angle);
    const Float mu = std::abs(std::cos(zenith_angle));
    // Create the spatial fields.
    std::vector<Optics_ext> k_ext(itot*jtot*ktot);
    std::vector<Optics_scat> ssa_asy(itot*jtot*ktot);

    auto load_binary = [](const std::string& name, void* ptr, const int size)
    {
        std::ifstream binary_file(name + ".bin", std::ios::in | std::ios::binary);

        if (binary_file)
            binary_file.read(reinterpret_cast<char*>(ptr), size*sizeof(Int));
        else
        {
            std::string error = "Cannot read file \"" + name + ".bin\"";
            throw std::runtime_error(error);
        }
    };

    std::vector<Float> k_ext_gas_tmp(itot*jtot*ktot);
    std::vector<Float> k_ext_cloud_tmp(itot*jtot*ktot);
    std::vector<Float> ssa_tmp(itot*jtot*ktot);
    std::vector<Float> asy_tmp(itot*jtot*ktot);
    std::vector<Float> surface_albedo_tmp(1);

    load_binary("k_ext_gas", k_ext_gas_tmp.data(), itot*jtot*ktot);
    load_binary("k_ext_cloud", k_ext_cloud_tmp.data(), itot*jtot*ktot);
    load_binary("ssa", ssa_tmp.data(), itot*jtot*ktot);
    load_binary("asy", asy_tmp.data(), itot*jtot*ktot);
    
    load_binary("albedo", surface_albedo_tmp.data(), 1);

    // Process the input data.
    Float k_ext_null = 0;
    Float k_ext_null_gas = 0;
    for (int k=0; k<ktot; ++k)
        for (int j=0; j<jtot; ++j)
            for (int i=0; i<itot; ++i)
            {
                const int ijk = i + j*itot + k*itot*jtot;

                k_ext[ijk].gas = k_ext_gas_tmp[ijk];
                k_ext[ijk].cloud = k_ext_cloud_tmp[ijk];
                ssa_asy[ijk].ssa = ssa_tmp[ijk];
                ssa_asy[ijk].asy = asy_tmp[ijk];

                if (k_ext_null < k_ext_gas_tmp[ijk] + k_ext_cloud_tmp[ijk])
                    k_ext_null = k_ext_gas_tmp[ijk] + k_ext_cloud_tmp[ijk];
                if (k_ext_null_gas < k_ext_gas_tmp[ijk])
                    k_ext_null_gas = k_ext_gas_tmp[ijk];
            }

    const Float fi = (Float)itot/ngrid_h;
    const Float fj = (Float)jtot/ngrid_h;
    const Float fk = (Float)ktot/ngrid_v;

    std::vector<Grid_knull> k_null_grid(ngrid_h*ngrid_h*ngrid_v);//max(k_null_gas_min, k_ext_null_gas));
    for (int k=0; k<ngrid_v; ++k)
        for (int j=0; j<ngrid_h; ++j)
            for (int i=0; i<ngrid_h; ++i)
            {
                const int i0 = i*fi;
                const Float i1_tmp = (i+1)*fi;
                const int i1 = min(itot,(int)(std::floor(i1_tmp) > i1_tmp ? std::floor(i1_tmp) : std::floor(i1_tmp)+1));
                
                const int j0 = j*fj;
                const Float j1_tmp = (j+1)*fj;
                const int j1 = min(jtot,(int)(std::floor(j1_tmp) > j1_tmp ? std::floor(j1_tmp) : std::floor(j1_tmp)+1));
                
                const int k0 = k*fk;
                const Float k1_tmp = (k+1)*fk;
                const int k1 = min(ktot,(int)(std::floor(k1_tmp) > k1_tmp ? std::floor(k1_tmp) : std::floor(k1_tmp)+1));

                const int ijk_grid = i + j*ngrid_h + k*ngrid_h*ngrid_h; 
                Float k_abs_min = std::numeric_limits<Float>::max();
                Float k_ext_min = std::numeric_limits<Float>::max();
                Float k_ext_max = Float(0.); 
                for (int kk=k0; kk<k1; ++kk)
                    for (int jj=j0; jj<j1; ++jj)
                        for (int ii=i0; ii<i1; ++ii)
                        {
                            const int ijk_orig = ii + jj*itot + kk*itot*jtot; 
                            const Float k_ext_tot = k_ext[ijk_orig].gas + k_ext[ijk_orig].cloud;
                            k_abs_min = min(k_abs_min, k_ext_tot * (1-ssa_tmp[ijk_orig]));
                            k_ext_min = min(k_ext_min, k_ext_tot);
                            k_ext_max = max(k_ext_max, k_ext_tot);
                        }
                 if (k_ext_min == k_ext_max) k_ext_min = k_ext_max * Float(0.9);
                 k_null_grid[ijk_grid].k_min = k_ext_min;
                 k_null_grid[ijk_grid].k_max = k_ext_max;
            }

    //// PREPARE OUTPUT ARRAYS ////
    std::vector<Float> camera_count(icam*jcam);
    std::vector<Float> camera_shot(icam*jcam);
    std::vector<int> counter(2);

    //// COPY THE DATA TO THE GPU.
    // kn grid
    Grid_knull* k_null_grid_gpu = allocate_gpu<Grid_knull>(ngrid_h*ngrid_h*ngrid_v);
    copy_to_gpu(k_null_grid_gpu, k_null_grid.data(), ngrid_h*ngrid_h*ngrid_v);
    
    // Input array.
    Optics_ext* k_ext_gpu = allocate_gpu<Optics_ext>(itot*jtot*ktot);
    Optics_scat* ssa_asy_gpu = allocate_gpu<Optics_scat>(itot*jtot*ktot);

    copy_to_gpu(k_ext_gpu, k_ext.data(), itot*jtot*ktot);
    copy_to_gpu(ssa_asy_gpu, ssa_asy.data(), itot*jtot*ktot);

    // Background array;
    Optics_ext* k_ext_bg_gpu = allocate_gpu<Optics_ext>(1);
    Optics_scat* ssa_asy_bg_gpu = allocate_gpu<Optics_scat>(1);
    Float* z_lev_bg_gpu = allocate_gpu<Float>(2);
    const int kbg=1;
    //copy_to_gpu(k_ext_gpu, k_ext.data(), itot*jtot*ktot);
    //copy_to_gpu(ssa_asy_gpu, ssa_asy.data(), itot*jtot*ktot);
    //copy_to_gpu(z_lev_gpu, z_lev_bg.data(), itot*jtot*ktot);
    
    
    // Output arrays. Copy them in order to enable restarts later.
    Float* camera_count_gpu = allocate_gpu<Float>(icam*jcam);
    Float* camera_shot_gpu = allocate_gpu<Float>(icam*jcam);
    int* counter_gpu = allocate_gpu<int>(2);

    copy_to_gpu(camera_count_gpu, camera_count.data(), icam*jcam);
    copy_to_gpu(camera_shot_gpu, camera_shot.data(), icam*jcam);

    // Cloud dimensions
    std::vector<Int> cloud_mask_v(ktot);
    std::vector<Float> cloud_dims(2);
    
    Int* cloud_mask_v_gpu = allocate_gpu<Int>(ktot);
    Float* cloud_dims_gpu = allocate_gpu<Float>(2);

    copy_to_gpu(cloud_mask_v_gpu, cloud_mask_v.data(), ktot);
    copy_to_gpu(cloud_dims_gpu, cloud_dims.data(), 2);

    // albedo
    Float* surface_albedo_gpu = allocate_gpu<Float>(1);
    copy_to_gpu(surface_albedo_gpu, surface_albedo_tmp.data(), 1);
    
    const int block_size_m = ktot;
    dim3 grid_m{1}, block_m{block_size_m};
    //auto start_m = std::chrono::high_resolution_clock::now();

    copy_from_gpu(cloud_dims.data(), cloud_dims_gpu, 2);
    // std::cout<<cloud_dims[0]<<" - "<<cloud_dims[1]<<std::endl;

    //// RUN THE RAY TRACER ////
    curandDirectionVectors32_t* qrng_vectors;
    curandGetDirectionVectors32(
                &qrng_vectors,
                CURAND_SCRAMBLED_DIRECTION_VECTORS_32_JOEKUO6);
    unsigned int* qrng_constants;
    curandGetScrambleConstants32(&qrng_constants);

    curandDirectionVectors32_t* qrng_vectors_gpu = allocate_gpu<curandDirectionVectors32_t>(2);
    copy_to_gpu(qrng_vectors_gpu, qrng_vectors, 2);
    unsigned int* qrng_constants_gpu = allocate_gpu<unsigned int>(2);
    copy_to_gpu(qrng_constants_gpu, qrng_constants, 2);

    dim3 grid{grid_size}, block{block_size};

    auto start = std::chrono::high_resolution_clock::now();
    ray_tracer_kernel_bw<<<grid, block>>>(
            photons_per_thread, k_null_grid_gpu, 
            camera_count_gpu,
            camera_shot_gpu,
            counter_gpu,
            icam, jcam,
            k_ext_gpu, ssa_asy_gpu,
            k_ext_bg_gpu, ssa_asy_bg_gpu,
            z_lev_bg_gpu,
            surface_albedo_gpu,
            mu,
            x_size, y_size, z_size,
            dx_grid, dy_grid, dz_grid,
            dir_x, dir_y, dir_z,
            itot, jtot, ktot, kbg,
            qrng_vectors_gpu, qrng_constants_gpu);

    cuda_safe_call(cudaDeviceSynchronize());

    auto end = std::chrono::high_resolution_clock::now();
    double duration = std::chrono::duration_cast<std::chrono::duration<double>>(end - start).count();
    std::cout << "Duration: " << std::setprecision(5) << duration << " (s)" << std::endl;
    //// END RUNNING OF RAY TRACER ////


    //// COPY OUTPUT BACK TO CPU ////
    copy_from_gpu(camera_count.data(), camera_count_gpu, icam*jcam);
    copy_from_gpu(camera_shot.data(), camera_shot_gpu, icam*jcam);
    copy_from_gpu(counter.data(), counter_gpu, 2);
    printf("%d \n",counter[0]);
    //for (int i=0; i<icam*jcam; ++i) printf("%f \n",camera_count[i]);
    
    //// SAVE THE OUTPUT TO DISK ////
    auto save_binary = [](const std::string& name, void* ptr, const int size)
    {
        std::ofstream binary_file(name + ".bin", std::ios::out | std::ios::trunc | std::ios::binary);

        if (binary_file)
            binary_file.write(reinterpret_cast<const char*>(ptr), size*sizeof(Float));
        else
        {
            std::string error = "Cannot write file \"" + name + ".bin\"";
            throw std::runtime_error(error);
        }
    };

    save_binary("camera", camera_count.data(), icam*jcam);
    save_binary("shot", camera_shot.data(), icam*jcam);
    save_binary("knull", k_null_grid.data(), ngrid_h*ngrid_h*ngrid_v);
}


int main(int argc, char* argv[])
{
    if (argc != 2)
    {
        std::cout << "The number of photons is must be a power of two (2**n), please add the exponent n" << std::endl;
        return 1;
    }

    const Int n_photons = std::pow(Int(2), static_cast<Int>(std::stoi(argv[1])));

    if (n_photons < grid_size * block_size)
    {
        std::cerr << "Sorry, the number of photons must be larger than " << grid_size * block_size
            << " (n >= " << std::log2(grid_size*block_size) << ") to guarantee one photon per thread" << std::endl;
        return 1;
    }

    run_ray_tracer(n_photons);

    return 0;
}
