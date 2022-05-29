#include <float.h>
#include <curand_kernel.h>
#include "raytracer_kernels.h"
#include <iostream>

namespace
{
    // using Int = unsigned long long;
    const Int Atomic_reduce_const = (Int)(-1LL);
    
    //using Int = unsigned int;
    //const Int Atomic_reduce_const = (Int)(-1);
    
    #ifdef RTE_RRTMGP_SINGLE_PRECISION
    // using Float = float;
    const Float Float_epsilon = FLT_EPSILON;
    // constexpr int block_size = 512;
    // constexpr int grid_size = 64;
    #else
    // using Float = double;
    const Float Float_epsilon = DBL_EPSILON;
    // constexpr int block_size = 512;
    // constexpr int grid_size = 64;
    #endif
    
    constexpr Float w_thres = 0.5;
    
    
    struct Vector
    {
        Float x;
        Float y;
        Float z;
    
    };
    
    
    static inline __device__
    Vector operator*(const Vector v, const Float s) { return Vector{s*v.x, s*v.y, s*v.z}; }
    static inline __device__
    Vector operator*(const Float s, const Vector v) { return Vector{s*v.x, s*v.y, s*v.z}; }
    static inline __device__
    Vector operator-(const Vector v1, const Vector v2) { return Vector{v1.x-v2.x, v1.y-v2.y, v1.z-v2.z}; }
    static inline __device__
    Vector operator+(const Vector v1, const Vector v2) { return Vector{v1.x+v2.x, v1.y+v2.y, v1.z+v2.z}; }
    
    __device__
    Vector cross(const Vector v1, const Vector v2)
    {
        return Vector{
                v1.y*v2.z - v1.z*v2.y,
                v1.z*v2.x - v1.x*v2.z,
                v1.x*v2.y - v1.y*v2.x};
    }
    
    
    __device__
    Float dot(const Vector v1, const Vector v2)
    {
        return v1.x*v2.x + v1.y*v2.y + v1.z*v2.z;
    }
    
    __device__
    Float norm(const Vector v) { return sqrt(v.x*v.x + v.y*v.y + v.z*v.z); }
    
    
    __device__
    Vector normalize(const Vector v)
    {
        const Float length = norm(v);
        return Vector{ v.x/length, v.y/length, v.z/length};
    }
    
    enum class Photon_kind { Direct, Diffuse };
    
    struct Photon
    {
        Vector position;
        Vector direction;
        Photon_kind kind;
    };
    
    
    __device__
    Float pow2(const Float d) { return d*d; }
    
    __device__
    Float rayleigh(const Float random_number)
    {
        const Float q = Float(4.)*random_number - Float(2.);
        const Float d = Float(1.) + pow2(q);
        const Float u = pow(-q + sqrt(d), Float(1./3.));
        return u - Float(1.)/u;
    }
    
    
    __device__
    Float henyey(const Float g, const Float random_number)
    {
        const Float a = pow2(Float(1.) - pow2(g));
        const Float b = Float(2.)*g*pow2(Float(2.)*random_number*g + Float(1.) - g);
        const Float c = -g/Float(2.) - Float(1.)/(Float(2.)*g);
        return Float(-1.)*(a/b) - c;
    }
    
    
    __device__
    Float sample_tau(const Float random_number)
    {
        // Prevent log(0) possibility.
        return Float(-1.)*log(-random_number + Float(1.) + Float_epsilon);
    }
    
    
    __device__
    inline int float_to_int(const Float s_size, const Float ds, const int ntot_max)
    {
        const int ntot = static_cast<int>(s_size / ds);
        return ntot < ntot_max ? ntot : ntot_max-1;
    }
    
    template<typename T>
    struct Random_number_generator
    {
        __device__ Random_number_generator(unsigned int tid)
        {
            curand_init(tid, tid, 0, &state);
        }
    
        __device__ T operator()();
    
        curandState state;
    };
    
    
    template<>
    __device__ double Random_number_generator<double>::operator()()
    {
        return 1. - curand_uniform_double(&state);
    }
    
    
    template<>
    __device__ float Random_number_generator<float>::operator()()
    {
        return 1.f - curand_uniform(&state);
    }


    struct Quasi_random_number_generator_2d
    {
        __device__ Quasi_random_number_generator_2d(
                curandDirectionVectors32_t* vectors, unsigned int* constants, unsigned int offset)
        {
            curand_init(vectors[0], constants[0], offset, &state_x);
            curand_init(vectors[1], constants[1], offset, &state_y);
        }
    
        __device__ void xy(unsigned int* x, unsigned int* y, const int itot, const int jtot)
        {
            *x = curand(&state_x);
            *y = curand(&state_y);

            while (true)
            {
                const int i = *x / static_cast<unsigned int>((1ULL << 32) / 512);
                const int j = *y / static_cast<unsigned int>((1ULL << 32) / 512);

                if (i < itot && j < jtot)
                {
                    return;
                }
                else
                {
                    *x = curand(&state_x);
                    *y = curand(&state_y);
                }
            }
        }
    
        curandStateScrambledSobol32_t state_x;
        curandStateScrambledSobol32_t state_y;
    };
    
    
    __device__
    inline void reset_photon(
            Photon& photon, Int& photons_shot, Float* __restrict__ const toa_down_count,
            Quasi_random_number_generator_2d& qrng,
            Random_number_generator<Float>& rng,
            const Float tod_inc_direct, const Float tod_inc_diffuse,
            const Float x_size, const Float y_size, const Float z_size,
            const Float dx_grid, const Float dy_grid, const Float dz_grid,
            const Float dir_x, const Float dir_y, const Float dir_z,
            const Bool generation_completed, Float& weight,
            const int itot, const int jtot)
    {
        ++photons_shot;
        if (!generation_completed)
        {
            unsigned int random_number_x;
            unsigned int random_number_y;
            qrng.xy(&random_number_x, &random_number_y, itot, jtot);

            const int i = random_number_x / static_cast<unsigned int>((1ULL << 32) / 512);
            const int j = random_number_y / static_cast<unsigned int>((1ULL << 32) / 512);
    
            photon.position.x = x_size * random_number_x / (1ULL << 32);
            photon.position.y = y_size * random_number_y / (1ULL << 32);
            photon.position.z = z_size;
    
            const Float tod_diff_frac = tod_inc_diffuse / (tod_inc_direct + tod_inc_diffuse);
            if (rng() >= tod_diff_frac)
            {
                photon.direction.x = dir_x;
                photon.direction.y = dir_y;
                photon.direction.z = dir_z;
                photon.kind = Photon_kind::Direct;
            }
            else
            {   
                const Float mu_surface = sqrt(rng());
                const Float azimuth_surface = Float(2.*M_PI)*rng();
            
                photon.direction.x = mu_surface*sin(azimuth_surface);
                photon.direction.y = mu_surface*cos(azimuth_surface);
                photon.direction.z = Float(-1) * (sqrt(Float(1.) - mu_surface*mu_surface + Float_epsilon));
                photon.kind = Photon_kind::Diffuse;
            }

            const int ij = i + j*itot;
            if (ij < 0 || ij >=itot*jtot) printf("outofbounds 3");
            atomicAdd(&toa_down_count[ij], Float(1.));
            weight = 1;
    
        }
    }
    
    
    __device__
    inline void write_photon_out(Float* field_out, const Float w)
    {
        atomicAdd(field_out, w);
    }
}


__global__
void ray_tracer_kernel(
        const Int photons_to_shoot,
        const Float* __restrict__ k_null_grid,
        Float* __restrict__ toa_down_count,
        Float* __restrict__ tod_up_count,
        Float* __restrict__ surface_down_direct_count,
        Float* __restrict__ surface_down_diffuse_count,
        Float* __restrict__ surface_up_count,
        Float* __restrict__ atmos_direct_count,
        Float* __restrict__ atmos_diffuse_count,
        const Optics_ext* __restrict__ k_ext, const Optics_scat* __restrict__ ssa_asy,
        const Float tod_inc_direct,
        const Float tod_inc_diffuse,
        const Float* __restrict__ surface_albedo,
        const Float x_size, const Float y_size, const Float z_size,
        const Float dx_grid, const Float dy_grid, const Float dz_grid,
        const Float dir_x, const Float dir_y, const Float dir_z,
        const int itot, const int jtot, const int ktot,
        curandDirectionVectors32_t* qrng_vectors, unsigned int* qrng_constants) // const Float* __restrict__ cloud_dims)
{
    const Float kgrid_h = x_size/Float(ngrid_h);
    const Float kgrid_v = z_size/Float(ngrid_v);
    
    const int n = blockDim.x * blockIdx.x + threadIdx.x;
    
    Photon photon;
    Random_number_generator<Float> rng(n);
    Quasi_random_number_generator_2d qrng(qrng_vectors, qrng_constants, n*photons_to_shoot);

    const Float s_min = max(z_size, max(y_size, x_size)) * Float_epsilon;

    // Set up the initial photons.
    const Bool completed = false;
    Int photons_shot = Atomic_reduce_const;
    Float weight;

    reset_photon(
            photon, photons_shot, toa_down_count,
            qrng, rng,
            tod_inc_direct, tod_inc_diffuse,
            x_size, y_size, z_size,
            dx_grid, dy_grid, dz_grid,
            dir_x, dir_y, dir_z,
            completed, weight,
            itot, jtot);

    Float tau = Float(0.);
    Float d_max = Float(0.);
    Float k_ext_null;
    Bool transition = false;
    int i_n, j_n, k_n;
    
    while (photons_shot < photons_to_shoot)
    {
        const Bool photon_generation_completed = (photons_shot == photons_to_shoot - 1);

        // if d_max is zero, find current grid and maximum distance
        if (d_max == Float(0.))
        {
            i_n = float_to_int(photon.position.x, kgrid_h, ngrid_h);
            j_n = float_to_int(photon.position.y, kgrid_h, ngrid_h);
            k_n = float_to_int(photon.position.z, kgrid_v, ngrid_v);
            const Float sx = abs((photon.direction.x > 0) ? ((i_n+1) * kgrid_h - photon.position.x)/photon.direction.x : (i_n*kgrid_h - photon.position.x)/photon.direction.x);
            const Float sy = abs((photon.direction.y > 0) ? ((j_n+1) * kgrid_h - photon.position.y)/photon.direction.y : (j_n*kgrid_h - photon.position.y)/photon.direction.y);
            const Float sz = abs((photon.direction.z > 0) ? ((k_n+1) * kgrid_v - photon.position.z)/photon.direction.z : (k_n*kgrid_v - photon.position.z)/photon.direction.z);
            d_max = min(sx, min(sy, sz));
            const int ijk_n = i_n + j_n*ngrid_h + k_n*ngrid_h*ngrid_h;
            k_ext_null = k_null_grid[ijk_n];
        }
        
        if (!transition)
        {
            tau = sample_tau(rng());
        }
        transition = false;
        const Float dn = max(Float_epsilon, tau / k_ext_null);
        
        if (dn >= d_max)
        {
            const Float dx = photon.direction.x * (s_min + d_max);
            const Float dy = photon.direction.y * (s_min + d_max);
            const Float dz = photon.direction.z * (s_min + d_max);
            
            photon.position.x += dx;
            photon.position.y += dy;
            photon.position.z += dz;
            
            // surface hit
            if (photon.position.z < Float_epsilon)
            {        
                photon.position.z = Float_epsilon;
                const int i = float_to_int(photon.position.x, dx_grid, itot);
                const int j = float_to_int(photon.position.y, dy_grid, jtot);
                const int ij = i + j*itot;
                d_max = Float(0.);
            if (ij < 0 || ij >=itot*jtot) printf("outofbounds 1");
        
                // Add surface irradiance
                if (photon.kind == Photon_kind::Direct)
                    write_photon_out(&surface_down_direct_count[ij], weight);
                else if (photon.kind == Photon_kind::Diffuse)
                    write_photon_out(&surface_down_diffuse_count[ij], weight);
        
                // Update weights and add upward surface flux
                const Float local_albedo = surface_albedo[0];
                weight *= local_albedo;
                write_photon_out(&surface_up_count[ij], weight);
        
                if (weight < w_thres)
                    weight = (rng() > weight) ? Float(0.) : Float(1.);
        
                // only with nonzero weight continue ray tracing, else start new ray
                if (weight > Float(0.))
                {
                    const Float mu_surface = sqrt(rng());
                    const Float azimuth_surface = Float(2.*M_PI)*rng();
        
                    photon.direction.x = mu_surface*sin(azimuth_surface);
                    photon.direction.y = mu_surface*cos(azimuth_surface);
                    photon.direction.z = sqrt(Float(1.) - mu_surface*mu_surface + Float_epsilon);
                    photon.kind = Photon_kind::Diffuse;
                }
                else
                {
                    reset_photon(
                            photon, photons_shot, toa_down_count,
                            qrng, rng,
                            tod_inc_direct, tod_inc_diffuse,
                            x_size, y_size, z_size,
                            dx_grid, dy_grid, dz_grid,
                            dir_x, dir_y, dir_z,
                            photon_generation_completed, weight,
                            itot, jtot);
                }
            }
        
            // TOD exit
            else if (photon.position.z >= z_size) 
            {
                d_max = Float(0.);
                
                const int i = float_to_int(photon.position.x, dx_grid, itot);
                const int j = float_to_int(photon.position.y, dy_grid, jtot);
                const int ij = i + j*itot;
            if (ij < 0 || ij >=itot*jtot) printf("outofbounds 2");
                write_photon_out(&tod_up_count[ij], weight);
                
                reset_photon(
                        photon, photons_shot, toa_down_count,
                        qrng, rng,
                        tod_inc_direct, tod_inc_diffuse,
                        x_size, y_size, z_size,
                        dx_grid, dy_grid, dz_grid,
                        dir_x, dir_y, dir_z,
                        photon_generation_completed, weight,
                        itot, jtot);

            }
            // regular cell crossing: adjust tau and apply periodic BC
            else
            {
                photon.position.x += photon.direction.x>0 ? s_min : -s_min;
                photon.position.y += photon.direction.y>0 ? s_min : -s_min;
                photon.position.z += photon.direction.z>0 ? s_min : -s_min;
                
                // Cyclic boundary condition in x.
                photon.position.x = fmod(photon.position.x, x_size);
                if (photon.position.x < Float(0.))
                    photon.position.x += x_size;

                // Cyclic boundary condition in y.
                photon.position.y = fmod(photon.position.y, y_size);
                if (photon.position.y < Float(0.))
                    photon.position.y += y_size;
                
                tau -= d_max * k_ext_null;
                d_max = Float(0.);
                transition = true;
            }
        }
        else
        {
            Float dx = photon.direction.x * dn;
            Float dy = photon.direction.y * dn;
            Float dz = photon.direction.z * dn;

            photon.position.x = (dx > 0) ? min(photon.position.x + dx, (i_n+1) * kgrid_h - s_min) : max(photon.position.x + dx, (i_n) * kgrid_h + s_min);
            photon.position.y = (dy > 0) ? min(photon.position.y + dy, (j_n+1) * kgrid_h - s_min) : max(photon.position.y + dy, (j_n) * kgrid_h + s_min);
            photon.position.z = (dz > 0) ? min(photon.position.z + dz, (k_n+1) * kgrid_v - s_min) : max(photon.position.z + dz, (k_n) * kgrid_v + s_min);

            // Calculate the 3D index.
            const int i = float_to_int(photon.position.x, dx_grid, itot);
            const int j = float_to_int(photon.position.y, dy_grid, jtot);
            const int k = float_to_int(photon.position.z, dz_grid, ktot);
            const int ijk = i + j*itot + k*itot*jtot;
            
            // Handle the action.
            const Float random_number = rng();
            const Float k_ext_tot = k_ext[ijk].gas + k_ext[ijk].cloud;
            
            // Compute probability not being absorbed and store weighted absorption probability
            const Float f_no_abs = Float(1.) - (Float(1.) - ssa_asy[ijk].ssa) * (k_ext_tot/k_ext_null);
            if (ijk < 0) printf("writing hr lower \n");
            if (ijk >= itot*jtot*ktot) printf("writing hr upper \n");
            if (photon.kind == Photon_kind::Direct)
                write_photon_out(&atmos_direct_count[ijk], weight*(1-f_no_abs));
            else
                write_photon_out(&atmos_diffuse_count[ijk], weight*(1-f_no_abs));
            
            // Update weights (see Iwabuchi 2006: https://doi.org/10.1175/JAS3755.1)
            weight *= f_no_abs;
            if (weight < w_thres)
                weight = (rng() > weight) ? Float(0.) : Float(1.);

            // only with nonzero weight continue ray tracing, else start new ray
            if (weight > Float(0.))
            {
                // Null collision.
                if (random_number >= ssa_asy[ijk].ssa / (ssa_asy[ijk].ssa - Float(1.) + k_ext_null / k_ext_tot))
                {
                    d_max -= dn;
                }
                // Scattering.
                else
                {
                    d_max = Float(0.);
                    const Bool cloud_scatter = rng() < (k_ext[ijk].cloud / k_ext_tot);

                    const Float cos_scat = cloud_scatter ? henyey(ssa_asy[ijk].asy, rng()) : rayleigh(rng());
                    const Float sin_scat = max(Float(0.), sqrt(Float(1.) - cos_scat*cos_scat + Float_epsilon));

                    Vector t1{Float(0.), Float(0.), Float(0.)};
                    if (fabs(photon.direction.x) < fabs(photon.direction.y))
                    {
                        if (fabs(photon.direction.x) < fabs(photon.direction.z))
                            t1.x = Float(1.);
                        else
                            t1.z = Float(1.);
                    }
                    else
                    {
                        if (fabs(photon.direction.y) < fabs(photon.direction.z))
                            t1.y = Float(1.);
                        else
                            t1.z = Float(1.);
                    }
                    t1 = normalize(t1 - photon.direction*dot(t1, photon.direction));
                    Vector t2 = cross(photon.direction, t1);

                    const Float phi = Float(2.*M_PI)*rng();

                    photon.direction = cos_scat*photon.direction
                            + sin_scat*(sin(phi)*t1 + cos(phi)*t2);

                    photon.kind = Photon_kind::Diffuse;
                }
            }
            else
            {
                d_max = Float(0.);
                reset_photon(
                        photon, photons_shot, toa_down_count,
                        qrng, rng,
                        tod_inc_direct, tod_inc_diffuse,
                        x_size, y_size, z_size,
                        dx_grid, dy_grid, dz_grid,
                        dir_x, dir_y, dir_z,
                        photon_generation_completed, weight,
                        itot, jtot);
    
            }
        }
    }
}
