#include <float.h>
#include <curand_kernel.h>
#include "raytracer_kernels.h"
#include <iomanip>
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
    constexpr Float fov_h = 60./180.*M_PI;
    constexpr Float fov_v = 60./180.*M_PI; 
    constexpr Float zenith_0 = 60./180.*M_PI; 
    constexpr Float azimuth_0 = 360./180*M_PI;
    constexpr Float half_angle = 2.5/180. * M_PI; 

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
        Vector start;
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
    Float rayleigh_phase(const Float cos_angle)
    {
        return Float(3.)/(Float(16.)*M_PI) * (1+cos_angle*cos_angle);
    }

    __device__
    Float henyey_phase(const Float g, const Float cos_angle)
    {
        const Float denom = 1 + g*g - 2*g*cos_angle;
        return Float(1.)/(Float(4.)*M_PI) * (1-g*g) / (denom*sqrt(denom));
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
    
    __device__
    inline int reset_photon(
            Photon& photon, Int& photons_shot,
            const int ij_cam,
            const Float x_size, const Float y_size, const Float z_size,
            const Float dx_grid, const Float dy_grid, const Float dz_grid,
            const Float dir_x, const Float dir_y, const Float dir_z,
            const bool generation_completed, Float& weight,
            const int cam_nx, const int cam_ny,
            const Float cam_dx, const Float cam_dy,
            const int itot, const int jtot)
    {
        ++photons_shot;
        if (!generation_completed)
        {
            const int i = fmod(ij_cam, cam_nx);
            const int j = int(ij_cam/cam_nx);
    
            photon.position.x = Float(0.);// x_size * random_number_x / (1ULL << 32);
            photon.position.y = Float(24000.); // y_size * random_number_y / (1ULL << 32);
            photon.position.z = Float(3000.); // z_size;
            
            //const Float zenith_inc = fov_v* random_number_y/ (1ULL<<32);
            //const Float azimuth_inc = fov_h* random_number_x/ (1ULL<<32);
            const Float zenith_angle = zenith_0 + j * cam_dy;
            const Float azimuth_angle = azimuth_0 + i * cam_dx;
            photon.direction.x = -std::sin(zenith_angle) * std::cos(azimuth_angle);
            photon.direction.y = -std::sin(zenith_angle) * std::sin(azimuth_angle);
            photon.direction.z = -std::cos(zenith_angle);
            
            photon.kind = Photon_kind::Direct;
            weight = 1;
        }
        return 0;
    }
    
    __device__
    inline Float probability_from_sun(
            Photon photon, Vector sun_direction, const Float solid_angle, const Float g, const int scat_type_idx )
    {
        const Float cos_angle = dot(photon.direction, sun_direction) / (norm(photon.direction) * norm(sun_direction));
        if (scat_type_idx==0)
        {
            if (isnan(henyey_phase(g,cos_angle))) 
            {
            //printf("X %f %f %f\n",dot(photon.direction, sun_direction),norm(photon.direction),norm(sun_direction));
            printf("Y %f %f %f %f %f %f %f %f\n",dot(photon.direction, sun_direction), norm(photon.direction), photon.direction.x, photon.direction.y, photon.direction.z, sun_direction.x, sun_direction.y,sun_direction.z);
            
            }return henyey_phase(g, cos_angle); 
        }
        else
        {
            return rayleigh_phase(cos_angle);
        }
    }
    
    struct Quasi_random_number_generator_2d
    {
        __device__ Quasi_random_number_generator_2d(
                curandDirectionVectors32_t* vectors, unsigned int* constants, unsigned int offset)
        {
            curand_init(vectors[0], constants[0], offset, &state_x);
            curand_init(vectors[1], constants[1], offset, &state_y);
        }
    
        __device__ unsigned int x() { return curand(&state_x); }
        __device__ unsigned int y() { return curand(&state_y); }
    
        curandStateScrambledSobol32_t state_x;
        curandStateScrambledSobol32_t state_y;
    };
    
    
    __device__
    inline void write_photon_out(Float* field_out, const Float w)
    {
        atomicAdd(field_out, w);
    }

    __device__
    Float transmission_direct_sun(
            Photon photon,
            const Vector sun_dir,
            const Optics_ext* __restrict__ k_ext,
            const Float dx_grid, const Float dy_grid, const Float dz_grid,
            const Float x_size, const Float y_size, const Float z_size,
            const Float itot, const Float jtot, const Float ktot,
            const Float s_min)

    {
        Float tau_ext = 0;
        while (photon.position.z < z_size)
        {
            const int i = float_to_int(photon.position.x, dx_grid, itot);
            const int j = float_to_int(photon.position.y, dy_grid, jtot);
            const int k = float_to_int(photon.position.z, dz_grid, ktot);
            const Float sx = abs((sun_dir.x > 0) ? ((i+1) * dx_grid - photon.position.x)/sun_dir.x : (i*dx_grid - photon.position.x)/sun_dir.x);
            const Float sy = abs((sun_dir.y > 0) ? ((j+1) * dz_grid - photon.position.y)/sun_dir.y : (j*dy_grid - photon.position.y)/sun_dir.y);
            const Float sz = abs((sun_dir.z > 0) ? ((k+1) * dy_grid - photon.position.z)/sun_dir.z : (k*dz_grid - photon.position.z)/sun_dir.z);
            const Float s = min(sx, min(sy, sz));
            
            const int ijk = i + j*itot + k*itot*jtot;
            
            //update tau_ext
            const Float k_ext_loc = k_ext[ijk].gas + k_ext[ijk].cloud;
            tau_ext += k_ext_loc*s;
            
            //move photon
            const Float dx = sun_dir.x * s;
            const Float dy = sun_dir.y * s;
            const Float dz = sun_dir.z * s;
            
            photon.position.x += sun_dir.x * s;
            photon.position.y += sun_dir.y * s;
            photon.position.z += sun_dir.z * s;
            
            photon.position.x += sun_dir.x>0 ? s_min : -s_min;
            photon.position.y += sun_dir.y>0 ? s_min : -s_min;
            photon.position.z += sun_dir.z>0 ? s_min : -s_min;
            
            // Cyclic boundary condition in x.
            photon.position.x = fmod(photon.position.x, x_size);
            if (photon.position.x < Float(0.))
                photon.position.x += x_size;

            // Cyclic boundary condition in y.
            photon.position.y = fmod(photon.position.y, y_size);
            if (photon.position.y < Float(0.))
                photon.position.y += y_size;
        }
        return exp(Float(-1.) * tau_ext);
    }
}


__global__
void ray_tracer_kernel_bw(
        const Int photons_to_shoot,
        const Float* __restrict__ k_null_grid,
        Float* __restrict__ camera_count,
        int* __restrict__ counter,
        const int cam_nx, const int cam_ny,
        const Optics_ext* __restrict__ k_ext, const Optics_scat* __restrict__ ssa_asy,
        const Float surface_albedo,
        const Float diffuse_fraction,
        const Float x_size, const Float y_size, const Float z_size,
        const Float dx_grid, const Float dy_grid, const Float dz_grid,
        const Float dir_x, const Float dir_y, const Float dir_z,
        const int itot, const int jtot, const int ktot,
        curandDirectionVectors32_t* qrng_vectors, unsigned int* qrng_constants)
{
    const int n = blockDim.x * blockIdx.x + threadIdx.x;
    const int pixels_per_thread = 2;    
    Vector sun_direction;
    sun_direction.x = -dir_x;
    sun_direction.y = -dir_y;
    sun_direction.z = -dir_z;

    const Float solid_angle = 2.*M_PI*(1-half_angle);
    const Float kgrid_h = x_size/Float(ngrid_h);
    const Float kgrid_v = z_size/Float(ngrid_v);
   
    const Float cam_dx = fov_h / Float(cam_nx);
    const Float cam_dy = fov_v / Float(cam_ny);
    
    Random_number_generator<Float> rng(n);
    Quasi_random_number_generator_2d qrng(qrng_vectors, qrng_constants, n * photons_to_shoot);

    const Float s_min = x_size * Float_epsilon;
    const int ff = 32;
    // Set up the initial photons.
    
    //for (int ip=0; ip<pixels_per_thread; ++ip)
    while (counter[0] < ff*cam_nx*cam_ny)
    {
        const int ij_cam = atomicAdd(&counter[0], 1) / ff;
        //if (ij_cam >= cam_nx*cam_ny)
        //{
        //    printf("%d %d \n",ij_cam,cam_nx*cam_ny);
        //    return;
        //}
        
        const bool completed = false;
        Int photons_shot = Atomic_reduce_const;
        Float weight;
        
        Photon photon;
        reset_photon(
                photon, photons_shot,
                ij_cam,
                x_size, y_size, z_size,
                dx_grid, dy_grid, dz_grid,
                dir_x, dir_y, dir_z,
                completed, weight,
                cam_nx, cam_ny, cam_dx, cam_dy,
                itot, jtot);

        Float tau;
        Float d_max = Float(0.);
        Float k_ext_null;
        bool transition = false;

        while (photons_shot < photons_to_shoot/ff)
        {
            const bool photon_generation_completed = (photons_shot == photons_to_shoot - 1);
            // if d_max is zero, find current grid and maximum distance
            if (d_max == Float(0.))
            {
                const int i = float_to_int(photon.position.x, kgrid_h, ngrid_h);
                const int j = float_to_int(photon.position.y, kgrid_h, ngrid_h);
                const int k = float_to_int(photon.position.z, kgrid_v, ngrid_v);
                const Float sx = abs((photon.direction.x > 0) ? ((i+1) * kgrid_h - photon.position.x)/photon.direction.x : (i*kgrid_h - photon.position.x)/photon.direction.x);
                const Float sy = abs((photon.direction.y > 0) ? ((j+1) * kgrid_h - photon.position.y)/photon.direction.y : (j*kgrid_h - photon.position.y)/photon.direction.y);
                const Float sz = abs((photon.direction.z > 0) ? ((k+1) * kgrid_v - photon.position.z)/photon.direction.z : (k*kgrid_v - photon.position.z)/photon.direction.z);
                d_max = min(sx, min(sy, sz));
                const int ijk = i + j*ngrid_h + k*ngrid_h*ngrid_h;
                k_ext_null = k_null_grid[ijk];
            }
            
            if (!transition)
            {
                tau = sample_tau(rng());
            }
            transition = false;
            const Float dn = max(Float_epsilon, tau / k_ext_null);
            if (dn >= d_max)
            {
                const Float dx = photon.direction.x * (d_max);
                const Float dy = photon.direction.y * (d_max);
                const Float dz = photon.direction.z * (d_max);
                
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
            
                    // Update weights and add upward surface flux
                    weight *= surface_albedo;
            
                    if (weight < w_thres)
                        weight = (rng() > weight) ? Float(0.) : Float(1.);
                    
                    // SUN SCATTERING GOES HERE
                    const Float p_sun = probability_from_sun(photon, sun_direction, solid_angle, Float(0.), 1);
                    const Float trans_sun = transmission_direct_sun(photon, sun_direction, k_ext,
                                                dx_grid, dy_grid, dz_grid,
                                                x_size, y_size, z_size,
                                                itot, jtot, ktot,
                                                s_min);
                    //camera_count[ij_cam] += weight * p_sun * trans_sun;
                    atomicAdd(&camera_count[ij_cam], weight * p_sun * trans_sun);
                    //if (n==0) printf("1. %f \n", weight * p_sun * trans_sun);

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
                                photon, photons_shot,
                                ij_cam,
                                x_size, y_size, z_size,
                                dx_grid, dy_grid, dz_grid,
                                dir_x, dir_y, dir_z,
                                photon_generation_completed, weight,
                                cam_nx, cam_ny, cam_dx, cam_dy,
                                itot, jtot);
                    }
                }
            
                // TOA exit
                else if (photon.position.z >= z_size) 
                {
                    d_max = Float(0.);
                    reset_photon(
                            photon, photons_shot,
                            ij_cam,
                            x_size, y_size, z_size,
                            dx_grid, dy_grid, dz_grid,
                            dir_x, dir_y, dir_z,
                            photon_generation_completed, weight,
                            cam_nx, cam_ny, cam_dx, cam_dy,
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

                photon.position.x += dx;
                photon.position.y += dy;
                photon.position.z += dz;
                
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
                        const bool cloud_scatter = rng() < (k_ext[ijk].cloud / k_ext_tot);
                        const Float g = cloud_scatter ? ssa_asy[ijk].asy : Float(0.);
                        const Float cos_scat = cloud_scatter ? henyey(g, rng()) : rayleigh(rng());
                        const Float sin_scat = max(Float(0.), sqrt(Float(1.) - cos_scat*cos_scat + Float_epsilon));

                        // SUN SCATTERING GOES HERE
                        const Float p_sun = probability_from_sun(photon, sun_direction, solid_angle, g, !cloud_scatter);
                        const Float trans_sun = transmission_direct_sun(photon, sun_direction, k_ext,
                                                    dx_grid, dy_grid, dz_grid,
                                                    x_size, y_size, z_size,
                                                    itot, jtot, ktot,
                                                    s_min);
                        //if (n==0) printf("2. %f \n", weight * p_sun * trans_sun);
                        const Float xxx = weight * p_sun * trans_sun;
                        //if (n==0 && isnan(xxx)) printf("%f %f %f %d \n", weight, p_sun, trans_sun, cloud_scatter);
                        //camera_count[ij_cam] += xxx;
                        atomicAdd(&camera_count[ij_cam], xxx);
                        
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
                            photon, photons_shot,
                            ij_cam,
                            x_size, y_size, z_size,
                            dx_grid, dy_grid, dz_grid,
                            dir_x, dir_y, dir_z,
                            photon_generation_completed, weight,
                            cam_nx, cam_ny, cam_dx, cam_dy,
                            itot, jtot);
        
                }
            }
        }
    }
}
