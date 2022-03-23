/*
 * This file is imported from MicroHH (https://github.com/earth-system-radiation/earth-system-radiation)
 * and is adapted for the testing of the C++ interface to the
 * RTE+RRTMGP radiation code.
 *
 * MicroHH is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * MicroHH is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with MicroHH.  If not, see <http://www.gnu.org/licenses/>.
 */

#include <boost/algorithm/string.hpp>
#include <cmath>
#include <numeric>
#include <curand_kernel.h>

#include "Radiation_solver_bw.h"
#include "Status.h"
#include "Netcdf_interface.h"

#include "Array.h"
#include "Gas_concs.h"
#include "Gas_optics_rrtmgp.h"
#include "Optical_props.h"
#include "Source_functions.h"
#include "Fluxes.h"
#include "Rte_lw.h"
#include "Rte_sw.h"
#include "rrtmgp_kernel_launcher_cuda.h"
#include "gpoint_kernel_launcher_cuda.h"


namespace
{
    template<typename TF>__global__
    void move_optprop_kernel(
        const int ncol, const int nlay, const TF* __restrict__ tau_in, const TF* __restrict__ ssa_in, 
        TF* __restrict__ tau_out, TF* __restrict__ ssa_out)
    {
        const int icol = blockIdx.x*blockDim.x + threadIdx.x;
        const int ilay = blockIdx.y*blockDim.y + threadIdx.y;
        if ((icol<ncol) && (ilay<nlay))
        {
            const int idx = icol + ilay*ncol;
            tau_out[idx] = tau_in[idx];
            ssa_out[idx] = ssa_in[idx];
        }

    }


    template<typename TF>__global__
    void scaling_to_subset_kernel(
            const int ncol, const int ngpt, TF* __restrict__ toa_src, const TF tsi_scaling)
    {
        const int icol = blockIdx.x*blockDim.x + threadIdx.x;
        if ( ( icol < ncol)  )
        {
            const int idx = icol;
            toa_src[idx] *= tsi_scaling;
        }
    }

    template<typename TF>
    void scaling_to_subset(
            const int ncol, const int ngpt, Array_gpu<TF,1>& toa_src, const TF tsi_scaling)
    {
        const int block_col = 16;
        const int grid_col  = ncol/block_col + (ncol%block_col > 0);
        
        dim3 grid_gpu(grid_col, 1);
        dim3 block_gpu(block_col, 1);
        scaling_to_subset_kernel<<<grid_gpu, block_gpu>>>(
            ncol, ngpt, toa_src.ptr(), tsi_scaling);
    }
    
    template<typename TF>__global__
    void scaling_to_subset_kernel(
            const int ncol, const int ngpt, TF* __restrict__ toa_src, const TF* __restrict__ tsi_scaling)
    {
        const int icol = blockIdx.x*blockDim.x + threadIdx.x;
        if ( ( icol < ncol)  )
        {
            const int idx = icol;
            toa_src[idx] *= tsi_scaling[icol];
        }
    }

    template<typename TF>
    void scaling_to_subset(
            const int ncol, const int ngpt, Array_gpu<TF,1>& toa_src, const Array_gpu<TF,1>& tsi_scaling)
    {
        const int block_col = 16;
        const int grid_col  = ncol/block_col + (ncol%block_col > 0);
        
        dim3 grid_gpu(grid_col, 1);
        dim3 block_gpu(block_col, 1);
        scaling_to_subset_kernel<<<grid_gpu, block_gpu>>>(
            ncol, ngpt, toa_src.ptr(), tsi_scaling.ptr());
    }


    std::vector<std::string> get_variable_string(
            const std::string& var_name,
            std::vector<int> i_count,
            Netcdf_handle& input_nc,
            const int string_len,
            bool trim=true)
    {
        // Multiply all elements in i_count.
        int total_count = std::accumulate(i_count.begin(), i_count.end(), 1, std::multiplies<>());

        // Add the string length as the rightmost dimension.
        i_count.push_back(string_len);

        // Read the entire char array;
        std::vector<char> var_char;
        var_char = input_nc.get_variable<char>(var_name, i_count);

        std::vector<std::string> var;

        for (int n=0; n<total_count; ++n)
        {
            std::string s(var_char.begin()+n*string_len, var_char.begin()+(n+1)*string_len);
            if (trim)
                boost::trim(s);
            var.push_back(s);
        }

        return var;
    }

    template<typename TF>
    Gas_optics_rrtmgp_gpu<TF> load_and_init_gas_optics(
            const Gas_concs_gpu<TF>& gas_concs,
            const std::string& coef_file)
    {
        // READ THE COEFFICIENTS FOR THE OPTICAL SOLVER.
        Netcdf_file coef_nc(coef_file, Netcdf_mode::Read);

        // Read k-distribution information.
        const int n_temps = coef_nc.get_dimension_size("temperature");
        const int n_press = coef_nc.get_dimension_size("pressure");
        const int n_absorbers = coef_nc.get_dimension_size("absorber");
        const int n_char = coef_nc.get_dimension_size("string_len");
        const int n_minorabsorbers = coef_nc.get_dimension_size("minor_absorber");
        const int n_extabsorbers = coef_nc.get_dimension_size("absorber_ext");
        const int n_mixingfracs = coef_nc.get_dimension_size("mixing_fraction");
        const int n_layers = coef_nc.get_dimension_size("atmos_layer");
        const int n_bnds = coef_nc.get_dimension_size("bnd");
        const int n_gpts = coef_nc.get_dimension_size("gpt");
        const int n_pairs = coef_nc.get_dimension_size("pair");
        const int n_minor_absorber_intervals_lower = coef_nc.get_dimension_size("minor_absorber_intervals_lower");
        const int n_minor_absorber_intervals_upper = coef_nc.get_dimension_size("minor_absorber_intervals_upper");
        const int n_contributors_lower = coef_nc.get_dimension_size("contributors_lower");
        const int n_contributors_upper = coef_nc.get_dimension_size("contributors_upper");

        // Read gas names.
        Array<std::string,1> gas_names(
                get_variable_string("gas_names", {n_absorbers}, coef_nc, n_char, true), {n_absorbers});

        Array<int,3> key_species(
                coef_nc.get_variable<int>("key_species", {n_bnds, n_layers, 2}),
                {2, n_layers, n_bnds});
        Array<TF,2> band_lims(coef_nc.get_variable<TF>("bnd_limits_wavenumber", {n_bnds, 2}), {2, n_bnds});
        Array<int,2> band2gpt(coef_nc.get_variable<int>("bnd_limits_gpt", {n_bnds, 2}), {2, n_bnds});
        Array<TF,1> press_ref(coef_nc.get_variable<TF>("press_ref", {n_press}), {n_press});
        Array<TF,1> temp_ref(coef_nc.get_variable<TF>("temp_ref", {n_temps}), {n_temps});

        TF temp_ref_p = coef_nc.get_variable<TF>("absorption_coefficient_ref_P");
        TF temp_ref_t = coef_nc.get_variable<TF>("absorption_coefficient_ref_T");
        TF press_ref_trop = coef_nc.get_variable<TF>("press_ref_trop");

        Array<TF,3> kminor_lower(
                coef_nc.get_variable<TF>("kminor_lower", {n_temps, n_mixingfracs, n_contributors_lower}),
                {n_contributors_lower, n_mixingfracs, n_temps});
        Array<TF,3> kminor_upper(
                coef_nc.get_variable<TF>("kminor_upper", {n_temps, n_mixingfracs, n_contributors_upper}),
                {n_contributors_upper, n_mixingfracs, n_temps});

        Array<std::string,1> gas_minor(get_variable_string("gas_minor", {n_minorabsorbers}, coef_nc, n_char),
                                       {n_minorabsorbers});

        Array<std::string,1> identifier_minor(
                get_variable_string("identifier_minor", {n_minorabsorbers}, coef_nc, n_char), {n_minorabsorbers});

        Array<std::string,1> minor_gases_lower(
                get_variable_string("minor_gases_lower", {n_minor_absorber_intervals_lower}, coef_nc, n_char),
                {n_minor_absorber_intervals_lower});
        Array<std::string,1> minor_gases_upper(
                get_variable_string("minor_gases_upper", {n_minor_absorber_intervals_upper}, coef_nc, n_char),
                {n_minor_absorber_intervals_upper});

        Array<int,2> minor_limits_gpt_lower(
                coef_nc.get_variable<int>("minor_limits_gpt_lower", {n_minor_absorber_intervals_lower, n_pairs}),
                {n_pairs, n_minor_absorber_intervals_lower});
        Array<int,2> minor_limits_gpt_upper(
                coef_nc.get_variable<int>("minor_limits_gpt_upper", {n_minor_absorber_intervals_upper, n_pairs}),
                {n_pairs, n_minor_absorber_intervals_upper});

        Array<BOOL_TYPE,1> minor_scales_with_density_lower(
                coef_nc.get_variable<BOOL_TYPE>("minor_scales_with_density_lower", {n_minor_absorber_intervals_lower}),
                {n_minor_absorber_intervals_lower});
        Array<BOOL_TYPE,1> minor_scales_with_density_upper(
                coef_nc.get_variable<BOOL_TYPE>("minor_scales_with_density_upper", {n_minor_absorber_intervals_upper}),
                {n_minor_absorber_intervals_upper});

        Array<BOOL_TYPE,1> scale_by_complement_lower(
                coef_nc.get_variable<BOOL_TYPE>("scale_by_complement_lower", {n_minor_absorber_intervals_lower}),
                {n_minor_absorber_intervals_lower});
        Array<BOOL_TYPE,1> scale_by_complement_upper(
                coef_nc.get_variable<BOOL_TYPE>("scale_by_complement_upper", {n_minor_absorber_intervals_upper}),
                {n_minor_absorber_intervals_upper});

        Array<std::string,1> scaling_gas_lower(
                get_variable_string("scaling_gas_lower", {n_minor_absorber_intervals_lower}, coef_nc, n_char),
                {n_minor_absorber_intervals_lower});
        Array<std::string,1> scaling_gas_upper(
                get_variable_string("scaling_gas_upper", {n_minor_absorber_intervals_upper}, coef_nc, n_char),
                {n_minor_absorber_intervals_upper});

        Array<int,1> kminor_start_lower(
                coef_nc.get_variable<int>("kminor_start_lower", {n_minor_absorber_intervals_lower}),
                {n_minor_absorber_intervals_lower});
        Array<int,1> kminor_start_upper(
                coef_nc.get_variable<int>("kminor_start_upper", {n_minor_absorber_intervals_upper}),
                {n_minor_absorber_intervals_upper});

        Array<TF,3> vmr_ref(
                coef_nc.get_variable<TF>("vmr_ref", {n_temps, n_extabsorbers, n_layers}),
                {n_layers, n_extabsorbers, n_temps});

        Array<TF,4> kmajor(
                coef_nc.get_variable<TF>("kmajor", {n_temps, n_press+1, n_mixingfracs, n_gpts}),
                {n_gpts, n_mixingfracs, n_press+1, n_temps});

        // Keep the size at zero, if it does not exist.
        Array<TF,3> rayl_lower;
        Array<TF,3> rayl_upper;

        if (coef_nc.variable_exists("rayl_lower"))
        {
            rayl_lower.set_dims({n_gpts, n_mixingfracs, n_temps});
            rayl_upper.set_dims({n_gpts, n_mixingfracs, n_temps});
            rayl_lower = coef_nc.get_variable<TF>("rayl_lower", {n_temps, n_mixingfracs, n_gpts});
            rayl_upper = coef_nc.get_variable<TF>("rayl_upper", {n_temps, n_mixingfracs, n_gpts});
        }

        // Is it really LW if so read these variables as well.
        if (coef_nc.variable_exists("totplnk"))
        {
            int n_internal_sourcetemps = coef_nc.get_dimension_size("temperature_Planck");

            Array<TF,2> totplnk(
                    coef_nc.get_variable<TF>( "totplnk", {n_bnds, n_internal_sourcetemps}),
                    {n_internal_sourcetemps, n_bnds});
            Array<TF,4> planck_frac(
                    coef_nc.get_variable<TF>("plank_fraction", {n_temps, n_press+1, n_mixingfracs, n_gpts}),
                    {n_gpts, n_mixingfracs, n_press+1, n_temps});

            // Construct the k-distribution.
            return Gas_optics_rrtmgp_gpu<TF>(
                    gas_concs,
                    gas_names,
                    key_species,
                    band2gpt,
                    band_lims,
                    press_ref,
                    press_ref_trop,
                    temp_ref,
                    temp_ref_p,
                    temp_ref_t,
                    vmr_ref,
                    kmajor,
                    kminor_lower,
                    kminor_upper,
                    gas_minor,
                    identifier_minor,
                    minor_gases_lower,
                    minor_gases_upper,
                    minor_limits_gpt_lower,
                    minor_limits_gpt_upper,
                    minor_scales_with_density_lower,
                    minor_scales_with_density_upper,
                    scaling_gas_lower,
                    scaling_gas_upper,
                    scale_by_complement_lower,
                    scale_by_complement_upper,
                    kminor_start_lower,
                    kminor_start_upper,
                    totplnk,
                    planck_frac,
                    rayl_lower,
                    rayl_upper);
        }
        else
        {
            Array<TF,1> solar_src_quiet(
                    coef_nc.get_variable<TF>("solar_source_quiet", {n_gpts}), {n_gpts});
            Array<TF,1> solar_src_facular(
                    coef_nc.get_variable<TF>("solar_source_facular", {n_gpts}), {n_gpts});
            Array<TF,1> solar_src_sunspot(
                    coef_nc.get_variable<TF>("solar_source_sunspot", {n_gpts}), {n_gpts});

            TF tsi = coef_nc.get_variable<TF>("tsi_default");
            TF mg_index = coef_nc.get_variable<TF>("mg_default");
            TF sb_index = coef_nc.get_variable<TF>("sb_default");

            return Gas_optics_rrtmgp_gpu<TF>(
                    gas_concs,
                    gas_names,
                    key_species,
                    band2gpt,
                    band_lims,
                    press_ref,
                    press_ref_trop,
                    temp_ref,
                    temp_ref_p,
                    temp_ref_t,
                    vmr_ref,
                    kmajor,
                    kminor_lower,
                    kminor_upper,
                    gas_minor,
                    identifier_minor,
                    minor_gases_lower,
                    minor_gases_upper,
                    minor_limits_gpt_lower,
                    minor_limits_gpt_upper,
                    minor_scales_with_density_lower,
                    minor_scales_with_density_upper,
                    scaling_gas_lower,
                    scaling_gas_upper,
                    scale_by_complement_lower,
                    scale_by_complement_upper,
                    kminor_start_lower,
                    kminor_start_upper,
                    solar_src_quiet,
                    solar_src_facular,
                    solar_src_sunspot,
                    tsi,
                    mg_index,
                    sb_index,
                    rayl_lower,
                    rayl_upper);
        }
        // End reading of k-distribution.
    }

    template<typename TF>
    Cloud_optics_gpu<TF> load_and_init_cloud_optics(
            const std::string& coef_file)
    {
        // READ THE COEFFICIENTS FOR THE OPTICAL SOLVER.
        Netcdf_file coef_nc(coef_file, Netcdf_mode::Read);

        // Read look-up table coefficient dimensions
        int n_band     = coef_nc.get_dimension_size("nband");
        int n_rghice   = coef_nc.get_dimension_size("nrghice");
        int n_size_liq = coef_nc.get_dimension_size("nsize_liq");
        int n_size_ice = coef_nc.get_dimension_size("nsize_ice");

        Array<TF,2> band_lims_wvn(coef_nc.get_variable<TF>("bnd_limits_wavenumber", {n_band, 2}), {2, n_band});

        // Read look-up table constants.
        TF radliq_lwr = coef_nc.get_variable<TF>("radliq_lwr");
        TF radliq_upr = coef_nc.get_variable<TF>("radliq_upr");
        TF radliq_fac = coef_nc.get_variable<TF>("radliq_fac");

        TF radice_lwr = coef_nc.get_variable<TF>("radice_lwr");
        TF radice_upr = coef_nc.get_variable<TF>("radice_upr");
        TF radice_fac = coef_nc.get_variable<TF>("radice_fac");

        Array<TF,2> lut_extliq(
                coef_nc.get_variable<TF>("lut_extliq", {n_band, n_size_liq}), {n_size_liq, n_band});
        Array<TF,2> lut_ssaliq(
                coef_nc.get_variable<TF>("lut_ssaliq", {n_band, n_size_liq}), {n_size_liq, n_band});
        Array<TF,2> lut_asyliq(
                coef_nc.get_variable<TF>("lut_asyliq", {n_band, n_size_liq}), {n_size_liq, n_band});

        Array<TF,3> lut_extice(
                coef_nc.get_variable<TF>("lut_extice", {n_rghice, n_band, n_size_ice}), {n_size_ice, n_band, n_rghice});
        Array<TF,3> lut_ssaice(
                coef_nc.get_variable<TF>("lut_ssaice", {n_rghice, n_band, n_size_ice}), {n_size_ice, n_band, n_rghice});
        Array<TF,3> lut_asyice(
                coef_nc.get_variable<TF>("lut_asyice", {n_rghice, n_band, n_size_ice}), {n_size_ice, n_band, n_rghice});

        return Cloud_optics_gpu<TF>(
                band_lims_wvn,
                radliq_lwr, radliq_upr, radliq_fac,
                radice_lwr, radice_upr, radice_fac,
                lut_extliq, lut_ssaliq, lut_asyliq,
                lut_extice, lut_ssaice, lut_asyice);
    }
}

template<typename TF>
Radiation_solver_longwave<TF>::Radiation_solver_longwave(
        const Gas_concs_gpu<TF>& gas_concs,
        const std::string& file_name_gas,
        const std::string& file_name_cloud)
{
    // Construct the gas optics classes for the solver.
    this->kdist_gpu = std::make_unique<Gas_optics_rrtmgp_gpu<TF>>(
            load_and_init_gas_optics<TF>(gas_concs, file_name_gas));

    this->cloud_optics_gpu = std::make_unique<Cloud_optics_gpu<TF>>(
            load_and_init_cloud_optics<TF>(file_name_cloud));
}

template<typename TF>
void Radiation_solver_longwave<TF>::solve_gpu(
        const bool switch_fluxes,
        const bool switch_cloud_optics,
        const bool switch_output_optical,
        const bool switch_output_bnd_fluxes,
        const Gas_concs_gpu<TF>& gas_concs,
        const Array_gpu<TF,2>& p_lay, const Array_gpu<TF,2>& p_lev,
        const Array_gpu<TF,2>& t_lay, const Array_gpu<TF,2>& t_lev,
        Array_gpu<TF,2>& col_dry,
        const Array_gpu<TF,1>& t_sfc, const Array_gpu<TF,2>& emis_sfc,
        const Array_gpu<TF,2>& lwp, const Array_gpu<TF,2>& iwp,
        const Array_gpu<TF,2>& rel, const Array_gpu<TF,2>& rei,
        Array_gpu<TF,3>& tau, Array_gpu<TF,3>& lay_source,
        Array_gpu<TF,3>& lev_source_inc, Array_gpu<TF,3>& lev_source_dec, Array_gpu<TF,2>& sfc_source,
        Array_gpu<TF,2>& lw_flux_up, Array_gpu<TF,2>& lw_flux_dn, Array_gpu<TF,2>& lw_flux_net,
        Array_gpu<TF,3>& lw_bnd_flux_up, Array_gpu<TF,3>& lw_bnd_flux_dn, Array_gpu<TF,3>& lw_bnd_flux_net)
{
    const int n_col = p_lay.dim(1);
    const int n_lay = p_lay.dim(2);
    const int n_lev = p_lev.dim(2);
    const int n_gpt = this->kdist_gpu->get_ngpt();
    const int n_bnd = this->kdist_gpu->get_nband();

    const BOOL_TYPE top_at_1 = p_lay({1, 1}) < p_lay({1, n_lay});

    optical_props = std::make_unique<Optical_props_1scl_gpu<TF>>(n_col, n_lay, *kdist_gpu);
    sources = std::make_unique<Source_func_lw_gpu<TF>>(n_col, n_lay, *kdist_gpu);

    if (switch_cloud_optics)
        cloud_optical_props = std::make_unique<Optical_props_1scl_gpu<TF>>(n_col, n_lay, *cloud_optics_gpu);

    if (col_dry.size() == 0)
    {
        col_dry.set_dims({n_col, n_lay});
        Gas_optics_rrtmgp_gpu<TF>::get_col_dry(col_dry, gas_concs.get_vmr("h2o"), p_lev);
    }

    if (switch_fluxes)
    {
        rrtmgp_kernel_launcher_cuda::zero_array(n_lev, n_col, lw_flux_up);
        rrtmgp_kernel_launcher_cuda::zero_array(n_lev, n_col, lw_flux_dn);
        rrtmgp_kernel_launcher_cuda::zero_array(n_lev, n_col, lw_flux_net);
    }
    
    const Array<int, 2>& band_limits_gpt(this->kdist_gpu->get_band_lims_gpoint());
    for (int igpt=1; igpt<=n_gpt; ++igpt)
    {
        int band = 0;
        for (int ibnd=1; ibnd<=n_bnd; ++ibnd)
        {
            if (igpt <= band_limits_gpt({2, ibnd}))
            {
                band = ibnd;
                break;
            }
        }
        
        kdist_gpu->gas_optics(
                igpt-1,
                p_lay,
                p_lev,
                t_lay,
                t_sfc,
                gas_concs,
                optical_props,
                *sources,
                col_dry,
                t_lev);

        if (switch_cloud_optics)
        {
            cloud_optics_gpu->cloud_optics(
                    band-1,
                    lwp,
                    iwp,
                    rel,
                    rei,
                    *cloud_optical_props);
            // cloud->delta_scale();

            // Add the cloud optical props to the gas optical properties.
            add_to(
                    dynamic_cast<Optical_props_1scl_gpu<TF>&>(*optical_props),
                    dynamic_cast<Optical_props_1scl_gpu<TF>&>(*cloud_optical_props));
        }
        
        // Store the optical properties, if desired.
        if (switch_output_optical)
        {
            gpoint_kernel_launcher_cuda::get_from_gpoint(
                    n_col, n_lay, igpt-1, tau, lay_source, lev_source_inc, lev_source_dec,
                    optical_props->get_tau(), (*sources).get_lay_source(),
                    (*sources).get_lev_source_inc(), (*sources).get_lev_source_dec());

            gpoint_kernel_launcher_cuda::get_from_gpoint(
                    n_col, igpt-1, sfc_source, (*sources).get_sfc_source());
        }


        if (switch_fluxes)
        {
            constexpr int n_ang = 1;

            std::unique_ptr<Fluxes_broadband_gpu<TF>> fluxes =
                    std::make_unique<Fluxes_broadband_gpu<TF>>(n_col, 1, n_lev);

            rte_lw.rte_lw(
                    optical_props,
                    top_at_1,
                    *sources,
                    emis_sfc.subset({{ {band, band}, {1, n_col}}}),
                    Array_gpu<TF,1>(), // Add an empty array, no inc_flux.
                    (*fluxes).get_flux_up(),
                    (*fluxes).get_flux_dn(),
                    n_ang);

            (*fluxes).net_flux();
            
            // Copy the data to the output.
            gpoint_kernel_launcher_cuda::add_from_gpoint(
                    n_col, n_lev, lw_flux_up, lw_flux_dn, lw_flux_net,
                    (*fluxes).get_flux_up(), (*fluxes).get_flux_dn(), (*fluxes).get_flux_net());


            if (switch_output_bnd_fluxes)
            {
                gpoint_kernel_launcher_cuda::get_from_gpoint(
                        n_col, n_lev, igpt-1, lw_bnd_flux_up, lw_bnd_flux_dn, lw_bnd_flux_net,
                        (*fluxes).get_flux_up(), (*fluxes).get_flux_dn(), (*fluxes).get_flux_net());

            }
        }
    }
}


template<typename TF>
TF get_x(const TF wv)
{
    const TF a = (wv - TF(442.0)) * ((wv < TF(442.0)) ? TF(0.0624) : TF(0.0374));  
    const TF b = (wv - TF(599.8)) * ((wv < TF(599.8)) ? TF(0.0264) : TF(0.0323));  
    const TF c = (wv - TF(501.1)) * ((wv < TF(501.1)) ? TF(0.0490) : TF(0.0382));  
    return TF(0.362) * std::exp(TF(-0.5)*a*a) + TF(1.056) * std::exp(TF(-0.5)*b*b) - TF(0.065) * std::exp(TF(-0.5)*c*c);
}

template<typename TF>
TF get_y(const TF wv)
{
    const TF a = (wv - TF(568.8)) * ((wv < TF(568.8)) ? TF(0.0213) : TF(0.0247));  
    const TF b = (wv - TF(530.9)) * ((wv < TF(530.9)) ? TF(0.0613) : TF(0.0322));  
    return TF(0.821) * std::exp(TF(-0.5)*a*a) + TF(.286) * std::exp(TF(-0.5)*b*b);
}
template<typename TF>
TF get_z(const TF wv)
{
    const TF a = (wv - TF(437.0)) * ((wv < TF(437.0)) ? TF(0.0845) : TF(0.0278));  
    const TF b = (wv - TF(459.0)) * ((wv < TF(459.0)) ? TF(0.0385) : TF(0.0725));  
    return TF(1.217) * std::exp(TF(-0.5)*a*a) + TF(0.681) * std::exp(TF(-0.5)*b*b);
}
    

template<typename TF>
TF Planck(TF wv)
{
    const TF h = TF(6.62607015e-34);
    const TF c = TF(299792458.);
    const TF k = TF(1.380649e-23);
    const TF nom = 2*h*c*c / (wv*wv*wv*wv*wv);
    const TF denom = exp(h*c/(wv*k*TF(5778)))-TF(1.);
    return (nom/denom);
}

template<typename TF>
TF Planck_integrator(
        const TF wv1, const TF wv2)
{
    const int n = 100;
    const TF sa = 6.771e-5;
    const TF dwv = (wv2-wv1)/TF(n);
    TF sum = 0;
    for (int i=0; i<n; ++i)
    {
        const TF wv = (wv1 + i*dwv)*1e-9;
        sum += Planck(wv) * dwv;
    }
    return sum * TF(1e-9) * sa;
}

template<typename TF>
TF rayleigh_mean(
    const TF wv1, const TF wv2)
{
    const TF n = 1.000287;
    const TF Ns = 2.546899e19;
    const TF dwv = (wv2-wv1)/100.;
    TF sigma_mean = 0;
    for (int i=0; i<100; ++i)
    {
        const TF wv = (wv1 + i*dwv);
        const TF n = 1+1e-8*(8060.77 + 2481070/(132.274-pow((wv/1e3),-2)) + 17456.3/(39.32957-pow((wv/1e3),-2)));
        const TF nom = 24*M_PI*M_PI*M_PI*pow((n*n-1),2);
        const TF denom = pow((wv/1e7),4) * Ns*Ns * pow((n*n +2), 2);
        sigma_mean += nom/denom * 1.055;
    }
    return sigma_mean / 100.;
}

template<typename TF>
TF xyz_irradiance(
        const TF wv1, const TF wv2,
        TF (*get_xyz)(TF))
{
    TF wv = wv1; //int n = 1000; 
    const TF dwv = TF(0.1);//(wv2-wv1)/TF(n);
    TF sum = 0;
    //for (int i=0; i<n; ++i)
    while (wv < wv2)
    {
        const TF wv_tmp = wv + dwv/TF(2.);// = (wv1 + i*dwv) + dwv/TF(2.);
        //const TF wv = (wv1 + i*dwv) + dwv/TF(2.);
        sum += get_xyz(wv_tmp) * Planck(wv_tmp*TF(1e-9)) * dwv;
        wv += dwv;
    }
    return sum * TF(1e-9);
}

template<typename TF>
Radiation_solver_shortwave<TF>::Radiation_solver_shortwave(
        const Gas_concs_gpu<TF>& gas_concs,
        const std::string& file_name_gas,
        const std::string& file_name_cloud)
{
    // Construct the gas optics classes for the solver.
    this->kdist_gpu = std::make_unique<Gas_optics_rrtmgp_gpu<TF>>(
            load_and_init_gas_optics<TF>(gas_concs, file_name_gas));

    this->cloud_optics_gpu = std::make_unique<Cloud_optics_gpu<TF>>(
            load_and_init_cloud_optics<TF>(file_name_cloud));
}


template<typename TF>
void Radiation_solver_shortwave<TF>::solve_gpu(
        const bool tune_step,
        const bool switch_cloud_optics,
        const bool switch_output_bnd_fluxes,
        const Int ray_count,
        const Gas_concs_gpu<TF>& gas_concs,
        const Array_gpu<TF,2>& p_lay, const Array_gpu<TF,2>& p_lev,
        const Array_gpu<TF,2>& t_lay, const Array_gpu<TF,2>& t_lev,
        const Array_gpu<TF,1>& z_lev,
        const Array_gpu<TF,1>& grid_dims,
        Array_gpu<TF,2>& col_dry,
        const Array_gpu<TF,2>& sfc_alb_dir, const Array_gpu<TF,2>& sfc_alb_dif,
        const Array_gpu<TF,1>& tsi_scaling, const Array_gpu<TF,1>& mu0,
        const Array_gpu<TF,2>& lwp, const Array_gpu<TF,2>& iwp,
        const Array_gpu<TF,2>& rel, const Array_gpu<TF,2>& rei,
        Array_gpu<TF,3>& XYZ)

{
    const int n_col = p_lay.dim(1);
    const int n_lay = p_lay.dim(2);
    const int n_lev = p_lev.dim(2);
    const int n_gpt = this->kdist_gpu->get_ngpt();
    const int n_bnd = this->kdist_gpu->get_nband();
    
    const int dx_grid = grid_dims({1});
    const int dy_grid = grid_dims({2});
    const int dz_grid = grid_dims({3});
    const int n_z     = grid_dims({4});
    const int n_col_y = grid_dims({5});
    const int n_col_x = grid_dims({6});
    
    const int cam_nx = XYZ.dim(1);
    const int cam_ny = XYZ.dim(2);
    const int cam_ns = XYZ.dim(3);
    const BOOL_TYPE top_at_1 = p_lay({1, 1}) < p_lay({1, n_lay});

    optical_props = std::make_unique<Optical_props_2str_gpu<TF>>(n_col, n_lay, *kdist_gpu);
    cloud_optical_props = std::make_unique<Optical_props_2str_gpu<TF>>(n_col, n_lay, *cloud_optics_gpu);
    
    if (col_dry.size() == 0)
    {
        col_dry.set_dims({n_col, n_lay});
        Gas_optics_rrtmgp_gpu<TF>::get_col_dry(col_dry, gas_concs.get_vmr("h2o"), p_lev);
    }

    Array_gpu<TF,1> toa_src({n_col});
    Array_gpu<TF,2> flux_camera({cam_nx, cam_ny});

    Array<int,2> cld_mask_liq({n_col, n_lay});
    Array<int,2> cld_mask_ice({n_col, n_lay});
    
    rrtmgp_kernel_launcher_cuda::zero_array(cam_ns, cam_nx, cam_ny, XYZ);

    const Array<int, 2>& band_limits_gpt(this->kdist_gpu->get_band_lims_gpoint());
    TF total_source = 0.;
    
    for (int igpt=1; igpt<=n_gpt; ++igpt)
    {
        int band = 0;
        for (int ibnd=1; ibnd<=n_bnd; ++ibnd)
        {
            if (igpt <= band_limits_gpt({2, ibnd}))
            {
                band = ibnd;
                break;
            }
        }
        //if (!tune_step && (! (band == 10 || band == 11 || band ==12))) continue; 
        //if (band !=11) continue; 
        if (igpt !=1) continue; 

        const TF solar_source_band = kdist_gpu->band_source(band_limits_gpt({1,band}), band_limits_gpt({2,band}));
        
        printf("-> %d %f \n", band, solar_source_band);
        kdist_gpu->gas_optics(
                  igpt-1,
                  p_lay,
                  p_lev,
                  t_lay,
                  gas_concs,
                  optical_props,
                  toa_src,
                  col_dry);
        
        // scaling_to_subset(n_col, n_gpt, toa_src, tsi_scaling);
        // scaling_to_subset(n_col, n_gpt, toa_src, local_planck/total_planck);
        
        if (switch_cloud_optics)
        {
            cloud_optics_gpu->cloud_optics(
                    band-1,
                    lwp,
                    iwp,
                    rel,
                    rei,
                    *cloud_optical_props);

 
            cloud_optical_props->delta_scale();
        
            // Add the cloud optical props to the gas optical properties.
            add_to(
                    dynamic_cast<Optical_props_2str_gpu<TF>&>(*optical_props),
                    dynamic_cast<Optical_props_2str_gpu<TF>&>(*cloud_optical_props));
        }
        if (tune_step) return;
        const Array<TF, 2>& band_limits_wn(this->kdist_gpu->get_band_lims_wavenumber());
        
        const int nwv = 1;
        const TF wv1 = 1. / band_limits_wn({2,band}) * Float(1.e7);
        const TF wv2 = 1. / band_limits_wn({1,band}) * Float(1.e7);
        const TF dwv = (wv2-wv1)/TF(nwv);
        const TF total_planck = Planck_integrator(wv1,wv2);
        
        for (int iwv=0; iwv<nwv; ++iwv)
        {
            const TF wv1_sub = wv1 + iwv*dwv;
            const TF wv2_sub = wv1 + (iwv+1)*dwv;
            const TF local_planck = Planck_integrator(wv1_sub,wv2_sub);   
            const TF rayleigh = rayleigh_mean(wv1_sub, wv2_sub);   
            // //const TF toa_factor = local_planck / total_planck * TF(1.)/solar_source_band; 
            // const TF toa_factor = TF(.9287681652667048)/solar_source_band; 
            const TF toa_factor = TF(1.);
            //total_source += toa_src({1}) * toa_factor * mu0({1});
            std::unique_ptr<Fluxes_broadband_gpu<TF>> fluxes =
                    std::make_unique<Fluxes_broadband_gpu<TF>>(cam_nx, cam_ny, 1);
           // HIGHTUNES OPP:
    Netcdf_file input_nc("hightune_opts.nc", Netcdf_mode::Read);

    const int hnx = input_nc.get_dimension_size("x");
    const int hny = input_nc.get_dimension_size("y");
    const int hnz = input_nc.get_dimension_size("z");
   
    Array<TF,2> tau_in(input_nc.get_variable<TF>("tau", {n_lay, n_col_y, n_col_x}), {n_col_y*n_col_x, n_lay});
    Array<TF,2> ssa_in(input_nc.get_variable<TF>("ssa", {n_lay, n_col_y, n_col_x}), {n_col_y*n_col_x, n_lay});
    Array_gpu<TF,2> tau_gpu_in(tau_in);
    Array_gpu<TF,2> ssa_gpu_in(ssa_in);
    const int block_col = 16;
    const int block_lay = 16;
    const int grid_col  = n_col/block_col + (n_col%block_col > 0);
    const int grid_lay  = n_lay/block_lay + (n_lay%block_lay > 0);
    
    dim3 grid_gpu(grid_col, grid_lay, 1);
    dim3 block_gpu(block_col, block_lay, 1);
   
    move_optprop_kernel<<<grid_gpu,block_gpu>>>(n_col, n_lay, tau_gpu_in.ptr(), ssa_gpu_in.ptr(),  optical_props->get_tau().ptr(), optical_props->get_ssa().ptr());
   // printf("-- \n");
   // (*optical_props).get_tau() = std::move(tau_gpu_in);
   // printf("-- \n");
   // optical_props->get_ssa() = std::move(ssa_gpu_in);

            // XYZ factors
            Array<TF,1> xyz_factor({3});
            xyz_factor({1}) = 1;//xyz_irradiance(wv1_sub,wv2_sub,&get_x);
            xyz_factor({2}) = 1;//xyz_irradiance(wv1_sub,wv2_sub,&get_y);
            xyz_factor({3}) = 1;//xyz_irradiance(wv1_sub,wv2_sub,&get_z);
            Array_gpu<TF,1> xyz_factor_gpu(xyz_factor);
            printf("%d %d %f %f %f %f \n",iwv,igpt,toa_factor,xyz_factor({1}),toa_src({1}),total_planck);
            if (!switch_cloud_optics) 
            {
                rrtmgp_kernel_launcher_cuda::zero_array(n_col, n_lay, cloud_optical_props->get_tau());
                rrtmgp_kernel_launcher_cuda::zero_array(n_col, n_lay, cloud_optical_props->get_ssa());
            }

            TF zenith_angle = TF(0.)/TF(180.) * M_PI;//std::acos(mu0({1}));
            TF azimuth_angle = M_PI;//TF(3.4906585); //3.14; // sun approximately from south
            
            
            raytracer.trace_rays(
                    ray_count,
                    n_col_x, n_col_y, n_z, n_lay,
                    dx_grid, dy_grid, dz_grid,
                    z_lev,
                    dynamic_cast<Optical_props_2str_gpu<TF>&>(*optical_props),
                    dynamic_cast<Optical_props_2str_gpu<TF>&>(*cloud_optical_props),
                    sfc_alb_dir.subset({{ {band, band}, {1, n_col}}}),
                    zenith_angle, 
                    azimuth_angle,
                    toa_src,
                    toa_factor,
                    rayleigh,
                    col_dry,
                    gas_concs.get_vmr("h2o"),
                    flux_camera); // just a dummy name, this array will the the camera count!!!!
            //return; 
            raytracer.add_xyz_camera(
                    cam_nx, cam_ny,
                    xyz_factor_gpu,
                    flux_camera,
                    XYZ);
            
//            if (wv1_sub > 409 && wv1_sub < 410)
//            {
//                printf("%d %f %e\n",igpt,toa_src({1}) * toa_factor * mu0({1}),rayleigh);
//                flux_camera.dump(std::to_string(igpt));
//            }            
            
            //    (*fluxes).net_flux();

            //    gpoint_kernel_launcher_cuda::add_from_gpoint(
            //            n_col, n_lev, sw_flux_up, sw_flux_dn, sw_flux_dn_dir, sw_flux_net,
            //            (*fluxes).get_flux_up(), (*fluxes).get_flux_dn(), (*fluxes).get_flux_dn_dir(), (*fluxes).get_flux_net());
                
            // gpoint_kernel_launcher_cuda::add_from_gpoint(
            //         n_col_x, n_col_y, rt_flux_toa_up, rt_flux_sfc_dir, rt_flux_sfc_dif, rt_flux_sfc_up,
            //         (*fluxes).get_flux_toa_up(), (*fluxes).get_flux_sfc_dir(), (*fluxes).get_flux_sfc_dif(), (*fluxes).get_flux_sfc_up());

            // gpoint_kernel_launcher_cuda::add_from_gpoint(
            //         n_col, n_z, rt_flux_abs_dir, rt_flux_abs_dif,
            //         (*fluxes).get_flux_abs_dir(), (*fluxes).get_flux_abs_dif());

            //    if (switch_output_bnd_fluxes)
            //    {
            //        gpoint_kernel_launcher_cuda::get_from_gpoint(
            //                n_col, n_lev, igpt-1, sw_bnd_flux_up, sw_bnd_flux_dn, sw_bnd_flux_dn_dir, sw_bnd_flux_net,
            //                (*fluxes).get_flux_up(), (*fluxes).get_flux_dn(), (*fluxes).get_flux_dn_dir(), (*fluxes).get_flux_net());
            //    }
            
        }
    }
    //printf("%f %f \n", total_source,mu0({1}));
    //raytracer.normalize_xyz_camera(
     //       cam_nx, cam_ny, total_source, XYZ);
}

#ifdef RTE_RRTMGP_SINGLE_PRECISION
template class Radiation_solver_longwave<float>;
template class Radiation_solver_shortwave<float>;
#else
template class Radiation_solver_longwave<double>;
template class Radiation_solver_shortwave<double>;
#endif
