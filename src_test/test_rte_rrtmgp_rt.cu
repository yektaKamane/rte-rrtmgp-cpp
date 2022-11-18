/*
 * This file is a stand-alone executable developed for the
 * testing of the C++ interface to the RTE+RRTMGP radiation code.
 *
 * It is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This software is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this software.  If not, see <http://www.gnu.org/licenses/>.
 */

#include <boost/algorithm/string.hpp>
#include <chrono>
#include <iomanip>
#include <cuda_profiler_api.h>


#include "Status.h"
#include "Netcdf_interface.h"
#include "Array.h"
#include "raytracer_kernels.h"
#include "Radiation_solver_rt.h"
#include "Gas_concs.h"
#include "Types.h"
#include "Mem_pool_gpu.h"

void read_and_set_vmr(
        const std::string& gas_name, const int n_col_x, const int n_col_y, const int n_lay,
        const Netcdf_handle& input_nc, Gas_concs& gas_concs)
{
    const std::string vmr_gas_name = "vmr_" + gas_name;

    if (input_nc.variable_exists(vmr_gas_name))
    {
        std::map<std::string, int> dims = input_nc.get_variable_dimensions(vmr_gas_name);
        const int n_dims = dims.size();

        if (n_dims == 0)
        {
            gas_concs.set_vmr(gas_name, input_nc.get_variable<Float>(vmr_gas_name));
        }
        else if (n_dims == 1)
        {
            if (dims.at("lay") == n_lay)
                gas_concs.set_vmr(gas_name,
                        Array<Float,1>(input_nc.get_variable<Float>(vmr_gas_name, {n_lay}), {n_lay}));
            else
                throw std::runtime_error("Illegal dimensions of gas \"" + gas_name + "\" in input");
        }
        else if (n_dims == 3)
        {
            if (dims.at("lay") == n_lay && dims.at("y") == n_col_y && dims.at("x") == n_col_x)
                gas_concs.set_vmr(gas_name,
                        Array<Float,2>(input_nc.get_variable<Float>(vmr_gas_name, {n_lay, n_col_y, n_col_x}), {n_col_x * n_col_y, n_lay}));
            else
                throw std::runtime_error("Illegal dimensions of gas \"" + gas_name + "\" in input");
        }
    }
    else
    {
        Status::print_warning("Gas \"" + gas_name + "\" not available in input file.");
    }
}

void configure_memory_pool(int nlays, int ncols, int nchunks, int ngpts, int nbnds)
{
    /* Heuristic way to set up memory pool queues */
    std::map<std::size_t, std::size_t> pool_queues = {
        {64, 20},
        {128, 20},
        {256, 10},
        {512, 10},
        {1024, 5},
        {2048, 5},
        {nchunks * ngpts * sizeof(Float), 16},
        {nchunks * nbnds * sizeof(Float), 16},
        {(nlays + 1) * ncols * sizeof(Float), 14},
        {(nlays + 1) * nchunks * sizeof(Float), 10},
        {(nlays + 1) * nchunks * nbnds * sizeof(Float), 4},
        {(nlays + 1) * nchunks * ngpts * sizeof(int)/2, 6},
        {(nlays + 1) * nchunks * sizeof(Float), 18}
    };
    //outcomment, our own memory pool is not working yet for the RT version :(
    //#ifdef GPU_MEM_POOL
    //Memory_pool_gpu::init_instance(pool_queues);
    //#endif
}

bool parse_command_line_options(
        std::map<std::string, std::pair<bool, std::string>>& command_line_switches,
        std::map<std::string, std::pair<int, std::string>>& command_line_ints,
        int argc, char** argv)
{
    for (int i=1; i<argc; ++i)
    {
        std::string argument(argv[i]);
        boost::trim(argument);

        if (argument == "-h" || argument == "--help")
        {
            Status::print_message("Possible usage:");
            for (const auto& clo : command_line_switches)
            {
                std::ostringstream ss;
                ss << std::left << std::setw(30) << ("--" + clo.first);
                ss << clo.second.second << std::endl;
                Status::print_message(ss);
            }
            return true;
        }

        // Check if option starts with --
        if (argument[0] != '-' || argument[1] != '-')
        {
            std::string error = argument + " is an illegal command line option.";
            throw std::runtime_error(error);
        }
        else
            argument.erase(0, 2);

        // Check if option has prefix no-
        bool enable = true;
        if (argument[0] == 'n' && argument[1] == 'o' && argument[2] == '-')
        {
            enable = false;
            argument.erase(0, 3);
        }

        if (command_line_switches.find(argument) == command_line_switches.end())
        {
            std::string error = argument + " is an illegal command line option.";
            throw std::runtime_error(error);
        }
        else
        {
            command_line_switches.at(argument).first = enable;
        }

        // Check if a is integer is too be expect and if so, supplied
        if (command_line_ints.find(argument) != command_line_ints.end() && i+1 < argc)
        {
            std::string next_argument(argv[i+1]);
            boost::trim(next_argument);

            bool arg_is_int = true;
            for (int j=0; j<next_argument.size(); ++j)
                arg_is_int *= std::isdigit(next_argument[j]);

            if (arg_is_int)
            {
                command_line_ints.at(argument).first = std::stoi(argv[i+1]);
                ++i;
            }
        }
    }

    return false;
}

void print_command_line_options(
        const std::map<std::string, std::pair<bool, std::string>>& command_line_switches,
        const std::map<std::string, std::pair<int, std::string>>& command_line_ints)
{
    Status::print_message("Solver settings:");
    for (const auto& option : command_line_switches)
    {
        std::ostringstream ss;
        ss << std::left << std::setw(20) << (option.first);
        if (command_line_ints.find(option.first) != command_line_ints.end() && option.second.first)
            ss << " = " << std::boolalpha << command_line_ints.at(option.first).first << std::endl;
        else
            ss << " = " << std::boolalpha << option.second.first << std::endl;
        Status::print_message(ss);
   }
}


void solve_radiation(int argc, char** argv)
{
    Status::print_message("###### Starting RTE+RRTMGP solver ######");

    ////// FLOW CONTROL SWITCHES //////
    // Parse the command line options.
    std::map<std::string, std::pair<bool, std::string>> command_line_switches {
        {"shortwave"        , { true,  "Enable computation of shortwave radiation."}},
        {"longwave"         , { false, "Enable computation of longwave radiation." }},
        {"fluxes"           , { true,  "Enable computation of fluxes."             }},
        {"raytracing"       , { true,  "Use raytracing for flux computation. '--raytracing 256': use 256 rays per pixel" }},
        {"cloud-optics"     , { false, "Enable cloud optics."                      }},
        {"aerosol-optics"   , { false, "Enable aerosol optics."                    }},
        {"single-gpt"       , { false, "Output optical properties and fluxes for a single g-point. '--single-gpt 100': output 100th g-point" }},
        {"profiling"        , { false, "Perform additional profiling run."         }} };

    std::map<std::string, std::pair<int, std::string>> command_line_ints {
        {"raytracing", {32, "Number of rays initialised at TOD per pixel per quadraute."}},
        {"single-gpt", {1 , "g-point to store optical properties and fluxes of" }} };

    if (parse_command_line_options(command_line_switches, command_line_ints, argc, argv))
        return;

    const bool switch_shortwave         = command_line_switches.at("shortwave"        ).first;
    const bool switch_longwave          = command_line_switches.at("longwave"         ).first;
    const bool switch_fluxes            = command_line_switches.at("fluxes"           ).first;
    const bool switch_raytracing        = command_line_switches.at("raytracing"       ).first;
    const bool switch_cloud_optics      = command_line_switches.at("cloud-optics"     ).first;
    const bool switch_aerosol_optics    = command_line_switches.at("aerosol-optics"   ).first;
    const bool switch_single_gpt        = command_line_switches.at("single-gpt"       ).first;
    const bool switch_profiling         = command_line_switches.at("profiling"        ).first;

    // Print the options to the screen.
    print_command_line_options(command_line_switches, command_line_ints);

    Int photons_per_pixel = Int(command_line_ints.at("raytracing").first);
    if (Float(int(std::log2(Float(photons_per_pixel)))) != std::log2(Float(photons_per_pixel)))
    {
        std::string error = "number of photons per pixel should be a power of 2 ";
        throw std::runtime_error(error);
    }

    int single_gpt = command_line_ints.at("single-gpt").first;

    Status::print_message("Using "+ std::to_string(photons_per_pixel) + " rays per pixel");

    ////// READ THE ATMOSPHERIC DATA //////
    Status::print_message("Reading atmospheric input data from NetCDF.");

    Netcdf_file input_nc("rte_rrtmgp_input.nc", Netcdf_mode::Read);

    const int n_col_x = input_nc.get_dimension_size("x");
    const int n_col_y = input_nc.get_dimension_size("y");
    const int n_col = n_col_x * n_col_y;
    const int n_lay = input_nc.get_dimension_size("lay");
    const int n_lev = input_nc.get_dimension_size("lev");
    const int n_z = input_nc.get_dimension_size("z");

    Array<Float,1> grid_x(input_nc.get_variable<Float>("x", {n_col_x}), {n_col_x});
    Array<Float,1> grid_y(input_nc.get_variable<Float>("y", {n_col_y}), {n_col_y});
    Array<Float,1> grid_z(input_nc.get_variable<Float>("z", {n_z}), {n_z});

    const Vector<int> grid_cells = {n_col_x, n_col_y, n_z};
    const Vector<Float> grid_d = {grid_x({2}) - grid_x({1}), grid_y({2}) - grid_y({1}), grid_z({2}) - grid_z({1})};
    const Vector<int> kn_grid = {input_nc.get_variable<int>("ngrid_x"),
                                 input_nc.get_variable<int>("ngrid_y"),
                                 input_nc.get_variable<int>("ngrid_z")};

    // Read the atmospheric fields.
    Array<Float,2> p_lay(input_nc.get_variable<Float>("p_lay", {n_lay, n_col_y, n_col_x}), {n_col, n_lay});
    Array<Float,2> t_lay(input_nc.get_variable<Float>("t_lay", {n_lay, n_col_y, n_col_x}), {n_col, n_lay});
    Array<Float,2> p_lev(input_nc.get_variable<Float>("p_lev", {n_lev, n_col_y, n_col_x}), {n_col, n_lev});
    Array<Float,2> t_lev(input_nc.get_variable<Float>("t_lev", {n_lev, n_col_y, n_col_x}), {n_col, n_lev});

    // Fetch the col_dry in case present.
    Array<Float,2> col_dry;
    if (input_nc.variable_exists("col_dry"))
    {
        col_dry.set_dims({n_col, n_lay});
        col_dry = std::move(input_nc.get_variable<Float>("col_dry", {n_lay, n_col_y, n_col_x}));
    }

    // Create container for the gas concentrations and read gases.
    Gas_concs gas_concs;

    read_and_set_vmr("h2o", n_col_x, n_col_y, n_lay, input_nc, gas_concs);
    read_and_set_vmr("co2", n_col_x, n_col_y, n_lay, input_nc, gas_concs);
    read_and_set_vmr("o3" , n_col_x, n_col_y, n_lay, input_nc, gas_concs);
    read_and_set_vmr("n2o", n_col_x, n_col_y, n_lay, input_nc, gas_concs);
    read_and_set_vmr("co" , n_col_x, n_col_y, n_lay, input_nc, gas_concs);
    read_and_set_vmr("ch4", n_col_x, n_col_y, n_lay, input_nc, gas_concs);
    read_and_set_vmr("o2" , n_col_x, n_col_y, n_lay, input_nc, gas_concs);
    read_and_set_vmr("n2" , n_col_x, n_col_y, n_lay, input_nc, gas_concs);

    read_and_set_vmr("ccl4"   , n_col_x, n_col_y, n_lay, input_nc, gas_concs);
    read_and_set_vmr("cfc11"  , n_col_x, n_col_y, n_lay, input_nc, gas_concs);
    read_and_set_vmr("cfc12"  , n_col_x, n_col_y, n_lay, input_nc, gas_concs);
    read_and_set_vmr("cfc22"  , n_col_x, n_col_y, n_lay, input_nc, gas_concs);
    read_and_set_vmr("hfc143a", n_col_x, n_col_y, n_lay, input_nc, gas_concs);
    read_and_set_vmr("hfc125" , n_col_x, n_col_y, n_lay, input_nc, gas_concs);
    read_and_set_vmr("hfc23"  , n_col_x, n_col_y, n_lay, input_nc, gas_concs);
    read_and_set_vmr("hfc32"  , n_col_x, n_col_y, n_lay, input_nc, gas_concs);
    read_and_set_vmr("hfc134a", n_col_x, n_col_y, n_lay, input_nc, gas_concs);
    read_and_set_vmr("cf4"    , n_col_x, n_col_y, n_lay, input_nc, gas_concs);
    read_and_set_vmr("no2"    , n_col_x, n_col_y, n_lay, input_nc, gas_concs);

    Array<Float,2> lwp;
    Array<Float,2> iwp;
    Array<Float,2> rel;
    Array<Float,2> rei;

    if (switch_cloud_optics)
    {
        lwp.set_dims({n_col, n_lay});
        lwp = std::move(input_nc.get_variable<Float>("lwp", {n_lay, n_col_y, n_col_x}));

        iwp.set_dims({n_col, n_lay});
        iwp = std::move(input_nc.get_variable<Float>("iwp", {n_lay, n_col_y, n_col_x}));

        rel.set_dims({n_col, n_lay});
        rel = std::move(input_nc.get_variable<Float>("rel", {n_lay, n_col_y, n_col_x}));

        rei.set_dims({n_col, n_lay});
        rei = std::move(input_nc.get_variable<Float>("rei", {n_lay, n_col_y, n_col_x}));
    }

    Array<Float,2> rh;
    Array<Float,1> aermr01;
    Array<Float,1> aermr02;
    Array<Float,1> aermr03;
    Array<Float,1> aermr04;
    Array<Float,1> aermr05;
    Array<Float,1> aermr06;
    Array<Float,1> aermr07;
    Array<Float,1> aermr08;
    Array<Float,1> aermr09;
    Array<Float,1> aermr10;
    Array<Float,1> aermr11;

    if (switch_aerosol_optics)
    {
        rh.set_dims({n_col, n_lay});
        rh = std::move(input_nc.get_variable<Float>("rh", {n_lay, n_col_y, n_col_x}));

        aermr01.set_dims({n_lay});
        aermr01 = std::move(input_nc.get_variable<Float>("aermr01", {n_lay}));

        aermr02.set_dims({n_lay});
        aermr02 = std::move(input_nc.get_variable<Float>("aermr02", {n_lay}));

        aermr03.set_dims({n_lay});
        aermr03 = std::move(input_nc.get_variable<Float>("aermr03", {n_lay}));

        aermr04.set_dims({n_lay});
        aermr04 = std::move(input_nc.get_variable<Float>("aermr04", {n_lay}));

        aermr05.set_dims({n_lay});
        aermr05 = std::move(input_nc.get_variable<Float>("aermr05", {n_lay}));

        aermr06.set_dims({n_lay});
        aermr06 = std::move(input_nc.get_variable<Float>("aermr06", {n_lay}));

        aermr07.set_dims({n_lay});
        aermr07 = std::move(input_nc.get_variable<Float>("aermr07", {n_lay}));

        aermr08.set_dims({n_lay});
        aermr08 = std::move(input_nc.get_variable<Float>("aermr08", {n_lay}));

        aermr09.set_dims({n_lay});
        aermr09 = std::move(input_nc.get_variable<Float>("aermr09", {n_lay}));

        aermr10.set_dims({n_lay});
        aermr10 = std::move(input_nc.get_variable<Float>("aermr10", {n_lay}));

        aermr11.set_dims({n_lay});
        aermr11 = std::move(input_nc.get_variable<Float>("aermr11", {n_lay}));
    }


    ////// CREATE THE OUTPUT FILE //////
    // Create the general dimensions and arrays.
    Status::print_message("Preparing NetCDF output file.");

    Netcdf_file output_nc("rte_rrtmgp_output.nc", Netcdf_mode::Create);
    output_nc.add_dimension("col", n_col);
    output_nc.add_dimension("x", n_col_x);
    output_nc.add_dimension("y", n_col_y);
    output_nc.add_dimension("lay", n_lay);
    output_nc.add_dimension("lev", n_lev);
    output_nc.add_dimension("pair", 2);

    output_nc.add_dimension("z", n_z);
    auto nc_x = output_nc.add_variable<Float>("x", {"x"});
    auto nc_y = output_nc.add_variable<Float>("y", {"y"});
    auto nc_z = output_nc.add_variable<Float>("z", {"z"});
    nc_x.insert(grid_x.v(), {0});
    nc_y.insert(grid_y.v(), {0});
    nc_z.insert(grid_z.v(), {0});

    auto nc_lay = output_nc.add_variable<Float>("p_lay", {"lay", "col"});
    auto nc_lev = output_nc.add_variable<Float>("p_lev", {"lev", "col"});

    nc_lay.insert(p_lay.v(), {0, 0});
    nc_lev.insert(p_lev.v(), {0, 0});

    int ngpts = 0;
    int nbnds = 0;
    if (switch_longwave)
    {
        Netcdf_file coef_nc_lw("coefficients_lw.nc", Netcdf_mode::Read);
        nbnds = std::max(coef_nc_lw.get_dimension_size("bnd"), nbnds);
        ngpts = std::max(coef_nc_lw.get_dimension_size("gpt"), ngpts);
    }
    if (switch_shortwave)
    {
        Netcdf_file coef_nc_sw("coefficients_sw.nc", Netcdf_mode::Read);
        nbnds = std::max(coef_nc_sw.get_dimension_size("bnd"), nbnds);
        ngpts = std::max(coef_nc_sw.get_dimension_size("gpt"), ngpts);
    }
    configure_memory_pool(n_lay, n_col, 1024, ngpts, nbnds);


    ////// RUN THE LONGWAVE SOLVER //////
    if (switch_longwave)
    {
        // Initialize the solver.
        Status::print_message("Initializing the longwave solver.");

        Gas_concs_gpu gas_concs_gpu(gas_concs);

        Radiation_solver_longwave rad_lw(gas_concs_gpu, "coefficients_lw.nc", "cloud_coefficients_lw.nc");

        // Read the boundary conditions.
        const int n_bnd_lw = rad_lw.get_n_bnd_gpu();
        const int n_gpt_lw = rad_lw.get_n_gpt_gpu();

        Array<Float,2> emis_sfc(input_nc.get_variable<Float>("emis_sfc", {n_col_y, n_col_x, n_bnd_lw}), {n_bnd_lw, n_col});
        Array<Float,1> t_sfc(input_nc.get_variable<Float>("t_sfc", {n_col_y, n_col_x}), {n_col});

        // Create output arrays.
        Array_gpu<Float,2> lw_tau;
        Array_gpu<Float,2> lay_source;
        Array_gpu<Float,2> lev_source_inc;
        Array_gpu<Float,2> lev_source_dec;
        Array_gpu<Float,1> sfc_source;

        if (switch_single_gpt)
        {
            lw_tau        .set_dims({n_col, n_lay});
            lay_source    .set_dims({n_col, n_lay});
            lev_source_inc.set_dims({n_col, n_lay});
            lev_source_dec.set_dims({n_col, n_lay});
            sfc_source    .set_dims({n_col});
        }

        Array_gpu<Float,2> lw_flux_up;
        Array_gpu<Float,2> lw_flux_dn;
        Array_gpu<Float,2> lw_flux_net;

        if (switch_fluxes)
        {
            lw_flux_up .set_dims({n_col, n_lev});
            lw_flux_dn .set_dims({n_col, n_lev});
            lw_flux_net.set_dims({n_col, n_lev});
        }

        Array_gpu<Float,2> lw_gpt_flux_up;
        Array_gpu<Float,2> lw_gpt_flux_dn;
        Array_gpu<Float,2> lw_gpt_flux_net;

        if (switch_single_gpt)
        {
            lw_gpt_flux_up .set_dims({n_col, n_lev});
            lw_gpt_flux_dn .set_dims({n_col, n_lev});
            lw_gpt_flux_net.set_dims({n_col, n_lev});
        }


        // Solve the radiation.

        Status::print_message("Solving the longwave radiation.");

        auto run_solver = [&]()
        {
            Array_gpu<Float,2> p_lay_gpu(p_lay);
            Array_gpu<Float,2> p_lev_gpu(p_lev);
            Array_gpu<Float,2> t_lay_gpu(t_lay);
            Array_gpu<Float,2> t_lev_gpu(t_lev);
            Array_gpu<Float,2> col_dry_gpu(col_dry);
            Array_gpu<Float,1> t_sfc_gpu(t_sfc);
            Array_gpu<Float,2> emis_sfc_gpu(emis_sfc);
            Array_gpu<Float,2> lwp_gpu(lwp);
            Array_gpu<Float,2> iwp_gpu(iwp);
            Array_gpu<Float,2> rel_gpu(rel);
            Array_gpu<Float,2> rei_gpu(rei);


            cudaDeviceSynchronize();
            cudaEvent_t start;
            cudaEvent_t stop;
            cudaEventCreate(&start);
            cudaEventCreate(&stop);

            cudaEventRecord(start, 0);

            rad_lw.solve_gpu(
                    switch_fluxes,
                    switch_cloud_optics,
                    switch_single_gpt,
                    single_gpt,
                    gas_concs_gpu,
                    p_lay_gpu, p_lev_gpu,
                    t_lay_gpu, t_lev_gpu,
                    col_dry_gpu,
                    t_sfc_gpu, emis_sfc_gpu,
                    lwp_gpu, iwp_gpu,
                    rel_gpu, rei_gpu,
                    lw_tau, lay_source, lev_source_inc, lev_source_dec, sfc_source,
                    lw_flux_up, lw_flux_dn, lw_flux_net,
                    lw_gpt_flux_up, lw_gpt_flux_dn, lw_gpt_flux_net);

            cudaEventRecord(stop, 0);
            cudaEventSynchronize(stop);
            float duration = 0.f;
            cudaEventElapsedTime(&duration, start, stop);

            cudaEventDestroy(start);
            cudaEventDestroy(stop);

            Status::print_message("Duration longwave solver: " + std::to_string(duration) + " (ms)");
        };

        // Tuning step;
        run_solver();

        // Profiling step;
        cudaProfilerStart();
        run_solver();
        cudaProfilerStop();

        constexpr int n_measures=10;
        for (int n=0; n<n_measures; ++n)
            run_solver();


        //// Store the output.
        Status::print_message("Storing the longwave output.");
        Array<Float,2> lw_tau_cpu(lw_tau);
        Array<Float,2> lay_source_cpu(lay_source);
        Array<Float,1> sfc_source_cpu(sfc_source);
        Array<Float,2> lev_source_inc_cpu(lev_source_inc);
        Array<Float,2> lev_source_dec_cpu(lev_source_dec);
        Array<Float,2> lw_flux_up_cpu(lw_flux_up);
        Array<Float,2> lw_flux_dn_cpu(lw_flux_dn);
        Array<Float,2> lw_flux_net_cpu(lw_flux_net);
        Array<Float,2> lw_gpt_flux_up_cpu(lw_gpt_flux_up);
        Array<Float,2> lw_gpt_flux_dn_cpu(lw_gpt_flux_dn);
        Array<Float,2> lw_gpt_flux_net_cpu(lw_gpt_flux_net);

        output_nc.add_dimension("gpt_lw", n_gpt_lw);
        output_nc.add_dimension("band_lw", n_bnd_lw);

        auto nc_lw_band_lims_wvn = output_nc.add_variable<Float>("lw_band_lims_wvn", {"band_lw", "pair"});
        nc_lw_band_lims_wvn.insert(rad_lw.get_band_lims_wavenumber_gpu().v(), {0, 0});

        if (switch_single_gpt)
        {
            auto nc_lw_band_lims_gpt = output_nc.add_variable<int>("lw_band_lims_gpt", {"band_lw", "pair"});
            nc_lw_band_lims_gpt.insert(rad_lw.get_band_lims_gpoint_gpu().v(), {0, 0});

            auto nc_lw_tau = output_nc.add_variable<Float>("lw_tau", {"lay", "y", "x"});
            nc_lw_tau.insert(lw_tau_cpu.v(), {0, 0, 0});

            auto nc_lay_source     = output_nc.add_variable<Float>("lay_source"    , {"lay", "y", "x"});
            auto nc_lev_source_inc = output_nc.add_variable<Float>("lev_source_inc", {"lay", "y", "x"});
            auto nc_lev_source_dec = output_nc.add_variable<Float>("lev_source_dec", {"lay", "y", "x"});

            auto nc_sfc_source = output_nc.add_variable<Float>("sfc_source", {"y", "x"});

            nc_lay_source.insert    (lay_source_cpu.v()    , {0, 0, 0});
            nc_lev_source_inc.insert(lev_source_inc_cpu.v(), {0, 0, 0});
            nc_lev_source_dec.insert(lev_source_dec_cpu.v(), {0, 0, 0});

            nc_sfc_source.insert(sfc_source_cpu.v(), {0, 0});
        }

        if (switch_fluxes)
        {
            auto nc_lw_flux_up  = output_nc.add_variable<Float>("lw_flux_up" , {"lev", "y", "x"});
            auto nc_lw_flux_dn  = output_nc.add_variable<Float>("lw_flux_dn" , {"lev", "y", "x"});
            auto nc_lw_flux_net = output_nc.add_variable<Float>("lw_flux_net", {"lev", "y", "x"});

            nc_lw_flux_up .insert(lw_flux_up_cpu .v(), {0, 0, 0});
            nc_lw_flux_dn .insert(lw_flux_dn_cpu .v(), {0, 0, 0});
            nc_lw_flux_net.insert(lw_flux_net_cpu.v(), {0, 0, 0});

            if (switch_single_gpt)
            {
                auto nc_lw_gpt_flux_up  = output_nc.add_variable<Float>("lw_gpt_flux_up" , {"lev", "y", "x"});
                auto nc_lw_gpt_flux_dn  = output_nc.add_variable<Float>("lw_gpt_flux_dn" , {"lev", "y", "x"});
                auto nc_lw_gpt_flux_net = output_nc.add_variable<Float>("lw_gpt_flux_net", {"lev", "y", "x"});

                nc_lw_gpt_flux_up .insert(lw_gpt_flux_up_cpu.v(), {0, 0, 0});
                nc_lw_gpt_flux_dn .insert(lw_gpt_flux_dn_cpu.v(), {0, 0, 0});
                nc_lw_gpt_flux_net.insert(lw_gpt_flux_net_cpu.v(), {0, 0, 0});
            }
        }
    }


    ////// RUN THE SHORTWAVE SOLVER //////
    if (switch_shortwave)
    {
        // Initialize the solver.
        Status::print_message("Initializing the shortwave solver.");


        Gas_concs_gpu gas_concs_gpu(gas_concs);
        Radiation_solver_shortwave rad_sw(gas_concs_gpu, "coefficients_sw.nc", "cloud_coefficients_sw.nc", "aerosol_optics.nc");

        // Read the boundary conditions.
        const int n_bnd_sw = rad_sw.get_n_bnd_gpu();
        const int n_gpt_sw = rad_sw.get_n_gpt_gpu();

        Array<Float,1> mu0(input_nc.get_variable<Float>("mu0", {n_col_y, n_col_x}), {n_col});
        Array<Float,1> azi(input_nc.get_variable<Float>("azi", {n_col_y, n_col_x}), {n_col});
        Array<Float,2> sfc_alb_dir(input_nc.get_variable<Float>("sfc_alb_dir", {n_col_y, n_col_x, n_bnd_sw}), {n_bnd_sw, n_col});
        Array<Float,2> sfc_alb_dif(input_nc.get_variable<Float>("sfc_alb_dif", {n_col_y, n_col_x, n_bnd_sw}), {n_bnd_sw, n_col});

        Array<Float,1> tsi_scaling({n_col});
        if (input_nc.variable_exists("tsi"))
        {
            Array<Float,1> tsi(input_nc.get_variable<Float>("tsi", {n_col_y, n_col_x}), {n_col});
            const Float tsi_ref = rad_sw.get_tsi_gpu();
            for (int icol=1; icol<=n_col; ++icol)
                tsi_scaling({icol}) = tsi({icol}) / tsi_ref;
        }
        else if (input_nc.variable_exists("tsi_scaling"))
        {
            Float tsi_scaling_in = input_nc.get_variable<Float>("tsi_scaling");
            for (int icol=1; icol<=n_col; ++icol)
                tsi_scaling({icol}) = tsi_scaling_in;
        }
        else
        {
            for (int icol=1; icol<=n_col; ++icol)
                tsi_scaling({icol}) = Float(1.);
        }

        // Optional top of domain fluxes
        // Create output arrays.
        Array_gpu<Float,2> sw_tau;
        Array_gpu<Float,2> ssa;
        Array_gpu<Float,2> g;

        if (switch_single_gpt)
        {
            sw_tau    .set_dims({n_col, n_lay});
            ssa       .set_dims({n_col, n_lay});
            g         .set_dims({n_col, n_lay});
        }

        Array_gpu<Float,2> sw_flux_up;
        Array_gpu<Float,2> sw_flux_dn;
        Array_gpu<Float,2> sw_flux_dn_dir;
        Array_gpu<Float,2> sw_flux_net;

        Array_gpu<Float,2> rt_flux_tod_up;
        Array_gpu<Float,2> rt_flux_sfc_dir;
        Array_gpu<Float,2> rt_flux_sfc_dif;
        Array_gpu<Float,2> rt_flux_sfc_up;
        Array_gpu<Float,3> rt_flux_abs_dir;
        Array_gpu<Float,3> rt_flux_abs_dif;


        if (switch_fluxes)
        {
            sw_flux_up    .set_dims({n_col, n_lev});
            sw_flux_dn    .set_dims({n_col, n_lev});
            sw_flux_dn_dir.set_dims({n_col, n_lev});
            sw_flux_net   .set_dims({n_col, n_lev});
            if (switch_raytracing)
            {
                rt_flux_tod_up .set_dims({n_col_x, n_col_y});
                rt_flux_sfc_dir.set_dims({n_col_x, n_col_y});
                rt_flux_sfc_dif.set_dims({n_col_x, n_col_y});
                rt_flux_sfc_up .set_dims({n_col_x, n_col_y});
                rt_flux_abs_dir.set_dims({n_col_x, n_col_y, n_z});
                rt_flux_abs_dif.set_dims({n_col_x, n_col_y, n_z});
            }

        }

        Array_gpu<Float,2> sw_gpt_flux_up;
        Array_gpu<Float,2> sw_gpt_flux_dn;
        Array_gpu<Float,2> sw_gpt_flux_dn_dir;
        Array_gpu<Float,2> sw_gpt_flux_net;

        if (switch_single_gpt)
        {
            sw_gpt_flux_up    .set_dims({n_col, n_lev});
            sw_gpt_flux_dn    .set_dims({n_col, n_lev});
            sw_gpt_flux_dn_dir.set_dims({n_col, n_lev});
            sw_gpt_flux_net   .set_dims({n_col, n_lev});
        }


        // Solve the radiation.
        Status::print_message("Solving the shortwave radiation.");

        auto run_solver = [&]()
        {
            Array_gpu<Float,2> p_lay_gpu(p_lay);
            Array_gpu<Float,2> p_lev_gpu(p_lev);
            Array_gpu<Float,2> t_lay_gpu(t_lay);
            Array_gpu<Float,2> t_lev_gpu(t_lev);
            Array_gpu<Float,2> col_dry_gpu(col_dry);
            Array_gpu<Float,2> sfc_alb_dir_gpu(sfc_alb_dir);
            Array_gpu<Float,2> sfc_alb_dif_gpu(sfc_alb_dif);
            Array_gpu<Float,1> tsi_scaling_gpu(tsi_scaling);
            Array_gpu<Float,1> mu0_gpu(mu0);
            Array_gpu<Float,1> azi_gpu(azi);
            Array_gpu<Float,2> lwp_gpu(lwp);
            Array_gpu<Float,2> iwp_gpu(iwp);
            Array_gpu<Float,2> rel_gpu(rel);
            Array_gpu<Float,2> rei_gpu(rei);

            Array_gpu<Float,2> rh_gpu(rh);
            Array_gpu<Float,1> aermr01_gpu(aermr01);
            Array_gpu<Float,1> aermr02_gpu(aermr02);
            Array_gpu<Float,1> aermr03_gpu(aermr03);
            Array_gpu<Float,1> aermr04_gpu(aermr04);
            Array_gpu<Float,1> aermr05_gpu(aermr05);
            Array_gpu<Float,1> aermr06_gpu(aermr06);
            Array_gpu<Float,1> aermr07_gpu(aermr07);
            Array_gpu<Float,1> aermr08_gpu(aermr08);
            Array_gpu<Float,1> aermr09_gpu(aermr09);
            Array_gpu<Float,1> aermr10_gpu(aermr10);
            Array_gpu<Float,1> aermr11_gpu(aermr11);

            cudaDeviceSynchronize();
            cudaEvent_t start;
            cudaEvent_t stop;
            cudaEventCreate(&start);
            cudaEventCreate(&stop);

            cudaEventRecord(start, 0);

            rad_sw.solve_gpu(
                    switch_fluxes,
                    switch_raytracing,
                    switch_cloud_optics,
                    switch_aerosol_optics,
                    switch_single_gpt,
                    single_gpt,
                    photons_per_pixel,
                    grid_cells,
                    grid_d,
                    kn_grid,
                    gas_concs_gpu,
                    p_lay_gpu, p_lev_gpu,
                    t_lay_gpu, t_lev_gpu,
                    col_dry_gpu,
                    sfc_alb_dir_gpu, sfc_alb_dif_gpu,
                    tsi_scaling_gpu, mu0_gpu, azi_gpu,
                    lwp_gpu, iwp_gpu,
                    rel_gpu, rei_gpu,
                    rh,
                    aermr01, aermr02, aermr03, aermr04, aermr05,
                    aermr06, aermr07, aermr08, aermr09, aermr10, aermr11,
                    sw_tau, ssa, g,
                    sw_flux_up, sw_flux_dn,
                    sw_flux_dn_dir, sw_flux_net,
                    sw_gpt_flux_up, sw_gpt_flux_dn,
                    sw_gpt_flux_dn_dir, sw_gpt_flux_net,
                    rt_flux_tod_up,
                    rt_flux_sfc_dir,
                    rt_flux_sfc_dif,
                    rt_flux_sfc_up,
                    rt_flux_abs_dir,
                    rt_flux_abs_dif);

            cudaEventRecord(stop, 0);
            cudaEventSynchronize(stop);
            float duration = 0.f;
            cudaEventElapsedTime(&duration, start, stop);

            cudaEventDestroy(start);
            cudaEventDestroy(stop);

            Status::print_message("Duration shortwave solver: " + std::to_string(duration) + " (ms)");
        };

        // Tuning step;
        run_solver();

        // Profiling step;
        if (switch_profiling)
        {
            cudaProfilerStart();
            run_solver();
            cudaProfilerStop();
        }

        // Store the output.
        Status::print_message("Storing the shortwave output.");
        Array<Float,2> sw_tau_cpu(sw_tau);
        Array<Float,2> ssa_cpu(ssa);
        Array<Float,2> g_cpu(g);
        Array<Float,2> sw_flux_up_cpu(sw_flux_up);
        Array<Float,2> sw_flux_dn_cpu(sw_flux_dn);
        Array<Float,2> sw_flux_dn_dir_cpu(sw_flux_dn_dir);
        Array<Float,2> sw_flux_net_cpu(sw_flux_net);
        Array<Float,2> sw_gpt_flux_up_cpu(sw_gpt_flux_up);
        Array<Float,2> sw_gpt_flux_dn_cpu(sw_gpt_flux_dn);
        Array<Float,2> sw_gpt_flux_dn_dir_cpu(sw_gpt_flux_dn_dir);
        Array<Float,2> sw_gpt_flux_net_cpu(sw_gpt_flux_net);

        Array<Float,2> rt_flux_tod_up_cpu(rt_flux_tod_up);
        Array<Float,2> rt_flux_sfc_dir_cpu(rt_flux_sfc_dir);
        Array<Float,2> rt_flux_sfc_dif_cpu(rt_flux_sfc_dif);
        Array<Float,2> rt_flux_sfc_up_cpu(rt_flux_sfc_up);
        Array<Float,3> rt_flux_abs_dir_cpu(rt_flux_abs_dir);
        Array<Float,3> rt_flux_abs_dif_cpu(rt_flux_abs_dif);

        output_nc.add_dimension("gpt_sw", n_gpt_sw);
        output_nc.add_dimension("band_sw", n_bnd_sw);

        auto nc_sw_band_lims_wvn = output_nc.add_variable<Float>("sw_band_lims_wvn", {"band_sw", "pair"});
        nc_sw_band_lims_wvn.insert(rad_sw.get_band_lims_wavenumber_gpu().v(), {0, 0});

        if (switch_single_gpt)
        {
            auto nc_sw_band_lims_gpt = output_nc.add_variable<int>("sw_band_lims_gpt", {"band_sw", "pair"});
            nc_sw_band_lims_gpt.insert(rad_sw.get_band_lims_gpoint_gpu().v(), {0, 0});

            auto nc_sw_tau = output_nc.add_variable<Float>("sw_tau"   , {"lay", "y", "x"});
            auto nc_ssa    = output_nc.add_variable<Float>("sw_ssa"   , {"lay", "y", "x"});
            auto nc_g      = output_nc.add_variable<Float>("sw_g"     , {"lay", "y", "x"});

            nc_sw_tau.insert(sw_tau_cpu.v(), {0, 0, 0, 0});
            nc_ssa   .insert(ssa_cpu   .v(), {0, 0, 0, 0});
            nc_g     .insert(g_cpu     .v(), {0, 0, 0, 0});
        }

        if (switch_fluxes)
        {
            auto nc_sw_flux_up     = output_nc.add_variable<Float>("sw_flux_up"    , {"lev", "y", "x"});
            auto nc_sw_flux_dn     = output_nc.add_variable<Float>("sw_flux_dn"    , {"lev", "y", "x"});
            auto nc_sw_flux_dn_dir = output_nc.add_variable<Float>("sw_flux_dn_dir", {"lev", "y", "x"});
            auto nc_sw_flux_net    = output_nc.add_variable<Float>("sw_flux_net"   , {"lev", "y", "x"});

            nc_sw_flux_up    .insert(sw_flux_up_cpu    .v(), {0, 0, 0});
            nc_sw_flux_dn    .insert(sw_flux_dn_cpu    .v(), {0, 0, 0});
            nc_sw_flux_dn_dir.insert(sw_flux_dn_dir_cpu.v(), {0, 0, 0});
            nc_sw_flux_net   .insert(sw_flux_net_cpu   .v(), {0, 0, 0});

            if (switch_raytracing)
            {
                auto nc_rt_flux_tod_up  = output_nc.add_variable<Float>("rt_flux_tod_up",  {"y","x"});
                auto nc_rt_flux_sfc_dir = output_nc.add_variable<Float>("rt_flux_sfc_dir", {"y","x"});
                auto nc_rt_flux_sfc_dif = output_nc.add_variable<Float>("rt_flux_sfc_dif", {"y","x"});
                auto nc_rt_flux_sfc_up  = output_nc.add_variable<Float>("rt_flux_sfc_up",  {"y","x"});
                auto nc_rt_flux_abs_dir = output_nc.add_variable<Float>("rt_flux_abs_dir", {"z","y","x"});
                auto nc_rt_flux_abs_dif = output_nc.add_variable<Float>("rt_flux_abs_dif", {"z","y","x"});

                nc_rt_flux_tod_up .insert(rt_flux_tod_up_cpu .v(), {0,0});
                nc_rt_flux_sfc_dir.insert(rt_flux_sfc_dir_cpu.v(), {0,0});
                nc_rt_flux_sfc_dif.insert(rt_flux_sfc_dif_cpu.v(), {0,0});
                nc_rt_flux_sfc_up .insert(rt_flux_sfc_up_cpu .v(), {0,0});
                nc_rt_flux_abs_dir.insert(rt_flux_abs_dir_cpu.v(), {0,0,0});
                nc_rt_flux_abs_dif.insert(rt_flux_abs_dif_cpu.v(), {0,0,0});
            }


            if (switch_single_gpt)
            {
                auto nc_sw_gpt_flux_up     = output_nc.add_variable<Float>("sw_gpt_flux_up"    , {"lev", "y", "x"});
                auto nc_sw_gpt_flux_dn     = output_nc.add_variable<Float>("sw_gpt_flux_dn"    , {"lev", "y", "x"});
                auto nc_sw_gpt_flux_dn_dir = output_nc.add_variable<Float>("sw_gpt_flux_dn_dir", {"lev", "y", "x"});
                auto nc_sw_gpt_flux_net    = output_nc.add_variable<Float>("sw_gpt_flux_net"   , {"lev", "y", "x"});

                nc_sw_gpt_flux_up    .insert(sw_gpt_flux_up_cpu    .v(), {0, 0, 0});
                nc_sw_gpt_flux_dn    .insert(sw_gpt_flux_dn_cpu    .v(), {0, 0, 0});
                nc_sw_gpt_flux_dn_dir.insert(sw_gpt_flux_dn_dir_cpu.v(), {0, 0, 0});
                nc_sw_gpt_flux_net   .insert(sw_gpt_flux_net_cpu   .v(), {0, 0, 0});
            }
        }
    }

    Status::print_message("###### Finished RTE+RRTMGP solver ######");
}


int main(int argc, char** argv)
{
    try
    {
        solve_radiation(argc, argv);
    }

    // Catch any exceptions and return 1.
    catch (const std::exception& e)
    {
        std::string error = "EXCEPTION: " + std::string(e.what());
        Status::print_message(error);
        return 1;
    }
    catch (...)
    {
        Status::print_message("UNHANDLED EXCEPTION!");
        return 1;
    }

    // Return 0 in case of normal exit.
    return 0;
}
