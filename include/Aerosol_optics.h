//
// Created by Mirjam Tijhuis on 05/08/2022.
//

#ifndef MICROHHC_AEROSOL_OPTICS_H
#define MICROHHC_AEROSOL_OPTICS_H
#include "Array.h"
#include "Optical_props.h"
#include "Types.h"


// Forward declarations.
class Optical_props;
class Optical_props_gpu;


class Aerosol_optics : public Optical_props
{
public:
    Aerosol_optics(
            const Array<Float,2>& band_lims_wvn, const std::vector<Float>& rh_upper,
            const Array<Float,2>& mext_phobic, const Array<Float,2>& ssa_phobic, const Array<Float,2>& g_phobic,
            const Array<Float,3>& mext_philic, const Array<Float,3>& ssa_philic, const Array<Float,3>& g_philic);

    void aerosol_optics(
            const Array<Float,2>& aermr01, const Array<Float,2>& aermr02, const Array<Float,2>& aermr03, const Array<Float,2>& aermr04,
            const Array<Float,2>& aermr05, const Array<Float,2>& aermr06, const Array<Float,2>& aermr07, const Array<Float,2>& aermr08,
            const Array<Float,2>& aermr09, const Array<Float,2>& aermr10, const Array<Float,2>& aermr11,
            const Array<Float, 2> &rh, const Array<Float, 2> &dpg,
            Optical_props_2str& optical_props);

private:
    // Lookup table coefficients.
    Array<Float,2> mext_phobic;
    Array<Float,2> ssa_phobic;
    Array<Float,2> g_phobic;

    Array<Float,3> mext_philic;
    Array<Float,3> ssa_philic;
    Array<Float,3> g_philic;

    std::vector<Float> rh_upper;

};

#endif //MICROHHC_AEROSOL_OPTICS_H