template<typename TF>__global__
void get_from_gpoint_kernel(const int ncol, const int igpt, TF* __restrict__ var_full, const TF* __restrict__ var_sub)
{
    const int icol = blockIdx.x*blockDim.x + threadIdx.x;

    if ( (icol < ncol) )
    {
        const int idx_full = icol + igpt*ncol;
        const int idx_sub = icol;
        var_full[idx_full] = var_sub[idx_sub];
    }
}

template<typename TF>__global__
void add_from_gpoint_kernel(const int ncol, const int nlay,
              TF* __restrict__ var1_full, TF* __restrict__ var2_full, TF* __restrict__ var3_full,  TF* __restrict__ var4_full,
              const TF* __restrict__ var1_sub, const TF* __restrict__ var2_sub, const TF* __restrict__ var3_sub, const TF* __restrict__ var4_sub)
{
    const int icol = blockIdx.x*blockDim.x + threadIdx.x;
    const int ilay = blockIdx.y*blockDim.y + threadIdx.y;

    if ( (icol < ncol) && (ilay < nlay) )
    {
        const int idx = icol + ilay*ncol;
        var1_full[idx] += var1_sub[idx];
        var2_full[idx] += var2_sub[idx];
        var3_full[idx] += var3_sub[idx];
        var4_full[idx] += var4_sub[idx];
    }
}

template<typename TF>__global__
void add_from_gpoint_kernel(const int ncol, const int nlay,
              TF* __restrict__ var1_full, TF* __restrict__ var2_full,
              const TF* __restrict__ var1_sub, const TF* __restrict__ var2_sub)
{
    const int icol = blockIdx.x*blockDim.x + threadIdx.x;
    const int ilay = blockIdx.y*blockDim.y + threadIdx.y;

    if ( (icol < ncol) && (ilay < nlay) )
    {
        const int idx = icol + ilay*ncol;
        var1_full[idx] += var1_sub[idx];
        var2_full[idx] += var2_sub[idx];
    }
}

template<typename TF>__global__
void add_from_gpoint_kernel(const int ncol, const int nlay,
              TF* __restrict__ var1_full, TF* __restrict__ var2_full, TF* __restrict__ var3_full,
              const TF* __restrict__ var1_sub, const TF* __restrict__ var2_sub, const TF* __restrict__ var3_sub)
{
    const int icol = blockIdx.x*blockDim.x + threadIdx.x;
    const int ilay = blockIdx.y*blockDim.y + threadIdx.y;

    if ( (icol < ncol) && (ilay < nlay) )
    {
        const int idx = icol + ilay*ncol;
        var1_full[idx] += var1_sub[idx];
        var2_full[idx] += var2_sub[idx];
        var3_full[idx] += var3_sub[idx];
    }
}

template<typename TF>__global__
void get_from_gpoint_kernel(const int ncol, const int nlay, const int igpt,
              TF* __restrict__ var1_full, TF* __restrict__ var2_full, TF* __restrict__ var3_full,  TF* __restrict__ var4_full,
              const TF* __restrict__ var1_sub, const TF* __restrict__ var2_sub, const TF* __restrict__ var3_sub, const TF* __restrict__ var4_sub)
{
    const int icol = blockIdx.x*blockDim.x + threadIdx.x;
    const int ilay = blockIdx.y*blockDim.y + threadIdx.y;

    if ( (icol < ncol) && (ilay < nlay) )
    {
        const int idx_full = icol + ilay*ncol + igpt*nlay*ncol;
        const int idx_sub = icol + ilay*ncol;
        var1_full[idx_full] = var1_sub[idx_sub];
        var2_full[idx_full] = var2_sub[idx_sub];
        var3_full[idx_full] = var3_sub[idx_sub];
        var4_full[idx_full] = var4_sub[idx_sub];
    }
}

template<typename TF>__global__
void get_from_gpoint_kernel(const int ncol, const int nlay, const int igpt,
              TF* __restrict__ var1_full, TF* __restrict__ var2_full, TF* __restrict__ var3_full,
              const TF* __restrict__ var1_sub, const TF* __restrict__ var2_sub, const TF* __restrict__ var3_sub)
{
    const int icol = blockIdx.x*blockDim.x + threadIdx.x;
    const int ilay = blockIdx.y*blockDim.y + threadIdx.y;

    if ( (icol < ncol) && (ilay < nlay ) )
    {
        const int idx_full = icol + ilay*ncol + igpt*nlay*ncol;
        const int idx_sub = icol + ilay*ncol;
        var1_full[idx_full] = var1_sub[idx_sub];
        var2_full[idx_full] = var2_sub[idx_sub];
        var3_full[idx_full] = var3_sub[idx_sub];
    }
}
