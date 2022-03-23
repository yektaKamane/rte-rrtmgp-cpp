FILE(GLOB sourcefiles
    "../src_raytracer_cuda/raytracer_kernels_bw.cu"
    "../src_raytracer_cuda/raytracer_kernels.cu"
    "../src_raytracer_cuda/Raytracer.cu"
    "../src_raytracer_cuda/Raytracer_bw.cu")

include_directories("../include" SYSTEM ${INCLUDE_DIRS})

add_library(rte_rrtmgp_raytracer_cuda STATIC ${sourcefiles})
