FILE(GLOB sourcefiles
    "../src_raytracer_cuda/raytracer_kernels.cu"
    "../src_raytracer_cuda/Raytracer.cu")

include_directories("../include" SYSTEM ${INCLUDE_DIRS})

add_library(rte_rrtmgp_raytracer_cuda STATIC ${sourcefiles})
