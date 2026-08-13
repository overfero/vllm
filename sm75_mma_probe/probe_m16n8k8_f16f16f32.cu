
__global__ void probe_kernel(int *out) {
    unsigned a[2];
    unsigned b[1];
    unsigned cd[4];
    for (int i = 0; i < 2; i++) a[i] = 0;
    for (int i = 0; i < 1; i++) b[i] = 0;
    for (int i = 0; i < 4; i++) cd[i] = 0;

    asm volatile(
        "mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 "
        "{%0, %1, %2, %3}, "
        "{%4, %5}, "
        "{%6}, "
        "{%7, %8, %9, %10};\n"
        : "=r"(cd[0]), "=r"(cd[1]), "=r"(cd[2]), "=r"(cd[3])
        : "r"(a[0]), "r"(a[1]), "r"(b[0]), "r"(cd[0]), "r"(cd[1]), "r"(cd[2]), "r"(cd[3])
    );

    if (threadIdx.x == 0) out[0] = (int)cd[0];
}
