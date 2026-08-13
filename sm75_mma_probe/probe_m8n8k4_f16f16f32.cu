
__global__ void probe_kernel(int *out) {
    unsigned a[1];
    unsigned b[1];
    unsigned cd[8];
    for (int i = 0; i < 1; i++) a[i] = 0;
    for (int i = 0; i < 1; i++) b[i] = 0;
    for (int i = 0; i < 8; i++) cd[i] = 0;

    asm volatile(
        "mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32 "
        "{%0, %1, %2, %3, %4, %5, %6, %7}, "
        "{%8}, "
        "{%9}, "
        "{%10, %11, %12, %13, %14, %15, %16, %17};\n"
        : "=r"(cd[0]), "=r"(cd[1]), "=r"(cd[2]), "=r"(cd[3]), "=r"(cd[4]), "=r"(cd[5]), "=r"(cd[6]), "=r"(cd[7])
        : "r"(a[0]), "r"(b[0]), "r"(cd[0]), "r"(cd[1]), "r"(cd[2]), "r"(cd[3]), "r"(cd[4]), "r"(cd[5]), "r"(cd[6]), "r"(cd[7])
    );

    if (threadIdx.x == 0) out[0] = (int)cd[0];
}
