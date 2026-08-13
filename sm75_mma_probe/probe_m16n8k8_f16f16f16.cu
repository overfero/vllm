
__global__ void probe_kernel(int *out) {
    unsigned a[2];
    unsigned b[1];
    unsigned cd[2];
    for (int i = 0; i < 2; i++) a[i] = 0;
    for (int i = 0; i < 1; i++) b[i] = 0;
    for (int i = 0; i < 2; i++) cd[i] = 0;

    asm volatile(
        "mma.sync.aligned.m16n8k8.row.col.f16.f16.f16.f16 "
        "{%0, %1}, "
        "{%2, %3}, "
        "{%4}, "
        "{%5, %6};\n"
        : "=r"(cd[0]), "=r"(cd[1])
        : "r"(a[0]), "r"(a[1]), "r"(b[0]), "r"(cd[0]), "r"(cd[1])
    );

    if (threadIdx.x == 0) out[0] = (int)cd[0];
}
