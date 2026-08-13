"""Empirical probe: which raw PTX `mma.sync` shapes does ptxas actually
accept for sm_75 (real Tesla T4)? This is deliberately independent of
humming's code - raw inline PTX in a minimal .cu file, compiled with the
system nvcc/ptxas toolchain, targeting the real GPU's real compute
capability (7.5, confirmed via `nvidia-smi`).

Goal: answer "can SM75 execute X" with a real compiler verdict per shape,
not documentation memory. This is what root-caused Bug 2 in
humming-kernels==0.1.10's select_mma_op_class() - see
../humming_fix/patch.py for the fix this probe justifies.
"""
import subprocess
import os
from pathlib import Path

OUT = str(Path(__file__).resolve().parent)

# Each entry: (label, PTX mma instruction line, operand counts for a/b/c/d
# as (num_regs_a, num_regs_b, num_regs_c_d), dtype constraint used for
# register type in inline asm).
CASES = [
    # --- fp16 x fp16 -> fp16, various k ---
    dict(
        label="m16n8k8_f16f16f16",
        mma="mma.sync.aligned.m16n8k8.row.col.f16.f16.f16.f16",
        a_regs=2, b_regs=1, cd_regs=2, reg_ty="r",
    ),
    dict(
        label="m16n8k16_f16f16f16",
        mma="mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16",
        a_regs=4, b_regs=2, cd_regs=2, reg_ty="r",
    ),
    # --- fp16 x fp16 -> fp32 accum ---
    dict(
        label="m16n8k8_f16f16f32",
        mma="mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32",
        a_regs=2, b_regs=1, cd_regs=4, reg_ty="r",
    ),
    dict(
        label="m16n8k16_f16f16f32",
        mma="mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32",
        a_regs=4, b_regs=2, cd_regs=4, reg_ty="r",
    ),
    # --- older volta-style shape, sanity check it's still legal ---
    dict(
        label="m8n8k4_f16f16f32",
        mma="mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32",
        a_regs=1, b_regs=1, cd_regs=8, reg_ty="r",
    ),
]

TEMPLATE = """
__global__ void probe_kernel(int *out) {{
    unsigned a[{a_regs}];
    unsigned b[{b_regs}];
    unsigned cd[{cd_regs}];
    for (int i = 0; i < {a_regs}; i++) a[i] = 0;
    for (int i = 0; i < {b_regs}; i++) b[i] = 0;
    for (int i = 0; i < {cd_regs}; i++) cd[i] = 0;

    asm volatile(
        "{mma} "
        "{{{cd_out_list}}}, "
        "{{{a_list}}}, "
        "{{{b_list}}}, "
        "{{{cd_in_list}}};\\n"
        : {cd_out_constraints}
        : {a_constraints}, {b_constraints}, {cd_in_constraints}
    );

    if (threadIdx.x == 0) out[0] = (int)cd[0];
}}
"""


def build_source(case):
    a_regs, b_regs, cd_regs, ty = case["a_regs"], case["b_regs"], case["cd_regs"], case["reg_ty"]
    a_list = ", ".join(f"%{i}" for i in range(cd_regs, cd_regs + a_regs))
    b_list = ", ".join(f"%{i}" for i in range(cd_regs + a_regs, cd_regs + a_regs + b_regs))
    cd_in_list = ", ".join(f"%{i}" for i in range(cd_regs + a_regs + b_regs, cd_regs + a_regs + b_regs + cd_regs))
    cd_out_list = ", ".join(f"%{i}" for i in range(cd_regs))
    cd_out_constraints = ", ".join(f'"={ty}"(cd[{i}])' for i in range(cd_regs))
    a_constraints = ", ".join(f'"{ty}"(a[{i}])' for i in range(a_regs))
    b_constraints = ", ".join(f'"{ty}"(b[{i}])' for i in range(b_regs))
    cd_in_constraints = ", ".join(f'"{ty}"(cd[{i}])' for i in range(cd_regs))
    return TEMPLATE.format(
        mma=case["mma"], a_regs=a_regs, b_regs=b_regs, cd_regs=cd_regs,
        a_list=a_list, b_list=b_list, cd_in_list=cd_in_list, cd_out_list=cd_out_list,
        cd_out_constraints=cd_out_constraints, a_constraints=a_constraints,
        b_constraints=b_constraints, cd_in_constraints=cd_in_constraints,
    )


def main():
    os.makedirs(OUT, exist_ok=True)
    results = []
    for case in CASES:
        src = build_source(case)
        fname = os.path.join(OUT, f"probe_{case['label']}.cu")
        with open(fname, "w") as f:
            f.write(src)
        proc = subprocess.run(
            ["nvcc", "-arch=sm_75", "-c", fname, "-o", fname.replace(".cu", ".o")],
            capture_output=True, text=True,
        )
        ok = proc.returncode == 0
        results.append((case["label"], case["mma"], ok, proc.stderr.strip()))

    print("=" * 100)
    for label, mma, ok, err in results:
        status = "COMPILES on sm_75" if ok else "REJECTED on sm_75"
        print(f"[{status}] {label}\n    PTX: {mma}")
        if not ok:
            for line in err.splitlines():
                if "error" in line.lower():
                    print(f"    -> {line.strip()}")
        print()
    return results


if __name__ == "__main__":
    main()
