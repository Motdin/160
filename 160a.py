#!/usr/bin/env python3
"""

Install on Python 3.13
----------------------
    python3.13 -m pip install cuda-python numpy


"""

import argparse
import os
import sys
import time

import numpy as np  # still needed for the uint32 bigint arrays

from cuda import cuda, nvrtc


# --------------------------------------------------------------------------- #
# Error checking helpers
# --------------------------------------------------------------------------- #
def _check(err):
    """Raise on any non-CUDA_SUCCESS status, printing the error name."""
    if err[0] != cuda.CUresult.CUDA_SUCCESS:
        name = err[0].name
        raise RuntimeError(f"CUDA error: {name}")
    return err


def _check_nvrtc(err):
    if err[0] != nvrtc.nvrtcResult.NVRTC_SUCCESS:
        raise RuntimeError(f"NVRTC error: {err[0].name}")
    return err


# --------------------------------------------------------------------------- #
# Device helpers (replace the PyCUDA driver calls)
# --------------------------------------------------------------------------- #
def to_device(arr):
    """Copy a numpy array to device memory; return the CUdeviceptr."""
    arr = np.ascontiguousarray(arr)
    nbytes = arr.nbytes
    dptr = _check(cuda.cuMemAlloc_v2(nbytes))[1]
    _check(cuda.cuMemcpyHtoD_v2(dptr, arr.ctypes.data, nbytes))
    return dptr


def from_device(dptr, arr):
    """Copy device memory back into a numpy array."""
    arr = np.ascontiguousarray(arr)
    _check(cuda.cuMemcpyDtoH_v2(arr.ctypes.data, dptr, arr.nbytes))
    return arr


# --------------------------------------------------------------------------- #
# Domain helpers (unchanged logic from the original)
# --------------------------------------------------------------------------- #
def int_to_bigint_np(val):
    bigint_arr = np.zeros(8, dtype=np.uint32)
    for j in range(8):
        bigint_arr[j] = (val >> (32 * j)) & 0xFFFFFFFF
    return bigint_arr


def load_target_h160_bin(filename):
    targets_bin = bytearray()
    with open(filename) as f:
        for line in f:
            h160_hex = line.strip()
            if len(h160_hex) == 40:
                try:
                    targets_bin.extend(bytes.fromhex(h160_hex))
                except ValueError:
                    pass
    return targets_bin


def init_secp256k1_constants(module):
    """Copy secp256k1 p / n / G into the kernel's __constant__ globals."""
    # Fetch the device pointers of the three __constant__ symbols.
    def get_global_ptr(name):
        return _check(cuda.cuModuleGetGlobal(module, name.encode()))[1]

    const_p_gpu = get_global_ptr("const_p")
    const_n_gpu = get_global_ptr("const_n")
    const_G_gpu = get_global_ptr("const_G_jacobian")

    p_data = np.array(
        [0xFFFFFC2F, 0xFFFFFFFE, 0xFFFFFFFF, 0xFFFFFFFF,
         0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF],
        dtype=np.uint32,
    )
    _check(cuda.cuMemcpyHtoD_v2(const_p_gpu, p_data.ctypes.data, p_data.nbytes))

    n_data = np.array(
        [0xD0364141, 0xBFD25E8C, 0xAF48A03B, 0xBAAEDCE6,
         0xFFFFFFFE, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF],
        dtype=np.uint32,
    )
    _check(cuda.cuMemcpyHtoD_v2(const_n_gpu, n_data.ctypes.data, n_data.nbytes))

    g_x = np.array(
        [0x16F81798, 0x59F2815B, 0x2DCE28D9, 0x029BFCDB,
         0xCE870B07, 0x55A06295, 0xF9DCBBAC, 0x79BE667E],
        dtype=np.uint32,
    )
    g_y = np.array(
        [0xFB10D4B8, 0x9C47D08F, 0xA6855419, 0xFD17B448,
         0x0E1108A8, 0x5DA4FBFC, 0x26A3C465, 0x483ADA77],
        dtype=np.uint32,
    )
    g_z = np.array([1, 0, 0, 0, 0, 0, 0, 0], dtype=np.uint32)
    g_infinity = np.array([False], dtype=np.bool_)

    # Rebuild the exact same struct layout the kernel expects:
    #   struct ECPointJac { BigInt X, Y, Z; bool infinity; }
    #   struct BigInt    { uint32_t data[8]; }
    # 3 * 8 uint32 + 1 byte (padded by the compiler to the struct alignment).
    ecpoint_jac_dtype = np.dtype(
        [("X", np.uint32, 8), ("Y", np.uint32, 8),
         ("Z", np.uint32, 8), ("infinity", np.bool_)]
    )
    g_jac = np.zeros(1, dtype=ecpoint_jac_dtype)
    g_jac["X"], g_jac["Y"], g_jac["Z"], g_jac["infinity"] = g_x, g_y, g_z, g_infinity

    _check(cuda.cuMemcpyHtoD_v2(const_G_gpu, g_jac.ctypes.data, g_jac.nbytes))


def run_precomputation(module):
    precompute_kernel = _check(
        cuda.cuModuleGetFunction(module, b"precompute_G_table_kernel")
    )[1]
    _check(
        cuda.cuLaunchKernel(
            precompute_kernel,
            1, 1, 1,   # grid
            1, 1, 1,   # block (single thread does all 256 doublings)
            0, None,   # shared mem, stream
            None, None,
        )
    )
    _check(cuda.cuCtxSynchronize())
    print("[*] Precomputation table selesai.")


def compile_and_load(cuda_code_path, arch="sm_75"):
    """NVRTC-compile kernel160.cu and load the resulting module."""
    with open(cuda_code_path, "r") as f:
        cuda_code = f.read()

    prog = _check_nvrtc(
        nvrtc.nvrtcCreateProgram(
            cuda_code.encode(), b"kernel160.cu", 0, [], []
        )
    )[1]

    # NOTE: the kernel includes <cuda_runtime.h>, so we must point NVRTC at
    # the CUDA include directory for it to resolve.  CUDA_HOME is the most
    # common env var (e.g. /usr/local/cuda); fall back to a couple of defaults.
    cuda_home = os.environ.get("CUDA_HOME", "/usr/local/cuda")
    include_dir = os.path.join(cuda_home, "include")

    opts = []
    if os.path.isdir(include_dir):
        opts.append(f"-I{include_dir}")
    opts += ["-std=c++11", f"-arch={arch}"]

    opt_bytes = [o.encode() for o in opts]
    res = _check_nvrtc(
        nvrtc.nvrtcCompileProgram(prog, len(opt_bytes), opt_bytes)
    )[0]

    # Fetch the compile log (always populated; on failure it holds the errors).
    log_size = _check_nvrtc(nvrtc.nvrtcGetProgramLogSize(prog))[1]
    log = b" " * log_size
    _check_nvrtc(nvrtc.nvrtcGetProgramLog(prog, log))
    log_text = log.decode(errors="replace")

    if res != nvrtc.nvrtcResult.NVRTC_SUCCESS:
        sys.stderr.write(log_text)
        raise RuntimeError("NVRTC compilation failed (see log above)")

    # Retrieve the PTX/cubin and load it into the driver.
    code_size = _check_nvrtc(nvrtc.nvrtcGetCUBINSize(prog))[1]
    cubin = b" " * code_size
    from cuda import cuda as _cd  # noqa: F401  (kept local to match types)
    _check_nvrtc(nvrtc.nvrtcGetCUBIN(prog, cubin))

    module = _check(cuda.cuModuleLoadData(cubin))[1]
    return module


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Full CUDA HASH160 Brute Forcer with Sliding Window (cuda-python)"
    )
    parser.add_argument("--start", type=lambda x: int(x, 0), required=True,
                        help="Kunci privat awal")
    parser.add_argument("--end", type=lambda x: int(x, 0), required=True,
                        help="Kunci privat akhir")
    parser.add_argument("--step", type=lambda x: int(x, 0), default=1,
                        help="Langkah penambahan")
    parser.add_argument("--file", default="target.txt",
                        help="File target HASH160")
    parser.add_argument("--keys-per-launch", type=int, default=2 ** 24,
                        help="Jumlah kunci per batch GPU (default: 16,777,216)")
    parser.add_argument("--arch", default="sm_75",
                        help="GPU compute capability (default: sm_75 / Tesla T4). "
                             "Set to your card's arch, e.g. sm_86, sm_89, sm_120.")
    args = parser.parse_args(argv)

    # ---- Init the CUDA driver / context (was `pycuda.autoinit`) ---- #
    _check(cuda.cuInit(0))
    device = _check(cuda.cuDeviceGet(0))[1]
    context = _check(cuda.cuCtxCreate(0, device))[1]

    try:
        # Make this context current on the thread (replaces autoinit's push).
        # cuCtxCreate already makes it current on the calling thread, but be
        # explicit for safety across threads.
        _check(cuda.cuCtxSetCurrent(context))

        kernel_path = "kernel160.cu"
        if not os.path.isfile(kernel_path):
            print(f"[!] FATAL: File '{kernel_path}' tidak ditemukan.")
            return 1

        print("[*] Mengkompilasi kernel Full CUDA (NVRTC)...")
        module = compile_and_load(kernel_path, arch=args.arch)

        init_secp256k1_constants(module)

        print("[*] Menjalankan precomputation...")
        run_precomputation(module)

        find_hash_kernel = _check(
            cuda.cuModuleGetFunction(module, b"find_hash_kernel_optimized")
        )[1]
        print("[*] Inisialisasi selesai.")

    except Exception:
        _check(cuda.cuCtxDestroy(context))
        raise

    target_bin = load_target_h160_bin(args.file)
    num_targets = len(target_bin) // 20
    if num_targets == 0:
        print(f"[!] Tidak ada HASH160 valid di {args.file}.")
        _check(cuda.cuCtxDestroy(context))
        return 1

    print(f"[*] Berhasil memuat {num_targets} HASH160 target.")

    # ---- Device allocations ---- #
    try:
        d_targets = to_device(np.frombuffer(bytes(target_bin), dtype=np.uint8))

        d_result = _check(cuda.cuMemAlloc_v2(32))[1]
        d_found_flag = _check(cuda.cuMemAlloc_v2(4))[1]
        _check(cuda.cuMemsetD32_v2(d_result, 0, 8))
        _check(cuda.cuMemsetD32_v2(d_found_flag, 0, 1))

        # ---- The scanning loop (logic unchanged from original) ---- #
        total_keys_checked = 0
        start_time = time.time()
        iteration = 0
        current_start = args.start
        current_end = args.end
        found_flag_host = np.zeros(1, dtype=np.int32)

        block_size = 256

        while found_flag_host[0] == 0 and current_end >= 0:
            print(
                f"\n[+] Iterasi {iteration}: Rentang {hex(current_start)} - "
                f"{hex(current_end)} "
                f"(Size: {(current_end - current_start)//args.step + 1:,} keys)"
            )

            temp_current = current_start
            keys_in_window = (current_end - current_start) // args.step + 1
            keys_processed_in_window = 0

            while temp_current <= current_end and found_flag_host[0] == 0:
                keys_left = (current_end - temp_current) // args.step + 1
                keys_this_launch = min(args.keys_per_launch, keys_left)
                if keys_this_launch <= 0:
                    break

                # NOTE: `keys_per_launch` must stay within 32-bit `idx`;
                # the kernel computes `idx` as `blockIdx.x * blockDim.x +
                # threadIdx.x` then casts idx to uint32 inside bigint_mul_uint32.
                # With block_size=256, keys_per_launch up to 2**32 works, but a
                # single launch of 2**24 (16M) threads is already the default.
                start_key_np = int_to_bigint_np(temp_current)
                step_np = int_to_bigint_np(args.step)

                grid_size = (keys_this_launch + block_size - 1) // block_size

                d_start_key = to_device(start_key_np)
                d_step = to_device(step_np)

                # ---- Launch kernel (was `find_hash_kernel(...)`) ---- #
                kernel_params = np.array([np.uint64(int(d_start_key)),
                                          np.uint64(np.uint64(keys_this_launch)),
                                          np.uint64(int(d_step)),
                                          np.uint64(int(d_targets)),
                                          np.int32(np.int32(num_targets)),
                                          np.uint64(int(d_result)),
                                          np.uint64(int(d_found_flag))],
                                         dtype=np.uint64)
                _check(
                    cuda.cuLaunchKernel(
                        find_hash_kernel,
                        grid_size, 1, 1,
                        block_size, 1, 1,
                        0, None,
                        kernel_params.ctypes.data, None,
                    )
                )
                _check(cuda.cuCtxSynchronize())

                _check(cuda.cuMemFree_v2(d_start_key))
                _check(cuda.cuMemFree_v2(d_step))

                total_keys_checked += keys_this_launch
                keys_processed_in_window += keys_this_launch

                from_device(d_found_flag, found_flag_host)

                elapsed = time.time() - start_time
                speed = total_keys_checked / elapsed if elapsed > 0 else 0
                window_progress = 100 * keys_processed_in_window / keys_in_window
                progress_str = (
                    f"[+] Total: {total_keys_checked:,} | "
                    f"Kecepatan: {speed:,.2f} k/s | "
                    f"Window: {window_progress:.1f}% | Kunci: {hex(temp_current)}"
                )
                sys.stdout.write("\r" + progress_str.ljust(120))
                sys.stdout.flush()

                temp_current += keys_this_launch * args.step

            current_start -= 1
            current_end -= 1
            iteration += 1

            if current_end < 0:
                print("\n[!] Batas bawah tercapai (kunci negatif)")
                break

        if found_flag_host[0] == 1:
            sys.stdout.write("\n")
            sys.stdout.flush()
            print("\n[+] KUNCI PRIVAT DITEMUKAN!")
            found_privkey_np = np.zeros(8, dtype=np.uint32)
            from_device(d_result, found_privkey_np)

            privkey_int = 0
            for j in range(8):
                privkey_int |= int(found_privkey_np[j]) << (32 * j)

            print(f"    Kunci Privat: {hex(privkey_int)}")
            print(f"    Rentang ditemukan: {hex(current_start+1)} - "
                  f"{hex(current_end+1)} (Iterasi {iteration-1})")
            with open("found.txt", "a") as f:
                f.write(f"Kunci Privat: {hex(privkey_int)}\n")
                f.write(f"Rentang: {hex(current_start+1)} - {hex(current_end+1)}\n")
        else:
            print("\n\n[+] Pencarian selesai. Tidak ada yang cocok ditemukan.")
            print(f"    Total kunci dicoba: {total_keys_checked:,}")
            print(f"    Rentang terakhir: {hex(current_start+1)} - {hex(current_end+1)}")

        _check(cuda.cuMemFree_v2(d_targets))
        _check(cuda.cuMemFree_v2(d_result))
        _check(cuda.cuMemFree_v2(d_found_flag))

    except KeyboardInterrupt:
        print("\n\n[!] Dihentikan oleh pengguna")
        print(f"    Iterasi terakhir: {iteration} | Rentang: "
              f"{hex(current_start)} - {hex(current_end)}")
        print(f"    Total kunci dicoba: {total_keys_checked:,}")
        return 0
    finally:
        _check(cuda.cuCtxDestroy(context))

    return 0


if __name__ == "__main__":
    sys.exit(main())
