import pycuda.autoinit
import pycuda.driver as cuda
from pycuda.compiler import SourceModule
import numpy as np
import time
import os
import sys
import argparse

def int_to_bigint_np(val):
    bigint_arr = np.zeros(8, dtype=np.uint32)
    for j in range(8): bigint_arr[j] = (val >> (32 * j)) & 0xFFFFFFFF
    return bigint_arr

def load_target_h160_bin(filename):
    targets_bin = bytearray()
    with open(filename) as f:
        for line in f:
            h160_hex = line.strip()
            if len(h160_hex) == 40:
                try: targets_bin.extend(bytes.fromhex(h160_hex))
                except ValueError: pass
    return targets_bin

def init_secp256k1_constants(mod):
    p_data=np.array([0xFFFFFC2F,0xFFFFFFFE,0xFFFFFFFF,0xFFFFFFFF,0xFFFFFFFF,0xFFFFFFFF,0xFFFFFFFF,0xFFFFFFFF],dtype=np.uint32);const_p_gpu=mod.get_global("const_p")[0];cuda.memcpy_htod(const_p_gpu,p_data)
    n_data=np.array([0xD0364141,0xBFD25E8C,0xAF48A03B,0xBAAEDCE6,0xFFFFFFFE,0xFFFFFFFF,0xFFFFFFFF,0xFFFFFFFF],dtype=np.uint32);const_n_gpu=mod.get_global("const_n")[0];cuda.memcpy_htod(const_n_gpu,n_data)
    g_x=np.array([0x16F81798,0x59F2815B,0x2DCE28D9,0x029BFCDB,0xCE870B07,0x55A06295,0xF9DCBBAC,0x79BE667E],dtype=np.uint32);g_y=np.array([0xFB10D4B8,0x9C47D08F,0xA6855419,0xFD17B448,0x0E1108A8,0x5DA4FBFC,0x26A3C465,0x483ADA77],dtype=np.uint32);g_z=np.array([1,0,0,0,0,0,0,0],dtype=np.uint32);g_infinity=np.array([False],dtype=np.bool_)
    ecpoint_jac_dtype=np.dtype([('X',np.uint32,8),('Y',np.uint32,8),('Z',np.uint32,8),('infinity',np.bool_)]);g_jac=np.zeros(1,dtype=ecpoint_jac_dtype)
    g_jac['X'],g_jac['Y'],g_jac['Z'],g_jac['infinity']=g_x,g_y,g_z,g_infinity
    const_G_gpu=mod.get_global("const_G_jacobian")[0];cuda.memcpy_htod(const_G_gpu, g_jac)

# Fungsi baru untuk menjalankan precomputation
def run_precomputation(mod):
    precompute_kernel = mod.get_function("precompute_G_table_kernel")
    precompute_kernel(block=(1, 1, 1))
    cuda.Context.synchronize()
    print("[*] Precomputation table selesai.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Full CUDA HASH160 Brute Forcer with Sliding Window')
    parser.add_argument('--start', type=lambda x: int(x, 0), required=True, help='Kunci privat awal')
    parser.add_argument('--end', type=lambda x: int(x, 0), required=True, help='Kunci privat akhir')
    parser.add_argument('--step', type=lambda x: int(x, 0), default=1, help='Langkah penambahan')
    parser.add_argument('--file', default='target.txt', help='File target HASH160')
    parser.add_argument('--keys-per-launch', type=int, default=2**24, help='Jumlah kunci per batch GPU (default: 16,777,216)')
    args = parser.parse_args()

    # Inisialisasi di luar loop
    print("[*] Memuat file kernel CUDA tunggal...")
    try:
        with open('kernel160.cu', 'r') as f:
            full_cuda_code = f.read()
    except FileNotFoundError:
        print("[!] FATAL: File 'kernel.cu' tidak ditemukan.")
        sys.exit(1)

    print("[*] Mengkompilasi kernel Full CUDA...")
    # Tambahkan arch specification untuk Tesla T4 (sm_75)
    mod = SourceModule(full_cuda_code, no_extern_c=False, options=['-std=c++11', '-arch=sm_75'])
    init_secp256k1_constants(mod)

    # Jalankan precomputation sebelum kernel utama
    print("[*] Menjalankan precomputation...")
    run_precomputation(mod)

    # Gunakan kernel yang dioptimalkan dengan precomputation
    find_hash_kernel = mod.get_function("find_hash_kernel_optimized")
    print("[*] Inisialisasi selesai.")

    target_bin = load_target_h160_bin(args.file)
    num_targets = len(target_bin) // 20
    if num_targets == 0:
        print(f"[!] Tidak ada HASH160 valid di {args.file}."); sys.exit(1)

    print(f"[*] Berhasil memuat {num_targets} HASH160 target.")
    d_targets = cuda.mem_alloc(len(target_bin))
    cuda.memcpy_htod(d_targets, np.frombuffer(target_bin, dtype=np.uint8))

    d_result = cuda.mem_alloc(32); d_found_flag = cuda.mem_alloc(4)
    cuda.memset_d32(d_result, 0, 8); cuda.memset_d32(d_found_flag, 0, 1)

    # Inisialisasi untuk loop
    total_keys_checked = 0
    start_time = time.time()
    iteration = 0
    current_start = args.start
    current_end = args.end
    found_flag_host = np.zeros(1, dtype=np.int32)

    # =======================================================
    # MODIFIKASI: SLIDING WINDOW DENGAN MENGURANGI START DAN END
    # =======================================================
    try:
        while found_flag_host[0] == 0 and current_end >= 0:
            print(f"\n[+] Iterasi {iteration}: Rentang {hex(current_start)} - {hex(current_end)} (Size: {(current_end - current_start)//args.step + 1:,} keys)")

            temp_current = current_start
            keys_in_window = (current_end - current_start) // args.step + 1
            keys_processed_in_window = 0

            # Loop dalam untuk memindai seluruh rentang saat ini
            while temp_current <= current_end and found_flag_host[0] == 0:
                keys_left = (current_end - temp_current) // args.step + 1
                keys_this_launch = min(args.keys_per_launch, keys_left)
                if keys_this_launch <= 0: break

                start_key_np = int_to_bigint_np(temp_current)
                step_np = int_to_bigint_np(args.step)

                block_size = 256
                grid_size = (keys_this_launch + block_size - 1) // block_size

                # Luncurkan kernel yang dioptimalkan
                find_hash_kernel(
                    cuda.In(start_key_np), np.uint64(keys_this_launch), cuda.In(step_np),
                    d_targets, np.int32(num_targets),
                    d_result, d_found_flag,
                    block=(block_size, 1, 1), grid=(grid_size, 1)
                )
                cuda.Context.synchronize()

                total_keys_checked += keys_this_launch
                keys_processed_in_window += keys_this_launch

                # Periksa hasil
                cuda.memcpy_dtoh(found_flag_host, d_found_flag)

                elapsed = time.time() - start_time
                speed = total_keys_checked / elapsed if elapsed > 0 else 0
                window_progress = 100 * keys_processed_in_window / keys_in_window
                progress_str = f"[+] Total: {total_keys_checked:,} | Kecepatan: {speed:,.2f} k/s | Window: {window_progress:.1f}% | Kunci: {hex(temp_current)}"
                sys.stdout.write('\r' + progress_str.ljust(120)); sys.stdout.flush()

                temp_current += keys_this_launch * args.step

            # Geser window ke bawah (kurangi start dan end)
            current_start -= 1
            current_end -= 1
            iteration += 1


            # Hentikan jika mencapai batas bawah
            if current_end < 0:
                print("\n[!] Batas bawah tercapai (kunci negatif)")
                break

        # Setelah loop selesai, cetak hasilnya
        if found_flag_host[0] == 1:
            sys.stdout.write('\n'); sys.stdout.flush()
            print("\n[+] KUNCI PRIVAT DITEMUKAN!")
            found_privkey_np = np.zeros(8, dtype=np.uint32)
            cuda.memcpy_dtoh(found_privkey_np, d_result)

            privkey_int = 0
            for j in range(8): privkey_int |= int(found_privkey_np[j]) << (32 * j)

            print(f"    Kunci Privat: {hex(privkey_int)}")
            print(f"    Rentang ditemukan: {hex(current_start+1)} - {hex(current_end+1)} (Iterasi {iteration-1})")
            with open("found.txt", "a") as f:
                f.write(f"Kunci Privat: {hex(privkey_int)}\n")
                f.write(f"Rentang: {hex(current_start+1)} - {hex(current_end+1)}\n")
        else:
            print("\n\n[+] Pencarian selesai. Tidak ada yang cocok ditemukan.")
            print(f"    Total kunci dicoba: {total_keys_checked:,}")
            print(f"    Rentang terakhir: {hex(current_start+1)} - {hex(current_end+1)}")

    except KeyboardInterrupt:
        print("\n\n[!] Dihentikan oleh pengguna")
        print(f"    Iterasi terakhir: {iteration} | Rentang: {hex(current_start)} - {hex(current_end)}")
        print(f"    Total kunci dicoba: {total_keys_checked:,}")
        sys.exit(0)
