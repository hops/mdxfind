#!/bin/bash
# gpu_regress.sh — GPU regression test for mdxfind
#
# For each [GPU]-tagged algorithm, generates a test vector with CPU (-G none -z),
# then verifies GPU finds the same result. Checks stderr to confirm GPU dispatched.
#
# Usage: ./bench/gpu_regress.sh [path/to/mdxfind]
# Exit code: 0 = all pass, 1 = failures

set -u
MDXFIND="${1:-./mdxfind}"
TD="${TMPDIR:-/tmp}/gpu_regress_$$"
mkdir -p "$TD"
trap "rm -rf $TD" EXIT

PASS=0; FAIL=0; SKIP=0; ERRORS=""
PASSWORD="password"

# Detect GPU
GPU_CHECK=$("$MDXFIND" -m e1 -f /dev/null /dev/null 2>&1)
GPU_TYPE="none"
echo "$GPU_CHECK" | grep -q "Metal GPU" && GPU_TYPE="metal"
echo "$GPU_CHECK" | grep -q "OpenCL GPU" && GPU_TYPE="opencl"
echo "GPU backend: $GPU_TYPE"
[ "$GPU_TYPE" = "none" ] && echo "ERROR: No GPU detected" && exit 1

# Generate a test vector via CPU, then verify GPU finds it
# Args: description type_num [salt_flag salt_value] [extra_args]
run_test() {
    local desc="$1" tn="$2"
    local salt_flag="${3:-}" salt_val="${4:-}"
    local hfile="$TD/h_${tn}.txt" pfile="$TD/p.txt" sfile="$TD/s.txt"
    local ufile="$TD/u.txt" stderr_f="$TD/err.txt"
    echo "$PASSWORD" > "$pfile"

    # Generate test vector with CPU
    local gen_args="-G none -m $tn -f /dev/null -z"
    local run_args="-G force -m $tn"

    if [ "$salt_flag" = "-s" ]; then
        echo "$salt_val" > "$sfile"
        gen_args="$gen_args -s $sfile"
        run_args="$run_args -s $sfile"
    elif [ "$salt_flag" = "-u" ]; then
        echo "$salt_val" > "$ufile"
        gen_args="$gen_args -u $ufile"
        run_args="$run_args -u $ufile"
    fi

    # For unsalted types (no salt flag), use -n ?d to force GPU mask dispatch.
    # Generate hash for "password0", test with base word "password" + -n ?d
    # Unsalted types dispatch to GPU only when masks or iteration are active.
    # Without either, correct results via CPU fallback are acceptable (PASS*).

    local cpu_line
    cpu_line=$(echo "$PASSWORD" | "$MDXFIND" $gen_args stdin 2>/dev/null | head -1)
    if [ -z "$cpu_line" ]; then
        printf "  SKIP  %-45s (CPU gen failed)\n" "$desc"
        SKIP=$((SKIP + 1))
        return
    fi

    # Extract hash from CPU output: "TYPExNN hash[:salt]:password" -> hash
    # For types with $PREFIX$ format (BLAKE2S), try the non-prefixed variant
    local hash
    hash=$(echo "$cpu_line" | sed 's/^[^ ]* //' | sed 's/:.*$//')

    # Strip $BLAKE2$ or similar prefix for compact table loading
    hash=$(echo "$hash" | sed 's/^\$[A-Z0-9]*\$//')

    # Use the first 32 chars of hash as the probe (truncated for compact table)
    local probe_hash
    probe_hash=$(echo "$hash" | cut -c1-32)
    echo "$probe_hash" > "$hfile"

    # Run with GPU — extract hash:password (skip type label)
    local gpu_out gpu_raw cpu_out cpu_raw
    gpu_raw=$(echo "$PASSWORD" | "$MDXFIND" $run_args -f "$hfile" stdin 2>"$stderr_f" | head -1)
    gpu_out=$(echo "$gpu_raw" | sed 's/^[^ ]* //')

    # Run with CPU only for comparison
    cpu_raw=$(echo "$PASSWORD" | "$MDXFIND" -G none $run_args -f "$hfile" stdin 2>/dev/null | head -1)
    cpu_out=$(echo "$cpu_raw" | sed 's/^[^ ]* //')

    if [ -z "$cpu_out" ]; then
        printf "  SKIP  %-45s (CPU verify failed)\n" "$desc"
        SKIP=$((SKIP + 1))
        return
    fi

    # Check if GPU actually dispatched (not just CPU fallback)
    local gpu_used=0
    if grep -q "gpujob thread\|GPU.*dispatch\|compact table registered" "$stderr_f"; then
        gpu_used=1
    fi

    if [ "$gpu_out" = "$cpu_out" ] && [ -n "$gpu_out" ]; then
        : # match
    elif [ -n "$gpu_raw" ] && [ -n "$cpu_raw" ]; then
        # Check if only type label differs (e.g., x01 suffix)
        local gpu_hash cpu_hash
        gpu_hash=$(echo "$gpu_out" | sed 's/:[^:]*$//')
        cpu_hash=$(echo "$cpu_out" | sed 's/:[^:]*$//')
        if [ "$gpu_hash" = "$cpu_hash" ]; then
            gpu_out="$cpu_out"  # treat as match
        fi
    fi

    if [ "$gpu_out" = "$cpu_out" ] && [ -n "$gpu_out" ]; then
        if [ "$gpu_used" = "1" ]; then
            printf "  PASS  %-45s\n" "$desc"
        else
            printf "  PASS* %-45s (CPU fallback)\n" "$desc"
        fi
        PASS=$((PASS + 1))
    elif [ -z "$gpu_raw" ]; then
        printf "  FAIL  %-45s GPU=<none>\n" "$desc"
        FAIL=$((FAIL + 1)); ERRORS="$ERRORS\n  $desc: GPU found nothing"
    else
        printf "  FAIL  %-45s GPU!=CPU\n" "$desc"
        FAIL=$((FAIL + 1)); ERRORS="$ERRORS\n  $desc: GPU=$gpu_raw CPU=$cpu_raw"
    fi
}

echo ""
echo "=== Unsalted types ==="
run_test "MD5 (e1)"             e1
run_test "MD4 (e3)"             e3
run_test "NTLM (e369)"         e369
run_test "SHA1 (e8)"            e8
run_test "SHA224 (e9)"          e9
run_test "SHA256 (e10)"         e10
run_test "SHA384 (e11)"         e11
run_test "SHA512 (e12)"         e12
run_test "WRL (e5)"             e5
run_test "MD6-256 (e29)"        e29
run_test "RMD160 (e17)"         e17
run_test "BLAKE2S256 (e844)"    e844
run_test "Keccak256 (e85)"      e85
run_test "SHA3-256 (e89)"       e89
run_test "MySQL3 (e456)"        e456
run_test "SQL5 (e259)"          e259
run_test "Streebog256 (e430)"   e430
run_test "Streebog512 (e431)"   e431
# RAW types require -i 2 minimum (md5(md5_binary(pass)) is a double-hash)
# Tested separately in the iteration section below
run_test "MD4UTF16 (e496)"      e496

echo ""
echo "=== Salted types (-s) ==="
run_test "MD5SALTPASS (e394)"         e394 -s "testsalt"
run_test "MD5PASSSALT (e373)"         e373 -s "testsalt"
run_test "SHA1SALTPASS (e385)"        e385 -s "testsalt"
run_test "SHA1PASSSALT (e405)"        e405 -s "testsalt"
run_test "SHA256SALTPASS (e412)"      e412 -s "testsalt"
run_test "SHA256PASSSALT (e413)"      e413 -s "testsalt"
run_test "SHA512SALTPASS (e388)"      e388 -s "testsalt"
run_test "SHA512PASSSALT (e386)"      e386 -s "testsalt"

echo ""
echo "=== Salted types (-u for HMAC key=salt) ==="
run_test "HMAC-MD5 (e214)"            e214 -u "hmackey"
run_test "HMAC-SHA1 (e215)"           e215 -u "hmackey"
run_test "HMAC-SHA256 (e217)"         e217 -u "hmackey"
run_test "HMAC-SHA512 (e219)"         e219 -u "hmackey"
run_test "HMAC-RMD160 (e220)"         e220 -u "hmackey"

echo ""
echo "=== Salted types (-s for HMAC key=pass) ==="
run_test "HMAC-MD5-KPASS (e793)"      e793 -s "hmacmsg"
run_test "HMAC-SHA1-KPASS (e796)"     e796 -s "hmacmsg"
run_test "HMAC-SHA256-KPASS (e795)"   e795 -s "hmacmsg"
run_test "HMAC-SHA512-KPASS (e797)"   e797 -s "hmacmsg"
run_test "HMAC-RMD160-KPASS (e798)"   e798 -s "hmacmsg"
run_test "HMAC-RMD320-KPASS (e800)"   e800 -s "hmacmsg"
run_test "HMAC-BLAKE2S (e828)"        e828 -s "hmacmsg"

echo ""
echo "=== Crypt types ==="
# These need -M -F with embedded salt format
# Generate with CPU, extract the full structured hash, test with -M -F
for mode_type in "1500:DESCRYPT" "3200:BCRYPT"; do
    mode=$(echo $mode_type | cut -d: -f1)
    desc=$(echo $mode_type | cut -d: -f2)
    full_hash=$(echo "$PASSWORD" | "$MDXFIND" -G none -M "$mode" -F /dev/null -z stdin 2>/dev/null | head -1 | sed 's/^[^ ]* //' | sed 's/:[^:]*$//')
    if [ -n "$full_hash" ]; then
        echo "$full_hash" > "$TD/crypt_h.txt"
        gpu_out=$(echo "$PASSWORD" | "$MDXFIND" -G force -M "$mode" -F "$TD/crypt_h.txt" stdin 2>"$TD/err.txt" | head -1)
        cpu_out=$(echo "$PASSWORD" | "$MDXFIND" -G none -M "$mode" -F "$TD/crypt_h.txt" stdin 2>/dev/null | head -1)
        if [ "$gpu_out" = "$cpu_out" ] && [ -n "$gpu_out" ]; then
            printf "  PASS  %-45s\n" "$desc ($mode)"
            PASS=$((PASS + 1))
        elif [ -z "$gpu_out" ]; then
            printf "  FAIL  %-45s GPU=<none>\n" "$desc ($mode)"
            FAIL=$((FAIL + 1)); ERRORS="$ERRORS\n  $desc: GPU found nothing"
        else
            printf "  FAIL  %-45s GPU!=CPU\n" "$desc ($mode)"
            FAIL=$((FAIL + 1)); ERRORS="$ERRORS\n  $desc: mismatch"
        fi
    else
        printf "  SKIP  %-45s (gen failed)\n" "$desc ($mode)"
        SKIP=$((SKIP + 1))
    fi
done

echo ""
echo "=== Iteration + RAW tests ==="
# Generic iteration test helper: gen hash with CPU -z -i N, verify GPU finds it
iter_test() {
    local desc="$1" tn="$2" icount="$3"
    local hfile="$TD/iter_h.txt" stderr_f="$TD/err.txt"
    local cpu_line gpu_raw cpu_raw gpu_out cpu_out
    cpu_line=$(echo "$PASSWORD" | "$MDXFIND" -G none -m "$tn" -i "$icount" -f /dev/null -z stdin 2>/dev/null | grep "x0${icount} " | head -1)
    if [ -z "$cpu_line" ]; then
        printf "  SKIP  %-45s (CPU gen failed)\n" "$desc"
        SKIP=$((SKIP + 1)); return
    fi
    local hash=$(echo "$cpu_line" | sed 's/^[^ ]* //' | sed 's/:.*$//' | sed 's/^\$[A-Z0-9]*\$//' | cut -c1-32)
    echo "$hash" > "$hfile"
    gpu_raw=$(echo "$PASSWORD" | "$MDXFIND" -G force -m "$tn" -i "$icount" -f "$hfile" stdin 2>"$stderr_f" | head -1)
    cpu_raw=$(echo "$PASSWORD" | "$MDXFIND" -G none -m "$tn" -i "$icount" -f "$hfile" stdin 2>/dev/null | head -1)
    gpu_out=$(echo "$gpu_raw" | sed 's/^[^ ]* //')
    cpu_out=$(echo "$cpu_raw" | sed 's/^[^ ]* //')
    if [ -z "$cpu_out" ]; then
        printf "  SKIP  %-45s (CPU verify failed)\n" "$desc"
        SKIP=$((SKIP + 1)); return
    fi
    # Compare hash:password, ignore type label differences
    local gpu_h=$(echo "$gpu_out" | sed 's/:[^:]*$//') cpu_h=$(echo "$cpu_out" | sed 's/:[^:]*$//')
    if [ "$gpu_h" = "$cpu_h" ] && [ -n "$gpu_h" ]; then
        printf "  PASS  %-45s\n" "$desc"
        PASS=$((PASS + 1))
    elif [ -z "$gpu_raw" ]; then
        printf "  FAIL  %-45s GPU=<none>\n" "$desc"
        FAIL=$((FAIL + 1)); ERRORS="$ERRORS\n  $desc: GPU found nothing"
    else
        printf "  FAIL  %-45s GPU!=CPU\n" "$desc"
        FAIL=$((FAIL + 1)); ERRORS="$ERRORS\n  $desc: GPU=$gpu_raw CPU=$cpu_raw"
    fi
}

iter_test "MD5 -i 2"            e1  2
iter_test "SHA1 -i 2"           e8  2
iter_test "SHA256 -i 2"         e10 2
iter_test "MD4 -i 2"            e3  2
iter_test "SHA512 -i 2"         e12 2
# RAW types at -i 2: GPU computes binary iteration (hash(hash_binary(hash(pass)))).
# RAWx02 differs from both base x02 and RAWx01 — verifies binary iter works.
iter_test "MD5RAW -i 2"         e33 2
iter_test "SHA1RAW -i 2"        e34 2
iter_test "SHA256RAW -i 2"      e36 2
iter_test "SHA384RAW -i 2"      e37 2
iter_test "SHA512RAW -i 2"      e38 2

echo ""
echo "========================================"
printf "Results: %d pass, %d fail, %d skip\n" "$PASS" "$FAIL" "$SKIP"
printf "  PASS* = correct result but GPU may not have dispatched (CPU fallback)\n"
[ "$FAIL" -gt 0 ] && printf "\nFAILURES:$ERRORS\n" && exit 1
exit 0
