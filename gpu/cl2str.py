#!/usr/bin/env python3
"""Convert .cl kernel source to C string header for embedding.

Usage: python3 cl2str.py input.cl [output.h]
  output.h defaults to input_str.h (e.g., gpu_common.cl -> gpu_common_str.h)
  Variable name derived from output filename (e.g., gpu_common_str)

  python3 cl2str.py --all
  Processes all .cl files in the script directory.
"""
import sys, os, glob

def convert(src, dst):
    with open(src, 'r') as f:
        lines = f.readlines()

    varname = os.path.basename(dst).replace('.h', '').replace('-', '_')

    with open(dst, 'w') as out:
        out.write("/* Auto-generated from %s -- do not edit */\n" % os.path.basename(src))
        out.write("static const char %s[] =\n" % varname)
        for line in lines:
            line = line.rstrip('\n')
            escaped = line.replace('\\', '\\\\').replace('"', '\\"')
            out.write('    "%s\\n"\n' % escaped)
        out.write(";\n")

    print("%s -> %s (%d lines)" % (os.path.basename(src), os.path.basename(dst), len(lines)))

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))

    if len(sys.argv) > 1 and sys.argv[1] == '--all':
        for cl in sorted(glob.glob(os.path.join(script_dir, 'gpu_*.cl'))):
            base = os.path.splitext(os.path.basename(cl))[0]
            hdr = os.path.join(script_dir, base + '_str.h')
            convert(cl, hdr)
        return

    src = sys.argv[1] if len(sys.argv) > 1 else "gpu_kernels.cl"
    if len(sys.argv) > 2:
        dst = sys.argv[2]
    else:
        base = os.path.splitext(os.path.basename(src))[0]
        dst = base + '_str.h'

    if not os.path.isabs(src):
        src = os.path.join(script_dir, src)
    if not os.path.isabs(dst):
        dst = os.path.join(script_dir, dst)

    convert(src, dst)

if __name__ == '__main__':
    main()
