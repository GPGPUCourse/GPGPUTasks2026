"""Exercise conservative, exact and overflowing tile reservations on the stock Gnome mesh.

Usage: python3 scripts/test_coarse_capacity.py build-cuda/main_rasterization DEVICE_ID
"""
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile


def main(binary, device):
    binary = Path(binary).resolve()
    source = Path(__file__).resolve().parents[1] / 'data/gnome'
    header, data = (source / 'gnome.ply').read_text().split('end_header\n', 1)
    nverts = int(re.search(r'^element vertex (\d+)$', header, re.M)[1])
    lines = data.splitlines()
    vertices, faces = lines[:nverts], lines[nverts:]
    with tempfile.TemporaryDirectory() as tmp:
        folder = Path(tmp) / 'data/gnome'
        folder.mkdir(parents=True)
        shutil.copyfile(source / 'camera.txt', folder / 'camera.txt')
        for copies, expected in [(1, 'boxes'), (6, 'exact'), (12, 'overflow')]:
            # Keep Bin scratch large enough to reach Coarse; degenerate padding
            # faces generate neither fragments nor queue entries.
            triangles = faces * copies + ['3 0 0 0'] * (len(faces) * copies)
            head = re.sub(r'^element face \d+$', f'element face {len(triangles)}', header, flags=re.M)
            (folder / 'gnome.ply').write_text(head + 'end_header\n' + '\n'.join(vertices + triangles) + '\n')
            env = dict(os.environ, GPGPU_CUDA_RASTERIZER='tiled', GPGPU_VERIFY_CUDA_RASTERIZER='1')
            run = subprocess.run([str(binary), device], cwd=tmp, env=env, stdout=subprocess.PIPE,
                                 stderr=subprocess.STDOUT, text=True)
            stats = re.search(r'^\[CUDA tiles\].*?, (\d+) tile refs in (\d+) slots, .*$', run.stdout, re.M)
            if run.returncode or not stats or '[CUDA tiles] differing packed pixels vs CUDA baseline: 0' not in run.stdout:
                raise RuntimeError(run.stdout)
            actual = ('overflow' if 'baseline fallback' in stats[0] else
                      'exact' if stats[1] == stats[2] else 'boxes')
            if actual != expected:
                raise AssertionError(f'expected {expected}, got {stats[0]}')
            print(f'{actual}: {stats[0]}; identical packed pixels', flush=True)


if __name__ == '__main__':
    main(*sys.argv[1:])
