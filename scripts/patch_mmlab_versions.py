#!/usr/bin/env python3
# Copyright (c) Seeed Technology Co.,Ltd. All rights reserved.
"""Relax the hard-coded MMCV upper-version assertions of mmdet/mmcls.

mmdet 3.0.x and mmcls 1.0.0rc6 assert ``mmcv < 2.1.0`` at import time.
We deliberately use mmcv 2.2.0 - the final mmcv release and the only one
that compiles against recent PyTorch (>= 2.6) - so the assertion must be
relaxed in the *installed* packages.

The patch is applied in-place to the installed ``__init__.py`` files and
fails loudly if the expected pattern is not found (e.g. the package
versions drifted), instead of silently doing nothing.

Usage:
    python scripts/patch_mmlab_versions.py            # patch
    python scripts/patch_mmlab_versions.py --verify   # patch + import check
"""

import argparse
import importlib.util
import os
import re
import sys

# packages -> new mmcv maximum version (mmcv 3.x does not exist, so this
# effectively disables the upper bound while remaining an explicit bound)
PACKAGES = {
    'mmdet': '3.0.0',
    'mmcls': '3.0.0',
}

PATTERN = re.compile(r"mmcv_maximum_version\s*=\s*['\"][\d.]+['\"]")


def patch_package(name: str, new_maximum: str) -> bool:
    spec = importlib.util.find_spec(name)
    if spec is None or not spec.submodule_search_locations:
        print(f'[SKIP] {name} is not installed')
        return False

    init_file = os.path.join(spec.submodule_search_locations[0], '__init__.py')
    with open(init_file, 'r', encoding='utf-8') as f:
        content = f.read()

    replacement = f"mmcv_maximum_version = '{new_maximum}'"
    patched, count = PATTERN.subn(replacement, content)

    if count == 0:
        if f"mmcv_maximum_version = '{new_maximum}'" in content:
            print(f'[OK]   {name}: already patched ({init_file})')
            return True
        print(f'[FAIL] {name}: no mmcv_maximum_version found in {init_file}')
        print('       The installed version may have changed - please check it manually.')
        return False

    with open(init_file, 'w', encoding='utf-8') as f:
        f.write(patched)
    print(f'[OK]   {name}: mmcv_maximum_version -> {new_maximum} ({init_file})')
    return True


ADAFACTOR_OLD = "        OPTIMIZERS.register_module(name='Adafactor', module=Adafactor)\n        transformer_optimizers.append('Adafactor')"
ADAFACTOR_NEW = (
    "        if 'Adafactor' not in OPTIMIZERS:\n"
    "            OPTIMIZERS.register_module(name='Adafactor', module=Adafactor)\n"
    "            transformer_optimizers.append('Adafactor')"
)


def patch_mmengine_adafactor() -> bool:
    """Guard mmengine's transformers-Adafactor registration.

    mmengine (<= 0.10.7, including the current upstream) unconditionally
    registers transformers' Adafactor when transformers is installed. Since
    torch >= 2.9 ships torch.optim.Adafactor - already registered under the
    same name - importing mmengine.optim crashes with
    ``KeyError: 'Adafactor is already registered in optimizer ...'`` whenever
    both torch >= 2.9 and transformers are present (both are preinstalled on
    Colab).
    """
    spec = importlib.util.find_spec('mmengine')
    if spec is None or not spec.submodule_search_locations:
        print('[SKIP] mmengine is not installed')
        return False

    builder_file = os.path.join(
        spec.submodule_search_locations[0], 'optim', 'optimizer', 'builder.py'
    )
    with open(builder_file, 'r', encoding='utf-8') as f:
        content = f.read()

    if ADAFACTOR_NEW in content:
        print(f'[OK]   mmengine: Adafactor guard already present ({builder_file})')
        return True
    if ADAFACTOR_OLD not in content:
        print(f'[FAIL] mmengine: Adafactor registration pattern not found in {builder_file}')
        print('       The installed mmengine version may have changed - please check it manually.')
        return False

    with open(builder_file, 'w', encoding='utf-8') as f:
        f.write(content.replace(ADAFACTOR_OLD, ADAFACTOR_NEW))
    print(f'[OK]   mmengine: guarded Adafactor registration ({builder_file})')
    return True


def verify_imports() -> bool:
    ok = True
    for name in ('mmcv', 'mmdet', 'mmcls', 'mmengine'):
        try:
            module = __import__(name)
            print(f'[OK]   import {name} {module.__version__}')
        except Exception as exc:  # noqa: BLE001
            print(f'[FAIL] import {name}: {exc}')
            ok = False
    if ok:
        try:
            # triggers the optimizer registration code paths
            from mmengine.optim.optimizer import OPTIMIZERS  # noqa: F401

            print('[OK]   mmengine optimizer registry builds')
        except Exception as exc:  # noqa: BLE001
            print(f'[FAIL] mmengine optimizer registry: {exc}')
            ok = False
    return ok


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify', action='store_true', help='also verify the patched packages can be imported')
    args = parser.parse_args()

    results = [patch_package(name, maximum) for name, maximum in PACKAGES.items()]
    results.append(patch_mmengine_adafactor())
    if not all(results):
        sys.exit(1)

    if args.verify and not verify_imports():
        sys.exit(1)

    print('Done.')


if __name__ == '__main__':
    main()
