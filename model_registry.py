"""Model registry for SMIT segmentation models.

models.yaml (next to this file) describes every model: network settings, preprocessing,
inference options, labels and the SHA-256 checksum of its weights.

Weights are not stored in the repository; they are distributed separately and installed with
get_weights.py as  weights/<MODEL>.safetensors  in this repository (older releases: weights/<MODEL>.pt).
"""
import hashlib
import json
import os

import yaml

REPO_DIR = os.path.dirname(os.path.abspath(__file__))
REGISTRY_FILE = os.path.join(REPO_DIR, 'models.yaml')
WEIGHTS_DIR = os.path.join(REPO_DIR, 'weights')


def load_registry(path=REGISTRY_FILE):
    with open(path, 'r', encoding='utf-8') as f:
        reg = yaml.safe_load(f)
    return reg['models']


WEIGHT_EXTS = ('.safetensors', '.pt')  # preferred first


def weights_path(name, ext=None):
    """Installed weights of a model: weights/<MODEL>.safetensors, else weights/<MODEL>.pt.
    If neither exists, the .safetensors path (where get_weights.py installs current releases).
    With ext, the path for that file type."""
    if ext:
        return os.path.join(WEIGHTS_DIR, f'{name}{ext}')
    for e in WEIGHT_EXTS:
        p = os.path.join(WEIGHTS_DIR, f'{name}{e}')
        if os.path.isfile(p):
            return p
    return os.path.join(WEIGHTS_DIR, f'{name}{WEIGHT_EXTS[0]}')


def sha256(path, chunk=1 << 22):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(chunk), b''):
            h.update(b)
    return h.hexdigest()


def missing_weights_message(name):
    return (f"Weights for model '{name}' are not installed (expected {weights_path(name)}).\n"
            f"Model weights are distributed separately. Request access from the SMIT maintainers, then run:\n"
            f"  python get_weights.py --model {name} --url <link you received>\n"
            f"or, for a file you already downloaded:\n"
            f"  python get_weights.py --model {name} --file <downloaded file>")


def verify_weights(path, entry, quiet=False):
    """Check the file against the registry checksum. Returns True if it matches (or no checksum is listed).

    A successful check is remembered in <file>.sha256.json (keyed on size and modification time)
    so large files are hashed only once.
    """
    expected = (entry.get('weights', {}).get('sha256') or '').lower()
    if not expected:
        if not quiet:
            print(f"Note: no checksum listed in models.yaml for {os.path.basename(path)}; not verified.")
        return True
    st = os.stat(path)
    stamp = path + '.sha256.json'
    key = {'size': st.st_size, 'mtime': int(st.st_mtime), 'sha256': expected}
    try:
        with open(stamp) as f:
            if json.load(f) == key:
                return True
    except (OSError, ValueError):
        pass
    if not quiet:
        print(f"Verifying checksum of {path} ...")
    if sha256(path) != expected:
        return False
    try:
        with open(stamp, 'w') as f:
            json.dump(key, f)
    except OSError:
        pass  # read-only weights folder: verification simply repeats next time
    return True
