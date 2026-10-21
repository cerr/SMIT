"""Install weights for a SMIT model.

Weights are shared separately from this repository (e.g. a Box link sent on request).
This script downloads or copies them to weights/<MODEL>.pt in this repository
(e.g. weights/CT_ThoraxOARs_SMITplus.pt) and verifies the SHA-256 checksum listed in models.yaml.

Examples:
    python get_weights.py --list
    python get_weights.py --model CT_LungGTV_SMITplus --url "https://mskcc.box.com/shared/static/....gz"
    python get_weights.py --model CT_LungGTV_SMITplus --file D:\\downloads\\model_50k_lung_gtv.pt
    python get_weights.py --model CT_LungGTV_SMITplus          # only if models.yaml lists a source

Where the weights come from, in order:
  1. --url / --file given on the command line (e.g. a link sent by the maintainers);
  2. otherwise the model's weights.source in models.yaml, if it has one:
       {type: url, url: <direct download link>}
       {type: huggingface, repo: <org/repo>, file: <file in repo>, revision: <tag or commit>}
     A gated Hugging Face repository needs access approval and a login (hf auth login, or HF_TOKEN).

Accepted inputs: a PyTorch checkpoint (.pt/.pth), a .tar.gz / .tgz / .tar / .zip archive that contains
one, or a gzipped checkpoint.
"""
import argparse
import gzip
import os
import shutil
import sys
import tarfile
import tempfile
import urllib.request
import zipfile

from model_registry import load_registry, weights_path, sha256, verify_weights

CKPT_EXT = ('.pt', '.pth', '.ckpt')


def download(url, dest):
    print(f"Downloading {url}")
    req = urllib.request.Request(url, headers={'User-Agent': 'SMIT-get_weights'})
    with urllib.request.urlopen(req) as r, open(dest, 'wb') as f:
        total = int(r.headers.get('Content-Length') or 0)
        done, last = 0, -1
        while True:
            b = r.read(1 << 22)
            if not b:
                break
            f.write(b)
            done += len(b)
            pct = int(100 * done / total) if total else -1
            if pct != last and (pct % 5 == 0 or pct < 0):
                print(f"  {done / 2**20:,.0f} MB" + (f" ({pct}%)" if total else ""), flush=True)
                last = pct
    print(f"Downloaded {done / 2**20:,.0f} MB")


def is_torch_zip(path):
    """torch.save() writes a zip archive containing <name>/data.pkl."""
    try:
        with zipfile.ZipFile(path) as z:
            return any(n.endswith('data.pkl') for n in z.namelist())
    except zipfile.BadZipFile:
        return False


def pick(names, wanted):
    """Choose the checkpoint among archive members."""
    ck = [n for n in names if n.lower().endswith(CKPT_EXT)]
    for pref in (wanted, 'model_final.pt'):
        hit = [n for n in ck if os.path.basename(n) == pref]
        if hit:
            return hit[0]
    if len(ck) == 1:
        return ck[0]
    sys.exit(f"Could not tell which file in the archive is the checkpoint: {ck or names}")


def extract_checkpoint(src, workdir, wanted):
    """Return a path to the checkpoint file inside src (which may be an archive)."""
    if is_torch_zip(src):
        return src  # already a checkpoint
    with open(src, 'rb') as f:
        magic = f.read(4)
    if magic[:2] == b'\x1f\x8b' or tarfile.is_tarfile(src):
        if tarfile.is_tarfile(src):  # .tar, .tar.gz, .tgz
            with tarfile.open(src) as t:
                members = [m for m in t.getmembers() if m.isfile()]
                m = next(m for m in members if m.name == pick([m.name for m in members], wanted))
                out = os.path.join(workdir, 'ckpt')
                with t.extractfile(m) as fi, open(out, 'wb') as fo:  # stream copy: no path traversal
                    shutil.copyfileobj(fi, fo, 1 << 22)
                print(f"Extracted {m.name} from archive")
                return out
        out = os.path.join(workdir, 'ckpt')  # gzipped single checkpoint
        with gzip.open(src, 'rb') as fi, open(out, 'wb') as fo:
            shutil.copyfileobj(fi, fo, 1 << 22)
        return out
    if magic[:2] == b'PK':  # ordinary zip archive
        with zipfile.ZipFile(src) as z:
            name = pick(z.namelist(), wanted)
            out = os.path.join(workdir, 'ckpt')
            with z.open(name) as fi, open(out, 'wb') as fo:
                shutil.copyfileobj(fi, fo, 1 << 22)
            print(f"Extracted {name} from archive")
            return out
    return src  # already a checkpoint


def fetch_from_source(source, model, tmp):
    """Download the weights from the model's weights.source in models.yaml; returns the local file."""
    kind = source.get('type')
    if kind == 'url':
        raw = os.path.join(tmp, 'download')
        download(source['url'], raw)
        return raw
    if kind == 'huggingface':
        return fetch_huggingface(source, model, tmp)
    sys.exit(f"Unknown weights source type {kind!r} for {model} in models.yaml (use 'url' or 'huggingface').")


def fetch_huggingface(source, model, tmp):
    """Download one file from a (possibly gated) Hugging Face repository with the user's login."""
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        sys.exit("Downloading from Hugging Face needs huggingface_hub: pip install huggingface_hub")
    repo, fname = source['repo'], source.get('file', f'{model}.pt')
    revision = source.get('revision')
    page = f"https://huggingface.co/{repo}"
    print(f"Downloading {fname} from Hugging Face {repo}" + (f" @ {revision}" if revision else ''))
    try:
        return hf_hub_download(repo_id=repo, filename=fname, revision=revision, local_dir=tmp)
    except Exception as e:  # errors differ across huggingface_hub versions; classify by name / status
        name, text = type(e).__name__, str(e)
        if 'Gated' in name or '403' in text:
            sys.exit(f"Access to {repo} has not been granted to this account yet. Request access at {page}, "
                     f"then log in with  hf auth login  (or set HF_TOKEN) and try again.")
        if 'RepositoryNotFound' in name or '401' in text:
            sys.exit(f"Cannot open {repo}: it does not exist or you are not logged in. If access was granted, "
                     f"log in with  hf auth login  (or set HF_TOKEN). Model page: {page}")
        if 'EntryNotFound' in name or 'RevisionNotFound' in name:
            sys.exit(f"{fname} (revision {revision or 'main'}) was not found in {repo}: check weights.source in models.yaml.")
        raise


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--model', help='model name from models.yaml')
    src = ap.add_mutually_exclusive_group()
    src.add_argument('--url', help='download link you received for this model')
    src.add_argument('--file', help='already-downloaded checkpoint or archive')
    ap.add_argument('--force', action='store_true', help='replace weights that are already installed')
    ap.add_argument('--list', action='store_true', help='list models and whether their weights are installed')
    args = ap.parse_args()

    reg = load_registry()
    if args.list:
        for name in reg:
            p = weights_path(name)
            print(f"{name:28s} {'installed: ' + p if os.path.isfile(p) else 'not installed'}")
        return
    if not args.model:
        ap.error('--model is required (or --list)')
    if args.model not in reg:
        sys.exit(f"Unknown model {args.model}. Known: {', '.join(reg)}")
    entry = reg[args.model]
    source = None if (args.url or args.file) else entry.get('weights', {}).get('source')
    if not (args.url or args.file or source):
        sys.exit(f"The weights for {args.model} are shared on request. Use the link you received:\n"
                 f"  python get_weights.py --model {args.model} --url <link>\n"
                 f"or a file you already downloaded:\n"
                 f"  python get_weights.py --model {args.model} --file <downloaded file>")
    dest = weights_path(args.model)
    dest_dir = os.path.dirname(dest)
    if os.path.exists(dest) and not args.force:
        sys.exit(f"Already installed: {dest} (use --force to replace)")
    os.makedirs(dest_dir, exist_ok=True)

    with tempfile.TemporaryDirectory(dir=dest_dir, prefix='.get_weights_') as tmp:
        if args.url:
            raw = os.path.join(tmp, 'download')
            download(args.url, raw)
        elif source:
            raw = fetch_from_source(source, args.model, tmp)
        else:
            raw = args.file
            if not os.path.isfile(raw):
                sys.exit(f"No such file: {raw}")
        ckpt = extract_checkpoint(raw, tmp, os.path.basename(dest))
        expected = (entry.get('weights', {}).get('sha256') or '').lower()
        actual = sha256(ckpt)
        if expected and actual != expected:
            sys.exit(f"Checksum mismatch: got {actual}, models.yaml lists {expected}.\n"
                     f"This is not the expected {args.model} weights file (different version or a corrupted download). Nothing was installed.")
        if ckpt == raw:
            shutil.copyfile(raw, dest + '.part')
        else:
            shutil.move(ckpt, dest + '.part')
        os.replace(dest + '.part', dest)
    verify_weights(dest, entry, quiet=True)  # records the verification stamp
    print(f"Installed {args.model} weights: {dest}")
    print(f"SHA-256: {actual}" + ("  (matches models.yaml)" if expected else
                                  "\nmodels.yaml lists no checksum for this model yet; add the value above to weights.sha256."))


if __name__ == '__main__':
    main()
