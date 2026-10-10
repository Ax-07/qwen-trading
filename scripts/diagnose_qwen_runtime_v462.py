"""V4.6.2 - read-only Windows local Qwen/Unsloth runtime inventory.

No imports of torch/transformers/unsloth, network calls, model loads or filesystem writes.
Does not display or read tokens, .env files or Hugging Face authentication.
"""
from __future__ import annotations

import argparse
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import subprocess
import sys

PACKAGES = ('torch', 'transformers', 'unsloth', 'unsloth_zoo', 'accelerate', 'peft', 'bitsandbytes', 'safetensors', 'huggingface_hub', 'triton', 'xformers')
MODEL_MARKERS = ('config.json', 'adapter_config.json')
MAX_DEPTH = 4
MAX_VISITS = 6000
MAX_MATCHES = 35


def package_versions() -> dict[str, str | None]:
    versions = {}
    for name in PACKAGES:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def inspect_config(path: Path) -> dict:
    """Only public architecture metadata; never expose full JSON or private keys."""
    entry = {'directory': str(path.parent), 'marker': path.name}
    try:
        if path.stat().st_size > 1_000_000:
            entry['error'] = 'config_too_large'
            return entry
        data = json.loads(path.read_text(encoding='utf-8'))
        if not isinstance(data, dict):
            raise ValueError('configuration is not an object')
        if path.name == 'config.json':
            entry['model_type'] = data.get('model_type') if isinstance(data.get('model_type'), str) else None
            entry['architectures'] = [x for x in data.get('architectures', []) if isinstance(x, str)][:4] if isinstance(data.get('architectures'), list) else []
            entry['quantization_method'] = data.get('quantization_config', {}).get('quant_method') if isinstance(data.get('quantization_config'), dict) else None
        else:
            entry['adapter_type'] = data.get('peft_type') if isinstance(data.get('peft_type'), str) else None
            # Do not report base_model_name_or_path: may contain local identifying paths.
        entry['has_weight_files'] = any(path.parent.glob('*.safetensors')) or any(path.parent.glob('*.gguf')) or any(path.parent.glob('*.bin'))
    except (OSError, ValueError, UnicodeError) as exc:
        entry['error'] = type(exc).__name__
    return entry


def scan_checkpoints(roots: list[Path], *, max_depth: int = MAX_DEPTH, max_visits: int = MAX_VISITS) -> tuple[list[dict], dict]:
    """Bounded local directory walk. No symlink traversal. Paths preselected by caller."""
    matches = []
    visits = 0
    missing = []
    for root in roots:
        if not root.is_dir():
            missing.append(str(root))
            continue
        stack = [(root, 0)]
        while stack and visits < max_visits and len(matches) < MAX_MATCHES:
            folder, depth = stack.pop()
            visits += 1
            try:
                for filename in MODEL_MARKERS:
                    file = folder / filename
                    if file.is_file() and not file.is_symlink():
                        matches.append(inspect_config(file))
                if depth < max_depth:
                    dirs = [x for x in folder.iterdir() if x.is_dir() and not x.is_symlink() and x.name not in {'.git', 'node_modules', '.venv', '.venv-unsloth', '__pycache__'}]
                    stack.extend((x, depth + 1) for x in reversed(sorted(dirs, key=lambda z: z.name)))
            except (OSError, PermissionError):
                continue
    return matches, {'directories_examined': visits, 'truncated': visits >= max_visits or len(matches) >= MAX_MATCHES, 'missing_roots': missing}


def gpu_info() -> dict:
    try:
        p = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total,driver_version', '--format=csv,noheader'], text=True, capture_output=True, timeout=8, check=False)
        return {'available': p.returncode == 0, 'gpu_lines': [x.strip() for x in p.stdout.splitlines()][:8] if p.returncode == 0 else [], 'status': p.returncode}
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return {'available': False, 'gpu_lines': []}


def default_roots(project: Path) -> list[Path]:
    roots = [project / 'models', project / 'model', project / 'outputs', project / 'checkpoints', project / 'adapters', project / 'data' / 'models', project / 'results']
    hf_home = os.environ.get('HF_HOME')
    home = Path.home()
    roots.append(Path(hf_home).expanduser() / 'hub' if hf_home else home / '.cache' / 'huggingface' / 'hub')
    transformers_cache = os.environ.get('TRANSFORMERS_CACHE')
    if transformers_cache:
        roots.append(Path(transformers_cache).expanduser())
    return list(dict.fromkeys(roots))


def diagnose(project: Path, extra_roots: list[Path] | None = None, include_gpu: bool = True) -> dict:
    roots = default_roots(project) + (extra_roots or [])
    matches, scan = scan_checkpoints(roots)
    envs = {}
    for venv in ('.venv', '.venv-unsloth'):
        py = project / venv / 'Scripts' / 'python.exe'
        envs[venv] = {'python_exists': py.is_file(), 'path': str(py)}
    return {
        'version': 'V4.6.2-diagnostic', 'read_only': True,
        'current_python': sys.executable, 'python_version': sys.version.split()[0],
        'current_environment_packages': package_versions(),
        'virtual_environments': envs,
        'gpu': gpu_info() if include_gpu else {'skipped': True},
        'checkpoint_candidates': matches, 'scan': scan,
        'notes': ['A candidate directory does not prove the weights are loadable.', 'Packages listed are from the Python interpreter running this script.', 'No model loaded; no training; no orders.'],
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--project', type=Path, default=Path(__file__).resolve().parent.parent)
    ap.add_argument('--scan-root', type=Path, action='append', default=[], help='Additional known model directory (bounded scan)')
    ap.add_argument('--skip-gpu', action='store_true')
    args = ap.parse_args()
    result = diagnose(args.project.expanduser().resolve(), [r.expanduser().resolve() for r in args.scan_root], not args.skip_gpu)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    print('DIAGNOSTIC ONLY — no filesystem modifications, network requests, inference, or orders.')


if __name__ == '__main__':
    main()
