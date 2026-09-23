"""Reusable NEF correction runner. Classical looks are local; neural looks use native GPU or WSL."""
from pathlib import Path
import argparse
import json
import hashlib
import os
import shutil
import subprocess
import sys
import tempfile
import time
from importlib.metadata import version
import cv2
import numpy as np
from PIL import Image, ImageCms, ImageDraw
from underwater_pipeline import (load_source, metadata, preview_size, estimate,
    color_correct, protect_highlights, lab_to_rgb_safe, polish, richer,
    reduce_green_cast, classify_look, restore_source_detail,
    mild_pre_denoise, guided_upsample, polish_neural)

_recipe_cache = {}

CLASSICAL_LOOKS = ('natural', 'pop', 'vivid')
NEURAL_LOOKS = ('spectroformer', 'nu2net')
ALL_LOOKS = CLASSICAL_LOOKS + NEURAL_LOOKS

REPO_ROOT = Path(__file__).resolve().parents[1]
EVAL_ROOT = REPO_ROOT / 'eval'
RUN_UIE = EVAL_ROOT / 'scripts' / 'run_uie_look.py'
SPECTRO_CKPT = EVAL_ROOT / 'repos' / 'spectroformer' / 'checkpoints' / 'best.pth'
NU2_CKPT = EVAL_ROOT / 'repos' / 'uie_benchmark' / 'checkpoints' / 'UIEB' / 'NU2Net.ckpt'
NEURAL_LONG_EDGE = 2048
# Phase A neural finish: pre-denoise → model → guided upsample → detail → light polish.
NEURAL_FINISH = 'phase-a-v1'


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def win_to_wsl(path: Path) -> str:
    resolved = path.resolve()
    s = str(resolved)
    if len(s) >= 2 and s[1] == ':':
        drive = s[0].lower()
        rest = s[2:].replace('\\', '/')
        return f'/mnt/{drive}{rest}'
    return s.replace('\\', '/')


def recipe_id(look):
    if look in _recipe_cache:
        return _recipe_cache[look]
    root = Path(__file__).parent
    value = {
        'look': look,
        'code': [sha256(root / name) for name in ['process_batch.py', 'underwater_pipeline.py']],
        'dependencies': {
            name: version(name)
            for name in ['numpy', 'opencv-python-headless', 'rawpy', 'Pillow']
        },
    }
    if look in NEURAL_LOOKS:
        value['code'].append(sha256(RUN_UIE) if RUN_UIE.is_file() else 'missing-run-uie')
        ckpt = SPECTRO_CKPT if look == 'spectroformer' else NU2_CKPT
        value['weights'] = sha256(ckpt) if ckpt.is_file() else 'missing-weights'
        value['neural_long_edge'] = NEURAL_LONG_EDGE
        value['neural_finish'] = NEURAL_FINISH
    digest = hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()
    _recipe_cache[look] = digest
    return digest


def default_output_path(input_path, look):
    path = input_path.expanduser().resolve()
    name = path.stem if path.is_file() else path.name
    return path.parent / f'{name}-{look}'


def format_duration(seconds):
    seconds = max(0, int(seconds))
    hours, rest = divmod(seconds, 3600)
    minutes, secs = divmod(rest, 60)
    if hours:
        return f'{hours}h{minutes:02d}m{secs:02d}s'
    if minutes:
        return f'{minutes}m{secs:02d}s'
    return f'{secs}s'


def pid_running(pid):
    if pid <= 0:
        return False
    if os.name == 'nt':
        import ctypes
        kernel32 = ctypes.windll.kernel32
        handle = kernel32.OpenProcess(0x1000, False, int(pid))
        if not handle:
            return False
        try:
            code = ctypes.c_ulong()
            if kernel32.GetExitCodeProcess(handle, ctypes.byref(code)):
                return code.value == 259
            return True
        finally:
            kernel32.CloseHandle(handle)
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def lock_message(lock):
    hint = 'Output is locked.'
    pid_file = lock / 'pid'
    if pid_file.is_file():
        try:
            pid = int(pid_file.read_text(encoding='utf-8').strip())
        except ValueError:
            pid = None
        if pid is not None:
            if pid_running(pid):
                hint += f' Recorded PID {pid} appears to be running.'
            else:
                hint += f' Recorded PID {pid} is not running; this lock is likely stale.'
    return hint + ' If the previous run was killed, confirm it stopped before removing .processing.lock'


def look_choices(look):
    if look == 'auto':
        return ('natural', 'spectroformer')
    return (look,)


def output_occupied(path, output, look):
    names = [output / 'reports' / (path.stem + '.json'), output / 'previews' / (path.stem + '.jpg')]
    names.extend(output / (path.stem + '_' + item + '.png') for item in look_choices(look))
    return any(item.exists() for item in names)


def resume_look(path, output, look, recipes):
    if look == 'auto':
        report = output / 'reports' / (path.stem + '.json')
        if not report.is_file():
            return None
        try:
            saved = json.loads(report.read_text(encoding='utf-8'))
        except (OSError, ValueError):
            return None
        item = saved.get('look')
        if item not in recipes:
            return None
        return item if resume_valid(path, output, item, recipes[item]) else None
    return look if resume_valid(path, output, look, recipes[look]) else None


def write_json(path, value):
    temporary = path.with_suffix('.tmp.json')
    temporary.write_text(json.dumps(value, indent=2), encoding='utf-8')
    temporary.replace(path)


def resume_valid(path, output, look, recipe):
    dest = output / (path.stem + '_' + look + '.png')
    report = output / 'reports' / (path.stem + '.json')
    if not (dest.is_file() and report.is_file()):
        return False
    try:
        saved = json.loads(report.read_text(encoding='utf-8'))
        return (saved.get('recipe_id') == recipe and saved.get('source_sha256') == sha256(path)
                and saved.get('output_sha256') == sha256(dest)
                and (output / 'previews' / (path.stem + '.jpg')).is_file())
    except (OSError, ValueError):
        return False


def initial(rgb):
    h, w = rgb.shape[:2]
    params = estimate(preview_size(rgb, 1500))
    lab = np.empty((h, w, 3), np.float32)
    for start in range(0, h, 256):
        lab[start:start + 256] = cv2.cvtColor(color_correct(rgb[start:start + 256], params), cv2.COLOR_RGB2LAB)
    improved = cv2.createCLAHE(clipLimit=1.6, tileGridSize=(12, 8)).apply(np.round(lab[:, :, 0] * 2.55).astype(np.uint8))
    result = np.empty_like(rgb)
    for start in range(0, h, 256):
        tile = lab[start:start + 256]
        tile[:, :, 0] = .5 * tile[:, :, 0] + .5 * improved[start:start + 256] / 2.55
        result[start:start + 256] = np.round(lab_to_rgb_safe(protect_highlights(tile, rgb[start:start + 256])) * 255).astype(np.uint8)
    return result, params


def stats(rgb):
    means, std = cv2.meanStdDev(rgb)
    result = {}
    for i, name in enumerate(('Red', 'Green', 'Blue')):
        hist = cv2.calcHist([rgb], [i], None, [256], [0, 256]).ravel().astype(np.int64)
        result[name] = {'mean_8bit': float(means[i, 0]), 'std_8bit': float(std[i, 0]),
                        'histogram': hist.tolist(), 'zero_percent': float(100 * hist[0] / (rgb.shape[0] * rgb.shape[1])),
                        'highlight_percent': float(100 * hist[255] / (rgb.shape[0] * rgb.shape[1]))}
    return result


def resolve_auto_look(rgb):
    """Underwater → spectroformer; topside → natural."""
    classic = classify_look(rgb)
    return 'spectroformer' if classic == 'vivid' else 'natural'


def native_torch_device() -> str | None:
    """Return 'cuda', 'mps', or None for the current interpreter."""
    try:
        import torch
        if torch.cuda.is_available():
            return 'cuda'
        if getattr(torch.backends, 'mps', None) and torch.backends.mps.is_available():
            return 'mps'
    except Exception:
        return None
    return None


def check_neural_prereqs(look):
    if not RUN_UIE.is_file():
        raise RuntimeError(f'Missing neural runner: {RUN_UIE}')
    ckpt = SPECTRO_CKPT if look == 'spectroformer' else NU2_CKPT
    if not ckpt.is_file():
        raise RuntimeError(
            f'Missing {look} weights at {ckpt}. '
            'Run scripts/fetch_neural_weights.ps1 (Windows) or '
            'scripts/fetch_neural_weights.sh (macOS/Linux) and place checkpoints under eval/repos.'
        )
    if native_torch_device() is not None:
        return
    if shutil.which('wsl') is None:
        raise RuntimeError(
            f'Look {look} needs GPU torch in this venv '
            '(pip install -r requirements-neural.txt on Windows, or '
            'requirements-neural-macos.txt on Apple Silicon) or WSL2 uw_eval. '
            'Neither native CUDA/MPS torch nor wsl.exe is available.'
        )


def run_neural_native(look: str, rgb_path: Path, out_png: Path) -> str:
    """Run run_uie_look.py in the current venv (CUDA or Apple MPS)."""
    completed = subprocess.run(
        [
            sys.executable,
            str(RUN_UIE),
            '--look', look,
            '--input', str(rgb_path),
            '--output', str(out_png),
            '--long-edge', str(NEURAL_LONG_EDGE),
            '--device', 'auto',
        ],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
    )
    log = (completed.stdout or '') + (completed.stderr or '')
    if completed.returncode != 0:
        raise RuntimeError(
            f'Native {look} failed (exit {completed.returncode}).\n{log[-2000:]}'
        )
    if not out_png.is_file():
        raise RuntimeError(f'Native {look} produced no output at {out_png}\n{log[-2000:]}')
    return log


def run_neural_wsl(look: str, rgb_path: Path, out_png: Path) -> str:
    """Invoke WSL uw_eval run_uie_look.py; return stderr+stdout for errors."""
    if shutil.which('wsl') is None:
        raise RuntimeError('wsl.exe not on PATH')
    wsl_runner = win_to_wsl(RUN_UIE)
    wsl_in = win_to_wsl(rgb_path)
    wsl_out = win_to_wsl(out_png)
    inner = (
        'eval "$(/home/jkowall/micromamba/bin/micromamba shell hook -s bash)" && '
        'micromamba activate uw_eval && '
        f'python "{wsl_runner}" --look {look} --input "{wsl_in}" --output "{wsl_out}" '
        f'--long-edge {NEURAL_LONG_EDGE} --device auto'
    )
    completed = subprocess.run(
        ['wsl', '-e', 'bash', '-lc', inner],
        capture_output=True,
        text=True,
    )
    log = (completed.stdout or '') + (completed.stderr or '')
    if completed.returncode != 0:
        raise RuntimeError(
            f'WSL {look} failed (exit {completed.returncode}). '
            f'Ensure micromamba env uw_eval exists with torch+CUDA.\n{log[-2000:]}'
        )
    if not out_png.is_file():
        raise RuntimeError(f'WSL {look} produced no output at {out_png}\n{log[-2000:]}')
    return log


def run_neural(look: str, rgb_path: Path, out_png: Path) -> str:
    """Prefer native CUDA/MPS; fall back to WSL uw_eval on Windows."""
    check_neural_prereqs(look)
    device = native_torch_device()
    if device is not None:
        try:
            log = run_neural_native(look, rgb_path, out_png)
            label = f'windows-{device}' if os.name == 'nt' else f'native-{device}'
            print(f'Neural backend: {label} ({look})', flush=True)
            return log
        except RuntimeError as err:
            if shutil.which('wsl') is None:
                raise
            print(f'Native {device} failed ({err}); trying WSL…', flush=True)
    log = run_neural_wsl(look, rgb_path, out_png)
    print(f'Neural backend: wsl-uw_eval ({look})', flush=True)
    return log


def process_neural(path, output, look, recipe):
    started = time.monotonic()
    source_hash = sha256(path)
    original = load_source(path)
    if look == 'auto':
        look = resolve_auto_look(original)
        recipe = recipe_id(look)
    if look not in NEURAL_LOOKS:
        raise RuntimeError(f'internal: expected neural look, got {look}')

    h, w = original.shape[:2]
    with tempfile.TemporaryDirectory(prefix='uie_') as tmp:
        tmp_dir = Path(tmp)
        # Work at long-edge: mild denoise → model → guided upsample → detail → polish
        work = Image.fromarray(original)
        scale = NEURAL_LONG_EDGE / max(w, h)
        if scale < 1.0:
            nw, nh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
            work = work.resize((nw, nh), Image.BICUBIC)
        work_rgb = mild_pre_denoise(np.asarray(work, dtype=np.uint8))
        rgb_path = tmp_dir / f'{path.stem}_rgb.png'
        enhanced_path = tmp_dir / f'{path.stem}_out.png'
        Image.fromarray(work_rgb).save(rgb_path)
        run_neural(look, rgb_path, enhanced_path)
        enhanced = np.asarray(Image.open(enhanced_path).convert('RGB'), dtype=np.uint8)
        if enhanced.shape[0] != h or enhanced.shape[1] != w:
            enhanced = guided_upsample(enhanced, original)
        result = restore_source_detail(original, enhanced)
        result, mask, particles, black = polish_neural(result)
        cleanup = {
            'candidates': len(particles),
            'pixel_percent': 100 * float(np.count_nonzero(mask)) / mask.size,
            'black_offset': black,
        }
        del mask
        result, green_cast = reduce_green_cast(result)

    dest = output / (path.stem + '_' + look + '.png')
    exif, date = metadata(path, result.shape[1], result.shape[0])
    exif[305] = 'Python underwater correction: ' + look
    icc = ImageCms.ImageCmsProfile(ImageCms.createProfile('sRGB')).tobytes()
    tmp_png = dest.with_suffix('.tmp.png')
    Image.fromarray(result).save(tmp_png, icc_profile=icc, exif=exif)
    with Image.open(tmp_png) as saved:
        assert saved.size == original.shape[1::-1]
        assert saved.info.get('icc_profile') == icc
        actual_date = saved.getexif().get_ifd(34665).get(36867, saved.getexif().get(306))
        assert actual_date == date
        saved.verify()
    decoded = cv2.cvtColor(cv2.imread(str(tmp_png)), cv2.COLOR_BGR2RGB)
    assert np.array_equal(decoded, result)
    del decoded
    tmp_png.replace(dest)

    a = Image.fromarray(preview_size(original, 1000))
    b = Image.fromarray(preview_size(result, 1000))
    comparison = Image.new('RGB', (2000, a.height + 48), '#171b20')
    comparison.paste(a, (0, 48))
    comparison.paste(b, (1000, 48))
    draw = ImageDraw.Draw(comparison)
    draw.text((15, 12), path.stem + ' | source camera JPEG', fill='white')
    draw.text((1015, 12), 'Corrected | ' + look, fill='white')
    comparison.save(output / 'previews' / (path.stem + '.jpg'), quality=94, icc_profile=icc)
    if sha256(path) != source_hash:
        raise RuntimeError('Source changed during processing; output cannot be reused')
    report = {
        'source': path.name, 'output': dest.name, 'look': look,
        'recipe_id': recipe, 'source_sha256': source_hash, 'output_sha256': sha256(dest),
        'representation': 'Full-resolution embedded camera JPEG, assumed sRGB',
        'dimensions': list(result.shape[1::-1]), 'capture_date': str(date),
        'parameters': {
            'neural': look,
            'neural_finish': NEURAL_FINISH,
            'long_edge': NEURAL_LONG_EDGE,
            'pre_denoise': {'d': 5, 'sigma_color': 18.0, 'sigma_space': 4.0},
            'guided_upsample': {'radius': 8, 'eps': 1e-2},
            'detail_restore': {'sigma': 1.6, 'amount': 1.0},
            'polish_neural': {'clarity': 0.10, 'vibrance_gain': 0.08},
        },
        'cleanup': cleanup, 'green_cast': green_cast,
        'before': stats(original), 'after': stats(result),
        'seconds': round(time.monotonic() - started, 2),
        'validation': 'Exact PNG pixel round trip, dimensions, capture date and ICC verified',
    }
    write_json(output / 'reports' / (path.stem + '.json'), report)
    print('Verified', dest.name, flush=True)
    return {'name': dest.name, 'dimensions': report['dimensions'], 'bytes': dest.stat().st_size}


def process_classical(path, output, look, recipe):
    started = time.monotonic()
    source_hash = sha256(path)
    original = load_source(path)
    if look == 'auto':
        look = classify_look(original)
        recipe = recipe_id(look)
    result, params = initial(original)
    cleanup = None
    if look in ('pop', 'vivid'):
        result, mask, particles, black = polish(result)
        cleanup = {'candidates': len(particles), 'pixel_percent': 100 * float(np.count_nonzero(mask)) / mask.size, 'black_offset': black}
        del mask
    if look == 'vivid':
        for start in range(0, len(result), 256):
            result[start:start + 256] = richer(result[start:start + 256])
    # Final adaptive pass removes residual underwater green after chroma boost.
    result, green_cast = reduce_green_cast(result)
    dest = output / (path.stem + '_' + look + '.png')
    exif, date = metadata(path, result.shape[1], result.shape[0])
    exif[305] = 'Python underwater correction: ' + look
    icc = ImageCms.ImageCmsProfile(ImageCms.createProfile('sRGB')).tobytes()
    tmp = dest.with_suffix('.tmp.png')
    Image.fromarray(result).save(tmp, icc_profile=icc, exif=exif)
    with Image.open(tmp) as saved:
        assert saved.size == original.shape[1::-1]
        assert saved.info.get('icc_profile') == icc
        actual_date = saved.getexif().get_ifd(34665).get(36867, saved.getexif().get(306))
        assert actual_date == date
        saved.verify()
    decoded = cv2.cvtColor(cv2.imread(str(tmp)), cv2.COLOR_BGR2RGB)
    assert np.array_equal(decoded, result)
    del decoded
    tmp.replace(dest)
    a = Image.fromarray(preview_size(original, 1000))
    b = Image.fromarray(preview_size(result, 1000))
    comparison = Image.new('RGB', (2000, a.height + 48), '#171b20')
    comparison.paste(a, (0, 48))
    comparison.paste(b, (1000, 48))
    draw = ImageDraw.Draw(comparison)
    draw.text((15, 12), path.stem + ' | source camera JPEG', fill='white')
    draw.text((1015, 12), 'Corrected | ' + look, fill='white')
    comparison.save(output / 'previews' / (path.stem + '.jpg'), quality=94, icc_profile=icc)
    if sha256(path) != source_hash:
        raise RuntimeError('Source changed during processing; output cannot be reused')
    report = {'source': path.name, 'output': dest.name, 'look': look,
              'recipe_id': recipe, 'source_sha256': source_hash, 'output_sha256': sha256(dest),
              'representation': 'Full-resolution embedded camera JPEG, assumed sRGB',
              'dimensions': list(result.shape[1::-1]), 'capture_date': str(date), 'parameters': params,
              'cleanup': cleanup, 'green_cast': green_cast, 'before': stats(original), 'after': stats(result),
              'seconds': round(time.monotonic() - started, 2),
              'validation': 'Exact PNG pixel round trip, dimensions, capture date and ICC verified'}
    write_json(output / 'reports' / (path.stem + '.json'), report)
    print('Verified', dest.name, flush=True)
    return {'name': dest.name, 'dimensions': report['dimensions'], 'bytes': dest.stat().st_size}


def process(path, output, look, recipe):
    if look == 'auto':
        # Peek once to route classical vs neural before full process
        original = load_source(path)
        resolved = resolve_auto_look(original)
        recipe = recipe_id(resolved)
        if resolved in NEURAL_LOOKS:
            return process_neural(path, output, resolved, recipe)
        return process_classical(path, output, resolved, recipe)
    if look in NEURAL_LOOKS:
        return process_neural(path, output, look, recipe)
    return process_classical(path, output, look, recipe)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--input', required=True, type=Path)
    p.add_argument('--output', type=Path, help='Output folder; default is a sibling {input}-{look} directory')
    p.add_argument(
        '--look',
        choices=['natural', 'pop', 'vivid', 'auto', 'spectroformer', 'nu2net'],
        default='auto',
        help='auto (default): spectroformer underwater, natural topside. '
             'spectroformer/nu2net are GPU looks (native CUDA/MPS, or WSL fallback). '
             'vivid/natural/pop are classical CPU looks.',
    )
    p.add_argument('--resume', action='store_true', help='Skip outputs matching source, recipe, dependencies and PNG checksums')
    args = p.parse_args()
    if not args.input.exists():
        p.error('Input does not exist')
    paths = [args.input] if args.input.is_file() else sorted(x for x in args.input.iterdir() if x.suffix.lower() == '.nef')
    if not paths or any(x.suffix.lower() != '.nef' for x in paths):
        p.error('Input must contain NEF files')
    if len({x.stem.casefold() for x in paths}) != len(paths):
        p.error('Duplicate input stems would collide')
    output = args.output.resolve() if args.output else default_output_path(args.input, args.look)
    for directory in [output, output / 'previews', output / 'reports']:
        directory.mkdir(parents=True, exist_ok=True)
    lock = output / '.processing.lock'
    try:
        lock.mkdir()
    except FileExistsError:
        p.error(lock_message(lock))
    try:
        (lock / 'pid').write_text(str(os.getpid()), encoding='utf-8')
        started = time.monotonic()
        recipe = recipe_id(args.look)
        recipes = {item: recipe_id(item) for item in look_choices(args.look)}
        # Preflight neural once
        if args.look in NEURAL_LOOKS or args.look == 'auto':
            try:
                check_neural_prereqs('spectroformer' if args.look == 'auto' else args.look)
            except RuntimeError as err:
                if args.look in NEURAL_LOOKS:
                    p.error(str(err))
                # auto may still hit natural-only folders; warn but continue
                print('Warning:', err, flush=True)
        results = []
        errors = []
        skip = {}
        for path in paths:
            occupied = output_occupied(path, output, args.look)
            if occupied:
                matched = resume_look(path, output, args.look, recipes) if args.resume else None
                if matched:
                    skip[path] = matched
                else:
                    p.error(f'Unverified output collision for {path.name}; choose a fresh output folder')
        processed_times = []

        def write_summary(elapsed):
            processed = sum(1 for item in results if item.get('status') != 'verified-existing')
            resumed = sum(1 for item in results if item.get('status') == 'verified-existing')
            total_bytes = sum(item.get('bytes', 0) for item in results)
            write_json(output / 'batch_summary.json', {
                'look': args.look, 'recipe_id': recipe, 'completed': results, 'errors': errors,
                'processed': processed, 'resumed': resumed, 'output_bytes': total_bytes,
                'seconds': round(elapsed, 2)})

        for index, path in enumerate(paths, 1):
            elapsed = time.monotonic() - started
            remaining = sum(1 for item in paths[index - 1:] if item not in skip)
            eta = f'  ETA {format_duration(sum(processed_times) / len(processed_times) * remaining)}' if processed_times else ''
            print(f'Processing {index}/{len(paths)} {path.name}  elapsed {format_duration(elapsed)}{eta}', flush=True)
            try:
                if path in skip:
                    look = skip[path]
                    dest = output / (path.stem + '_' + look + '.png')
                    results.append({'name': dest.name, 'status': 'verified-existing', 'bytes': dest.stat().st_size})
                    print('Verified existing', dest.name, flush=True)
                else:
                    step = time.monotonic()
                    results.append(process(path, output, args.look, recipes.get(args.look, recipe)))
                    processed_times.append(time.monotonic() - step)
            except Exception as error:
                errors.append({'source': path.name, 'error': str(error)})
                print('Failed', path.name, type(error).__name__, str(error), flush=True)
            write_summary(time.monotonic() - started)
    finally:
        (lock / 'pid').unlink(missing_ok=True)
        lock.rmdir()
    elapsed = time.monotonic() - started
    processed = sum(1 for item in results if item.get('status') != 'verified-existing')
    resumed = sum(1 for item in results if item.get('status') == 'verified-existing')
    total_bytes = sum(item.get('bytes', 0) for item in results)
    print(
        f'Summary: completed {len(results)}/{len(paths)}  processed {processed}  resumed {resumed}  '
        f'failed {len(errors)}  output {total_bytes / 1e9:.2f} GB  elapsed {format_duration(elapsed)}',
        flush=True,
    )
    return bool(errors)


if __name__ == '__main__':
    raise SystemExit(main())
