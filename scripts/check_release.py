"""Inspect an explicit publication snapshot without staging or pushing anything.

Run: python scripts/check_release.py [--verify]
Local outputs are written below outputs/_runs/release-check/.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import unicodedata

ROOT = Path(__file__).resolve().parents[1]
TEXT_SUFFIXES = {'.py', '.md', '.txt', '.yaml', '.yml', '.json', '.ini', '.gitignore'}


def heading_slug(heading: str) -> str:
    """Approximate GitHub's Markdown heading anchors for local link checks."""
    heading = heading.strip().lower()
    heading = ''.join(char for char in heading
                      if char in '-_ ' or char.isalnum()
                      or unicodedata.category(char).startswith('M'))
    return re.sub(r'\s', '-', heading)


def markdown_anchor_findings(path: Path, source: str):
    headings = set()
    seen = {}
    for line in source.splitlines():
        match = re.match(r'^#{1,6}\s+(.+?)\s*$', line)
        if match:
            base = heading_slug(match.group(1))
            count = seen.get(base, 0)
            headings.add(f'{base}-{count}' if count else base)
            seen[base] = count + 1
    for line_number, line in enumerate(source.splitlines(), 1):
        for match in re.finditer(r'\]\(#([^)]*)\)', line):
            if match.group(1) not in headings:
                yield {'file': path.as_posix(), 'line': line_number,
                       'kind': 'broken-heading-link', 'anchor': match.group(1)}


def candidates():
    for name in ('README.md', 'LICENSE', 'THIRD_PARTY_NOTICES.md', 'requirements.txt',
                 'pytest.ini', 'conftest.py', '.gitignore', '项目经历.md'):
        yield ROOT / name
    for folder in ('models', 'utils', 'train', 'eval', 'scripts', 'tests', 'configs',
                   'demo', '.github', 'LICENSES'):
        for path in (ROOT / folder).rglob('*'):
            if path.is_file() and '__pycache__' not in path.parts and path.suffix != '.pyc':
                yield path
    yield ROOT / 'data/README.md'
    yield ROOT / 'data/ETTh1.csv'
    for path in (ROOT / 'docs/figs').glob('*.png'):
        yield path
    for path in (ROOT / 'docs').glob('*.md'):
        yield path
    for path in (ROOT / 'outputs').rglob('*.json'):
        parts = path.relative_to(ROOT / 'outputs').parts
        if '_runs' not in parts and not any(p.startswith('logs') for p in parts):
            yield path
    for name in ('outputs/ckpt/seed42.pt', 'outputs/ckpt_res/seed42.pt', 'outputs/scaler.npz'):
        yield ROOT / name


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify', action='store_true')
    args = parser.parse_args()
    out = ROOT / 'outputs/_runs/release-check'
    out.mkdir(parents=True, exist_ok=True)
    snapshot = Path(tempfile.mkdtemp(prefix='snapshot-', dir=out))
    findings, manifest = [], []
    secrets = re.compile(r'gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,}|'
                         r'sk-[A-Za-z0-9]{30,}|-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----')
    email = re.compile(r'[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}')
    private_path = re.compile(r'[A-Za-z]:[\\/]+Users[\\/]+|' + '/ho' + r'me/[^/\s]+/')
    for path in sorted(set(candidates())):
        rel = path.relative_to(ROOT).as_posix()
        if not path.is_file():
            findings.append({'file': rel, 'kind': 'missing'})
            continue
        raw = path.read_bytes()
        manifest.append({'path': rel, 'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()})
        if len(raw) >= 50 * 1024 * 1024:
            findings.append({'file': rel, 'kind': 'large-file'})
        if path.suffix in TEXT_SUFFIXES or path.name == '.gitignore':
            text = raw.decode('utf-8-sig')
            for number, line in enumerate(text.splitlines(), 1):
                if secrets.search(line):
                    findings.append({'file': rel, 'line': number, 'kind': 'possible-secret'})
                if any(not match.group().endswith('@users.noreply.github.com')
                       for match in email.finditer(line)):
                    findings.append({'file': rel, 'line': number, 'kind': 'possible-personal-email'})
                if private_path.search(line):
                    findings.append({'file': rel, 'line': number, 'kind': 'personal-path'})
            if path.name == 'README.md' and path == ROOT / 'README.md':
                findings.extend(markdown_anchor_findings(Path(rel), text))
        target = snapshot / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
    checks = []
    if args.verify and not findings:
        for command in (
            [sys.executable, '-m', 'pytest', '-o', 'addopts=-p no:cacheprovider', '-q'],
            [sys.executable, '-m', 'demo.test_api'],
            [sys.executable, 'eval/evaluate.py', '--ckpt', 'outputs/ckpt/seed42.pt',
             '--config', 'configs/base.yaml', '--out', 'outputs/figs'],
            [sys.executable, 'eval/evaluate.py', '--ckpt', 'outputs/ckpt_res/seed42.pt',
             '--config', 'configs/base.yaml', '--out', 'outputs/figs_res'],
        ):
            result = subprocess.run(command, cwd=snapshot, capture_output=True)
            log = out / f'check-{len(checks)}.log'
            log.write_bytes(result.stdout + result.stderr)
            checks.append({'command': command[1:], 'exit_code': result.returncode, 'log': log.name})
    report = {'files': manifest, 'total_bytes': sum(p['bytes'] for p in manifest),
              'findings': findings, 'checks': checks,
              'snapshot': snapshot.relative_to(ROOT).as_posix(),
              'scope': 'Candidate snapshot; not a Git index or proof of third-party authorship. '
                       'Tests use the existing interpreter, not a fresh dependency installation.'}
    (out / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({k: v for k, v in report.items() if k != 'files'}, ensure_ascii=True, indent=2))
    print(f'Candidate files: {len(manifest)}')
    return int(bool(findings) or any(c['exit_code'] for c in checks))


if __name__ == '__main__':
    raise SystemExit(main())
