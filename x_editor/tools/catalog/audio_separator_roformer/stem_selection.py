"""Select declared separator stems without inferring speaker identity."""

from pathlib import Path
import re


_VOCAL_POLICIES = frozenset({'vocals', 'vocal', 'speech', 'voice', 'foreground'})
_BACKGROUND_POLICIES = frozenset({'instrumental', 'accompaniment', 'background', 'music', 'no_vocals'})
_ROLE_LABELS = {'vocals': 'vocals', 'vocal': 'vocals', 'speech': 'vocals', 'voice': 'vocals',
                'instrumental': 'instrumental', 'accompaniment': 'instrumental',
                'no_vocals': 'instrumental', 'background': 'instrumental', 'music': 'instrumental'}


def stem_role(path: Path) -> str | None:
    name = path.stem.lower().replace('-', '_').replace(' ', '_')
    labels = re.findall(r'\(([^()]*)\)', name)
    if labels:
        return _ROLE_LABELS.get(labels[-1])
    for label in sorted(_ROLE_LABELS, key=len, reverse=True):
        if name == label or name.endswith('_' + label):
            return _ROLE_LABELS[label]
    return None


def select_stem(stem_paths: list[Path], policy: str) -> Path:
    candidates = list(dict.fromkeys(stem_paths))
    normalized = policy.strip().lower().replace('-', '_').replace(' ', '_')
    if not candidates or not normalized:
        raise ValueError('stem_selection_requires_candidates_and_policy')
    if normalized in {'first', 'primary', '0'}:
        return candidates[0]
    if normalized in {'last', 'residual', '_1'}:
        return candidates[-1]
    if normalized in _VOCAL_POLICIES | _BACKGROUND_POLICIES:
        wanted = 'vocals' if normalized in _VOCAL_POLICIES else 'instrumental'
        matches = [path for path in candidates if stem_role(path) == wanted]
    else:
        matches = [path for path in candidates
                   if normalized in path.stem.lower().replace('-', '_').replace(' ', '_')]
    if len(matches) != 1:
        raise ValueError(f'stem_selection_requires_unique_label: policy={policy!r}, matches={len(matches)}')
    return matches[0]
