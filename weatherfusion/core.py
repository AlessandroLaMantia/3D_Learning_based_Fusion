"""Official AMG-Fuse checkpoint loading and provenance helpers."""
from pathlib import Path
import hashlib
import torch
from .vendor.amgfuse import MSAKNet

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'pretrained/amgfuse_haze.pth'

def sha(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()

def load_model():
    torch.set_num_threads(4)
    model = MSAKNet(use_restormer=False)
    state = torch.load(BASE, map_location='cpu', weights_only=True)['AWF']
    state = {k.removeprefix('module.'): v for k, v in state.items()}
    # Restormer is the authors' training-only supervisor; all inference keys are required.
    expected = model.state_dict()
    missing = set(expected) - set(state)
    extra = set(state) - set(expected)
    if missing or any(not k.startswith('res.') for k in extra):
        raise ValueError(f'Checkpoint incompatibile: missing={missing}, extra={sorted(extra)[:5]}')
    model.load_state_dict({k: state[k] for k in expected}, strict=True)
    return model.eval()
