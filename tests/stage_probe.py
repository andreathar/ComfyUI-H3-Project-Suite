"""H3 Stage Save/Load must return exactly what the encoder produced.

The sampling run trusts the stage completely: a dropped minimax_refs
entry, a tuple turned into a list or a cast dtype would sample a
different shot without any error. Run from the ComfyUI checkout:

    .venv/bin/python custom_nodes/ComfyUI-H3-Project-Suite/tests/stage_probe.py
"""

import importlib
import os
import sys
import tempfile
import types

_PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(_PKG)))
sys.path.insert(0, os.path.dirname(_PKG))

_OUT = tempfile.mkdtemp()
fp = types.ModuleType("folder_paths")
fp.get_output_directory = lambda: _OUT
sys.modules["folder_paths"] = fp

import torch  # noqa: E402
import comfy.nested_tensor as nt  # noqa: E402

N = importlib.import_module(os.path.basename(_PKG) + ".nodes")


def same(a, b, path="root"):
    assert type(a) is type(b), (path, type(a), type(b))
    if torch.is_tensor(a):
        assert a.dtype == b.dtype and a.shape == b.shape and torch.equal(a, b), path
    elif isinstance(a, nt.NestedTensor):
        for i, (x, y) in enumerate(zip(a.unbind(), b.unbind(), strict=True)):
            same(x, y, "%s~%d" % (path, i))
    elif isinstance(a, dict):
        assert a.keys() == b.keys(), path
        for k in a:
            same(a[k], b[k], path + "." + k)
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b), path
        for i, (x, y) in enumerate(zip(a, b)):
            same(x, y, "%s[%d]" % (path, i))
    else:
        assert a == b, (path, a, b)


cond = [[torch.randn(1, 12, 8), {
    "pooled_output": None,
    "minimax_token_tags": torch.arange(12),
    "minimax_refs": [
        {"kind": "image", "latent_h": 2, "latent_w": 3, "latent": torch.randn(1, 24, 1, 2, 3)},
        {"kind": "video", "latent_t": 2, "ref_audio_t": 0, "audio_latent": None,
         "latent": torch.randn(1, 24, 2, 2, 3, dtype=torch.bfloat16)},
    ],
    "shape": (1, 2.5, "x"),
    "flag": True,
}]]
latent = {"samples": nt.NestedTensor([torch.randn(1, 24, 2, 2, 3), torch.randn(1, 32, 2, 5)]),
          "noise_mask": nt.NestedTensor([torch.ones(1, 1, 2, 2, 3), torch.ones(1, 32, 2, 5)])}

path, = N.H3StageSave().save(cond, latent, "h3_stages/probe")
assert path == os.path.join(_OUT, "h3_stages", "probe.safetensors"), path
cond2, latent2 = N.H3StageLoad().load("h3_stages/probe")
same(cond, cond2)
same(latent, latent2)

for bad in ("../escape", "h3_stages/../../escape"):
    try:
        N.H3StageSave().save(cond, latent, bad)
    except ValueError:
        pass
    else:
        raise AssertionError("escaping name accepted: " + bad)

try:
    N.H3StageSave().save([[torch.zeros(1), {"bad": object()}]], latent, "h3_stages/bad")
except TypeError:
    pass
else:
    raise AssertionError("unstageable value accepted")

N._st_save({"x": torch.zeros(1)}, os.path.join(_OUT, "h3_stages", "plain.safetensors"))
try:
    N.H3StageLoad().load("h3_stages/plain")
except ValueError:
    pass
else:
    raise AssertionError("non-stage file accepted")

print("stage_probe: PASS")
