"""Rebuild Meta-format DINOv3 ConvNeXt checkpoints from timm's ungated re-host.

Meta gates the DINOv3 downloads. timm re-hosts the *same tensors* ungated on the
Hugging Face Hub (``timm/convnext_<size>.dinov3_lvd1689m``) under timm's own
module names. This script downloads that file, renames the keys into Meta's
layout, and writes a plain state-dict ``.pth`` that

    torch.hub.load("facebookresearch/dinov3", model="dinov3_convnext_<size>",
                   weights=<pth>, trust_repo=True)

loads with ``strict=True`` — which is what ``cropcounter.dinov3_pyramid`` does.

The name map is 1:1 and total::

    stem.{0,1}.*                     -> downsample_layers.0.{0,1}.*
    stages.i.downsample.{0,1}.*      -> downsample_layers.i.{0,1}.*      (i >= 1)
    stages.i.blocks.j.conv_dw.*      -> stages.i.j.dwconv.*
    stages.i.blocks.j.mlp.fc1.*      -> stages.i.j.pwconv1.*
    stages.i.blocks.j.mlp.fc2.*      -> stages.i.j.pwconv2.*
    stages.i.blocks.j.norm.*         -> stages.i.j.norm.*
    stages.i.blocks.j.gamma          -> stages.i.j.gamma
    head.norm.*                      -> norm.*  AND  norms.3.*

The last line is the only one-to-many entry: Meta's ``ConvNeXt`` registers its
final LayerNorm as ``self.norm`` and then aliases the *same module* into
``self.norms[3]``, so the checkpoint carries it twice. Meta's state dict
therefore has exactly two more tensors than timm's.

Shape agreement proves nothing — a wrong-but-same-shape permutation loads
strictly and silently ruins the features. ``--verify`` is the real check: it
pushes the same real images through timm's own model and through the hub model
built from the converted file and compares every stage map, then repeats the
comparison with two *deliberately corrupted* mappings to show the check can
actually fail.

Usage::

    python convert_timm_dinov3.py --size tiny --check --verify
"""
from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch

# --- repo imports ----------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT / "src") not in sys.path:  # running the script straight from a clone
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from cropcounter.weights import WEIGHT_FILES  # noqa: E402

#: Backbone size -> the timm model name re-hosting Meta's tensors on the HF Hub.
TIMM_MODELS = {
    "tiny": "convnext_tiny.dinov3_lvd1689m",
    "small": "convnext_small.dinov3_lvd1689m",
    "base": "convnext_base.dinov3_lvd1689m",
}

#: Files to try in the HF repo, in order. safetensors first: no pickle, and it
#: is what timm ships for these three.
HF_CANDIDATE_FILES = ("model.safetensors", "pytorch_model.bin")

HF_URL = "https://huggingface.co/timm/{model}/resolve/main/{filename}"

#: Meta's layer-scale init. The "forgot to map gamma" negative control sets
#: every gamma back to this, which is what an unmapped LayerScale would be.
LAYER_SCALE_INIT = 1e-6

#: Where the hub repo is cached, for the network-free strict-load check.
HUB_DIR = Path(torch.hub.get_dir()) / "facebookresearch_dinov3_main"


# ---------------------------------------------------------------------------
# The name map. Pure, string-in/string-out, unit-tested.
# ---------------------------------------------------------------------------

def map_timm_key(key: str) -> List[str]:
    """Meta key name(s) for one timm key name.

    Returns a list because ``head.norm.*`` lands on two Meta keys (Meta aliases
    the final norm as both ``norm`` and ``norms.3``). Every other key maps 1:1.

    Raises:
        KeyError: for any key the map does not cover — a silently dropped
            tensor is exactly the failure this script exists to prevent, so an
            unknown name is loud rather than skipped.
    """
    if key.startswith("stem."):
        return ["downsample_layers.0." + key[len("stem."):]]

    if key.startswith("head."):
        if not key.startswith("head.norm."):
            raise KeyError(f"unmapped timm key {key!r} (only head.norm.* is expected)")
        suffix = key[len("head.norm."):]
        # Meta stores the final norm twice, under both names, same tensor.
        return [f"norm.{suffix}", f"norms.3.{suffix}"]

    parts = key.split(".")
    if parts[0] != "stages" or len(parts) < 4 or not parts[1].isdigit():
        raise KeyError(f"unmapped timm key {key!r}")
    stage = parts[1]

    if parts[2] == "downsample":
        if stage == "0":
            # timm's stage-0 downsample is an Identity and carries no tensors;
            # the stem is the stage-0 downsample in Meta's layout. A real key
            # here would collide with stem.* — refuse rather than overwrite.
            raise KeyError(f"unexpected stage-0 downsample key {key!r}; the stem covers it")
        return ["downsample_layers." + stage + "." + ".".join(parts[3:])]

    if parts[2] != "blocks" or not parts[3].isdigit():
        raise KeyError(f"unmapped timm key {key!r}")
    block, rest = parts[3], parts[4:]
    if not rest:
        raise KeyError(f"unmapped timm key {key!r}")

    if rest[0] == "mlp":
        if len(rest) < 2 or rest[1] not in ("fc1", "fc2"):
            raise KeyError(f"unmapped timm key {key!r}")
        renamed = ["pwconv1" if rest[1] == "fc1" else "pwconv2", *rest[2:]]
    elif rest[0] == "conv_dw":
        renamed = ["dwconv", *rest[1:]]
    elif rest[0] in ("norm", "gamma"):
        renamed = list(rest)
    else:
        raise KeyError(f"unmapped timm key {key!r}")

    return [f"stages.{stage}.{block}." + ".".join(renamed)]


def map_timm_keys(keys: Iterable[str]) -> Dict[str, List[str]]:
    """Map a whole timm key list, refusing collisions.

    Args:
        keys: timm ``state_dict`` key names.

    Returns:
        ``{timm_key: [meta_key, ...]}`` in input order.

    Raises:
        KeyError: an unmapped name.
        ValueError: two timm keys claiming the same Meta key.
    """
    mapping: Dict[str, List[str]] = {}
    claimed: Dict[str, str] = {}
    for key in keys:
        targets = map_timm_key(key)
        for target in targets:
            if target in claimed:
                raise ValueError(
                    f"Meta key {target!r} claimed by both {claimed[target]!r} and {key!r}"
                )
            claimed[target] = key
        mapping[key] = targets
    return mapping


def convert_state_dict(timm_sd: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Rename a timm DINOv3-ConvNeXt state dict into Meta's layout."""
    mapping = map_timm_keys(timm_sd.keys())
    meta_sd: Dict[str, torch.Tensor] = {}
    for source, targets in mapping.items():
        for target in targets:
            meta_sd[target] = timm_sd[source]
    return meta_sd


# ---------------------------------------------------------------------------
# Download + load
# ---------------------------------------------------------------------------

def download_timm_file(size: str, cache_dir: Path) -> Path:
    """Fetch timm's re-hosted checkpoint, returning the local path.

    Prefers ``huggingface_hub`` when it is importable (it handles resume,
    etags, and xet); otherwise plain HTTPS against the ``resolve/main`` URL.
    Either way the file lands under ``cache_dir``.
    """
    model = TIMM_MODELS[size]
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        hf_hub_download = None

    last_error: Optional[Exception] = None
    for filename in HF_CANDIDATE_FILES:
        if filename.endswith(".safetensors"):
            try:
                import safetensors.torch  # noqa: F401
            except ImportError:
                continue  # cannot read it; fall through to the .bin
        if hf_hub_download is not None:
            try:
                return Path(
                    hf_hub_download(
                        repo_id=f"timm/{model}", filename=filename, cache_dir=str(cache_dir)
                    )
                )
            except Exception as exc:  # noqa: BLE001 - try the next candidate file
                last_error = exc
                continue
        target = cache_dir / model / filename
        if target.is_file():
            return target
        target.parent.mkdir(parents=True, exist_ok=True)
        url = HF_URL.format(model=model, filename=filename)
        try:
            with urllib.request.urlopen(url) as response:  # noqa: S310 - fixed https host
                partial = target.with_suffix(target.suffix + ".part")
                with open(partial, "wb") as handle:
                    while chunk := response.read(1 << 20):
                        handle.write(chunk)
                partial.replace(target)
            return target
        except Exception as exc:  # noqa: BLE001 - try the next candidate file
            last_error = exc
            continue
    raise RuntimeError(f"could not download any of {HF_CANDIDATE_FILES} for timm/{model}: {last_error!r}")


def load_checkpoint(path: Path) -> Dict[str, torch.Tensor]:
    """Read a ``.safetensors`` or ``.bin`` checkpoint into a plain tensor dict."""
    if path.suffix == ".safetensors":
        from safetensors.torch import load_file

        return dict(load_file(str(path), device="cpu"))
    obj = torch.load(str(path), map_location="cpu", weights_only=True)
    if isinstance(obj, dict) and "state_dict" in obj:
        obj = obj["state_dict"]
    return dict(obj)


def build_hub_model(size: str, weights: Path) -> torch.nn.Module:
    """Load Meta's real hub constructor against ``weights``, strictly.

    ``_make_dinov3_convnext`` calls ``load_state_dict(..., strict=True)``, so a
    successful return here *is* the strict-load check. The hub repo is read
    from the local torch-hub cache, so no network is touched; ``source="local"``
    is used when that cache exists (matching what ``dinov3_pyramid`` gets from
    ``source="github"`` on an already-cached clone).

    One trap: Meta turns ``weights=<path>`` into a ``file://`` URL and hands it
    to ``load_state_dict_from_url``, which caches by *basename* under
    ``hub/checkpoints/``. Re-converting the same filename would then silently
    load the previous conversion. The stale copy is dropped first.
    """
    stale = Path(torch.hub.get_dir()) / "checkpoints" / Path(weights).name
    if stale.is_file() and stale.resolve() != Path(weights).resolve():
        stale.unlink()
    kwargs = dict(model=f"dinov3_convnext_{size}", weights=str(weights), trust_repo=True)
    if HUB_DIR.is_dir():
        return torch.hub.load(repo_or_dir=str(HUB_DIR), source="local", **kwargs)
    return torch.hub.load(repo_or_dir="facebookresearch/dinov3", source="github", **kwargs)


# ---------------------------------------------------------------------------
# Verification: same images through both models, then the same check corrupted
# ---------------------------------------------------------------------------

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

#: Real images the verification runs on: three fish frames + one wheat plot.
DEFAULT_IMAGE_DIRS = (
    (Path("data/cfd17_smoke/val/images"), 3),
    (Path("data/val/images"), 1),
)

#: (height, width) the images are resized to. Both multiples of 32, and
#: deliberately non-square so a transposed stage map cannot pass unnoticed.
VERIFY_SIZE = (256, 384)

#: Every stage's max-abs difference must be at or below this fraction of the
#: reference map's own scale.
REL_TOLERANCE = 1e-4


def load_images(
    dirs: Sequence[Tuple[Path, int]] = DEFAULT_IMAGE_DIRS,
    size: Tuple[int, int] = VERIFY_SIZE,
    root: Path = _REPO_ROOT,
) -> Tuple[torch.Tensor, List[str]]:
    """ImageNet-normalised NCHW batch of real images, plus their filenames."""
    import numpy as np
    import torch.nn.functional as F
    from PIL import Image

    tensors, names = [], []
    for directory, count in dirs:
        folder = root / directory
        if not folder.is_dir():
            continue
        for path in sorted(folder.glob("*.jpg"))[:count]:
            array = np.array(Image.open(path).convert("RGB"), dtype=np.uint8)
            chw = torch.from_numpy(array).permute(2, 0, 1).float().div_(255.0).unsqueeze(0)
            tensors.append(F.interpolate(chw, size=size, mode="bilinear", align_corners=False))
            names.append(path.name)
    if not tensors:
        raise FileNotFoundError(f"no .jpg images found under {[str(d) for d, _ in dirs]}")
    batch = torch.cat(tensors, dim=0)
    mean = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1)
    return (batch - mean) / std, names


def compare(ours: torch.Tensor, reference: torch.Tensor) -> Dict[str, float]:
    """Max-abs / mean-abs difference, and max-abs as a fraction of the reference scale."""
    if ours.shape != reference.shape:
        return {"shape_mismatch": 1.0, "max_abs": float("inf"), "rel": float("inf")}
    diff = (ours.float() - reference.float()).abs()
    scale = reference.float().abs().max().item()
    return {
        "max_abs": diff.max().item(),
        "mean_abs": diff.mean().item(),
        "ref_max_abs": scale,
        "rel": diff.max().item() / scale if scale > 0 else float("inf"),
    }


def stage_maps(hub_model: torch.nn.Module, x: torch.Tensor) -> Tuple[List[torch.Tensor], torch.Tensor]:
    """Raw stage maps (norm off) and the stage-3 map with Meta's final norm applied."""
    with torch.no_grad():
        raw = hub_model.get_intermediate_layers(x, n=[0, 1, 2, 3], reshape=True, norm=False)
        normed = hub_model.get_intermediate_layers(x, n=[0, 1, 2, 3], reshape=True)
    return list(raw), normed[3]


def corrupt_swap_blocks(meta_sd: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Negative control 1: swap two adjacent stage-2 blocks' tensors.

    The key SET is untouched, so the strict load still succeeds and every shape
    still matches — only the activations move. A check that cannot catch this
    is not checking anything.
    """
    out = dict(meta_sd)
    a, b = "stages.2.1.", "stages.2.2."
    for key in list(meta_sd):
        if key.startswith(a):
            out[key] = meta_sd[b + key[len(a):]]
        elif key.startswith(b):
            out[key] = meta_sd[a + key[len(b):]]
    return out


def corrupt_drop_gamma(meta_sd: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Negative control 2: leave every LayerScale gamma at its init value.

    This is what "I forgot to map ``.gamma``" looks like: the key is present
    (so strict loading is happy) but carries the constructor's 1e-6 instead of
    the pretrained scale.
    """
    return {
        key: (torch.full_like(value, LAYER_SCALE_INIT) if key.endswith(".gamma") else value)
        for key, value in meta_sd.items()
    }


def verify(size: str, meta_sd: Dict[str, torch.Tensor], weights_path: Path) -> Dict[str, object]:
    """Compare our converted trunk against timm's own model on real images.

    Runs the positive check, then the same check twice more against
    deliberately corrupted mappings.
    """
    import timm

    x, names = load_images()
    reference = timm.create_model(TIMM_MODELS[size], pretrained=True).eval()
    with torch.no_grad():
        timm_raw = reference.forward_intermediates(
            x, indices=[0, 1, 2, 3], norm=False, intermediates_only=True
        )
        timm_stage3_normed = reference.head.norm(timm_raw[3])

    model = build_hub_model(size, weights_path).eval()

    def run(tag: str, state: Optional[Dict[str, torch.Tensor]]) -> Dict[str, object]:
        if state is not None:
            model.load_state_dict(state, strict=True)  # strict even when corrupted
        raw, normed3 = stage_maps(model, x)
        rows = [compare(raw[i], timm_raw[i]) for i in range(4)]
        rows.append(compare(normed3, timm_stage3_normed))
        passed = all(row["rel"] <= REL_TOLERANCE for row in rows)
        return {"tag": tag, "rows": rows, "pass": passed}

    results = [run("converted", None)]
    results.append(run("control:swap-stage2-blocks", corrupt_swap_blocks(meta_sd)))
    results.append(run("control:drop-gamma", corrupt_drop_gamma(meta_sd)))
    return {
        "size": size,
        "timm_model": TIMM_MODELS[size],
        "timm_version": timm.__version__,
        "images": names,
        "input_shape": list(x.shape),
        "tolerance_rel": REL_TOLERANCE,
        "results": results,
    }


def compare_to_meta(meta_sd: Dict[str, torch.Tensor], reference_path: Path) -> Dict[str, object]:
    """Diff a conversion against a genuine Meta checkpoint, tensor by tensor.

    Only usable where a gated Meta file is already on disk (``base``, here).
    It is the strongest statement available about the name map: if renaming
    timm's ``base`` tensors reproduces Meta's own ``base`` file exactly, the map
    is not merely plausible, it is the map Meta used.
    """
    reference = torch.load(str(reference_path), map_location="cpu", weights_only=True)
    missing = sorted(set(reference) - set(meta_sd))
    extra = sorted(set(meta_sd) - set(reference))
    worst_key, worst = None, 0.0
    exact = 0
    for key in sorted(set(reference) & set(meta_sd)):
        diff = (meta_sd[key].float() - reference[key].float()).abs().max().item()
        exact += int(torch.equal(meta_sd[key], reference[key]))
        if diff >= worst:
            worst_key, worst = key, diff
    return {
        "reference": str(reference_path),
        "n_reference": len(reference),
        "n_converted": len(meta_sd),
        "missing_keys": missing,
        "extra_keys": extra,
        "bit_identical_tensors": exact,
        "worst_key": worst_key,
        "worst_max_abs": worst,
    }


ROW_LABELS = ("stage0 (s4)", "stage1 (s8)", "stage2 (s16)", "stage3 (s32)", "stage3 + final norm")


def format_verification(report: Dict[str, object]) -> str:
    """Render a verification report as a markdown table block."""
    lines = [
        f"### {report['size']} — `{report['timm_model']}`",
        "",
        f"Images ({len(report['images'])}, resized to "
        f"{report['input_shape'][2]}x{report['input_shape'][3]}, ImageNet-normalised): "
        + ", ".join(f"`{n}`" for n in report["images"]),
        "",
        "| variant | stage | max abs diff | mean abs diff | ref max abs | relative | <= 1e-4 |",
        "|---|---|---|---|---|---|---|",
    ]
    for result in report["results"]:
        for label, row in zip(ROW_LABELS, result["rows"]):
            ok = "yes" if row["rel"] <= report["tolerance_rel"] else "**NO**"
            lines.append(
                f"| `{result['tag']}` | {label} | {row['max_abs']:.3e} | {row['mean_abs']:.3e} "
                f"| {row['ref_max_abs']:.3e} | {row['rel']:.3e} | {ok} |"
            )
    lines.append("")
    for result in report["results"]:
        verdict = "PASS" if result["pass"] else "FAIL"
        lines.append(f"- `{result['tag']}`: **{verdict}**")
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------

def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="convert_timm_dinov3.py",
        description="Rebuild Meta-format DINOv3 ConvNeXt weights from timm's ungated re-host.",
    )
    parser.add_argument("--size", required=True, choices=sorted(TIMM_MODELS))
    parser.add_argument("--out", type=Path, default=Path("weights"))
    parser.add_argument("--cache-dir", type=Path, default=None,
                        help="where timm downloads land (default: <out>/timm)")
    parser.add_argument("--check", action="store_true",
                        help="load the written file through Meta's hub constructor (strict)")
    parser.add_argument("--verify", action="store_true",
                        help="compare stage activations against timm's own model, with controls")
    parser.add_argument("--report", type=Path, default=None,
                        help="append the --verify markdown table to this file")
    parser.add_argument("--json", type=Path, default=None, help="write the --verify report as JSON")
    parser.add_argument("--compare-meta", type=Path, default=None,
                        help="diff the conversion against a genuine Meta .pth, tensor by tensor")
    args = parser.parse_args(argv)

    cache_dir = args.cache_dir or (args.out / "timm")
    source = download_timm_file(args.size, cache_dir)
    timm_sd = load_checkpoint(source)
    meta_sd = convert_state_dict(timm_sd)

    args.out.mkdir(parents=True, exist_ok=True)
    out_path = args.out / WEIGHT_FILES[args.size]
    torch.save(meta_sd, out_path)
    print(f"[{args.size}] source   : {source}")
    print(f"[{args.size}] tensors  : timm {len(timm_sd)} -> Meta {len(meta_sd)} (+2 norm alias)")
    print(f"[{args.size}] dtypes   : {sorted({str(t.dtype) for t in meta_sd.values()})}")
    print(f"[{args.size}] written  : {out_path}  ({out_path.stat().st_size / 1e6:.1f} MB)")

    if args.compare_meta:
        diff = compare_to_meta(meta_sd, args.compare_meta)
        print(f"[{args.size}] vs {diff['reference']}:")
        print(f"[{args.size}]   keys {diff['n_converted']} vs {diff['n_reference']}, "
              f"missing {len(diff['missing_keys'])}, extra {len(diff['extra_keys'])}")
        print(f"[{args.size}]   bit-identical tensors {diff['bit_identical_tensors']}"
              f"/{diff['n_reference']}, worst max-abs {diff['worst_max_abs']:.3e} "
              f"({diff['worst_key']})")

    if args.check:
        model = build_hub_model(args.size, out_path)
        n_params = sum(p.numel() for p in model.parameters())
        print(f"[{args.size}] strict load OK via hub constructor; {n_params/1e6:.2f}M params")

    if args.verify:
        report = verify(args.size, meta_sd, out_path)
        text = format_verification(report)
        print(text)
        if args.report:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            with open(args.report, "a") as handle:
                handle.write(text + "\n")
        if args.json:
            args.json.parent.mkdir(parents=True, exist_ok=True)
            json.dump(report, open(args.json, "w"), indent=1)
        positive = report["results"][0]
        controls = report["results"][1:]
        if not positive["pass"]:
            print("VERIFICATION FAILED: converted weights do not reproduce timm's features")
            return 1
        if any(control["pass"] for control in controls):
            print("VERIFICATION VACUOUS: a corrupted mapping also passed")
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
