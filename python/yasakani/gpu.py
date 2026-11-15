# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 the YASAKANI authors
"""The GPU the engine runs on, and a context length that fits its memory.

Since 0.1.3 the engine library is built for the RTX 30, 40 and 50 series (compute capability 8.6, 8.9 and 12.x).  It has been tested and
tuned on the RTX 5090 only; on other cards it runs without that tuning, and its speed and operation are not guaranteed.

This fork adds the Tesla V100 (compute capability 7.0): the engine carries sm_70 code whose tensor-core instructions are
emulated (see src/volta_compat.cuh), so the V100 is treated as a supported card here.  It is neither tuned nor tested
like the RTX 5090, and it is much slower; a build for it needs a CUDA 12.x toolkit.

The engine reserves the key-value cache for the whole context when the model is opened.  On a card with less memory than the RTX 5090,
the default context of 262,144 tokens may not fit next to the model: Windows then keeps part of it in ordinary memory and generation
becomes several times slower, without an error.  :func:`plan_context` therefore picks, before the model is opened, the longest context
that fits the free GPU memory.  On cards with 30,000 MiB or more (the RTX 5090) it changes nothing: the context stays the model's own.

The numbers below were measured on the RTX 5090 with the released engine (Windows, CUDA context included); on smaller cards the fixed
part is expected to be a little smaller, so the rule errs on the safe side.  nvidia-smi is the only source: nothing here starts CUDA.
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
from dataclasses import dataclass, field, replace
from typing import Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from ._native import YasakaniError

MIB = 1 << 20
KV_BYTES_PER_TOKEN = 22400            # KV5 cache 22,144 B + position table 256 B per token of context, reserved at open
CONTEXT_STEP = 16384                  # a chosen context is a multiple of this
MIN_CONTEXT = 16384                   # below this the model is not opened
HEAD_OFF_BELOW = 65536                # under this, the draft model's output head is left out to make room
MARGIN_MIB = 512                      # the desktop's drift, CUDA graphs, image input (at most about 84 MiB), rounding
OVERHEAD_MIB = {"fastest": 2100, "quality": 2300}    # GPU memory above the three model files (measured 2,090 and 2,291 MiB)
QUALITY_WEIGHTS_MIB = 11000           # weights larger than this are counted with the overhead of the quality model
FULL_CONTEXT_MIN_TOTAL_MIB = 30000    # cards with this much memory keep the model's own context (as before 0.1.3)
MIN_TOTAL_MIB = 15000                 # "16 GB": the smallest cards the released models fit on (fastest model only)
QUALITY_MIN_TOTAL_MIB = 23000         # "24 GB": the quality model needs this much
HEAD_SLOWER = "about 3%"              # each pass without the output head, measured on the RTX 5090

QUERY_FIELDS = ("index", "uuid", "name", "compute_cap", "memory.total", "memory.used", "driver_version")
_TESTED = re.compile(r"(^|\s)RTX 5090$")


@dataclass(frozen=True)
class GpuInfo:
    """One row of nvidia-smi.  Values nvidia-smi could not give are None."""
    index: int
    name: str
    uuid: Optional[str] = None
    compute_cap: Optional[Tuple[int, int]] = None
    total_mib: Optional[int] = None
    used_mib: Optional[int] = None
    driver: Optional[str] = None

    @property
    def free_mib(self) -> Optional[int]:
        if self.total_mib is None or self.used_mib is None:
            return None
        return max(0, self.total_mib - self.used_mib)

    @property
    def tested(self) -> bool:
        """The card everything is tested and tuned on: the NVIDIA GeForce RTX 5090 (32 GB)."""
        return bool(_TESTED.search(self.name.strip())) and self.compute_cap in (None, (12, 0))

    @property
    def cc_text(self) -> str:
        return "%d.%d" % self.compute_cap if self.compute_cap else "?"

    @property
    def arch(self) -> str:
        """"ok" (7.0, 8.6 to 8.9, 12.x: this build has code for it), "old" (below 8.6, except 7.0), "no-code" (9.x to 11.x,
        13 and newer) or "unknown" (nvidia-smi did not tell)."""
        cc = self.compute_cap
        if cc is None:
            return "unknown"
        if cc == (7, 0):
            return "ok"
        if cc < (8, 6):
            return "old"
        if cc < (9, 0) or cc[0] == 12:
            return "ok"
        return "no-code"


def smi_path(environ: Optional[Mapping[str, str]] = None) -> Optional[str]:
    """The nvidia-smi to ask.  ``YASAKANI_NVIDIA_SMI`` names another one; set to ``none`` (or empty) nothing is asked."""
    env = os.environ if environ is None else environ
    if "YASAKANI_NVIDIA_SMI" in env:
        p = env["YASAKANI_NVIDIA_SMI"].strip()
        return p if p and p.lower() != "none" else None
    return shutil.which("nvidia-smi")


def _value(s: str) -> Optional[str]:
    s = s.strip()
    return None if (not s or s.startswith("[") or s.upper() == "N/A") else s


def _int(s: Optional[str]) -> Optional[int]:
    try:
        return int(round(float(s))) if s is not None else None
    except ValueError:
        return None


def _cc(s: Optional[str]) -> Optional[Tuple[int, int]]:
    m = re.fullmatch(r"(\d+)\.(\d+)", s or "")
    return (int(m.group(1)), int(m.group(2))) if m else None


def parse(text: str, fields: Sequence[str] = QUERY_FIELDS) -> List[GpuInfo]:
    """The output of ``nvidia-smi --query-gpu=<fields> --format=csv,noheader,nounits`` -> one GpuInfo per line.  Lines that do not
    have one value per field (an error text, "No devices were found") are skipped."""
    out = []
    for n, line in enumerate(ln for ln in text.splitlines() if ln.strip()):
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != len(fields):
            continue
        v = dict(zip(fields, parts))
        name = _value(v.get("name", "")) or ""
        index = _int(_value(v.get("index", ""))) if "index" in v else n
        if not name or index is None:
            continue
        out.append(GpuInfo(index=index, name=name, uuid=_value(v.get("uuid", "")), compute_cap=_cc(_value(v.get("compute_cap", ""))),
                           total_mib=_int(_value(v.get("memory.total", ""))), used_mib=_int(_value(v.get("memory.used", ""))),
                           driver=_value(v.get("driver_version", ""))))
    return out


def query(environ: Optional[Mapping[str, str]] = None, run: Callable = subprocess.run, timeout: float = 15) -> Optional[List[GpuInfo]]:
    """Every GPU nvidia-smi lists, or None when it cannot be asked.  Drivers that do not know ``compute_cap`` are asked without it."""
    exe = smi_path(environ)
    if not exe:
        return None
    for fields in (QUERY_FIELDS, tuple(f for f in QUERY_FIELDS if f != "compute_cap")):
        try:
            r = run([exe, "--query-gpu=" + ",".join(fields), "--format=csv,noheader,nounits"], capture_output=True, text=True,
                    timeout=timeout)
        except (OSError, subprocess.SubprocessError, ValueError):
            return None
        if r.returncode == 0:
            return parse(r.stdout or "", fields)
    return None


@dataclass
class Detected:
    """What is known about the GPU the engine will use.  ``candidates`` has one entry when it is known which GPU that is, several when
    the PC has more than one and CUDA's numbering cannot be matched with nvidia-smi's, and none when nothing is known."""
    gpus: List[GpuInfo] = field(default_factory=list)
    candidates: List[GpuInfo] = field(default_factory=list)
    device: int = 0

    @property
    def gpu(self) -> Optional[GpuInfo]:
        return self.candidates[0] if len(self.candidates) == 1 else None

    @property
    def known(self) -> bool:
        return bool(self.candidates)

    @property
    def tested(self) -> bool:
        return bool(self.candidates) and all(g.tested for g in self.candidates)

    @property
    def untested(self) -> bool:
        """True only when the GPU is known and it is not the RTX 5090 (unknown is not "untested")."""
        return bool(self.candidates) and not self.tested

    def worst(self) -> Optional[GpuInfo]:
        """The GPU itself, or for several candidates the smallest memory and the least free memory of them."""
        if not self.candidates:
            return None
        if len(self.candidates) == 1:
            return self.candidates[0]
        totals = [g.total_mib for g in self.candidates]
        frees = [g.free_mib for g in self.candidates]
        if None in totals or None in frees:
            return None
        caps = [g.compute_cap for g in self.candidates if g.compute_cap]
        t, f = min(totals), min(frees)
        return replace(self.candidates[0], name="the smallest of %d GPUs" % len(self.candidates), uuid=None, total_mib=t,
                       used_mib=t - f, compute_cap=min(caps) if caps else None)


def select(gpus: Sequence[GpuInfo], device: int = 0, environ: Optional[Mapping[str, str]] = None) -> List[GpuInfo]:
    """The GPU(s) that CUDA device ``device`` can be.  nvidia-smi numbers GPUs in PCI order; CUDA does too only with
    ``CUDA_DEVICE_ORDER=PCI_BUS_ID`` (its default puts the fastest first), and ``CUDA_VISIBLE_DEVICES`` renumbers them."""
    env = os.environ if environ is None else environ
    gpus = list(gpus)
    if not gpus or device < 0:
        return []
    pci_order = (env.get("CUDA_DEVICE_ORDER") or "").strip().upper() == "PCI_BUS_ID"
    if "CUDA_VISIBLE_DEVICES" in env:
        entries = [e.strip() for e in (env.get("CUDA_VISIBLE_DEVICES") or "").split(",") if e.strip()]
        if device >= len(entries):
            return []
        e = entries[device]
        if e.upper().startswith("GPU-"):
            m = [g for g in gpus if g.uuid and g.uuid.lower().startswith(e.lower())]
            return m if len(m) == 1 else gpus
        if e.isdigit() and (pci_order or len(gpus) == 1):
            return [g for g in gpus if g.index == int(e)]
        return gpus
    if len(gpus) == 1:
        return gpus if device == 0 else []
    if pci_order:
        return [g for g in gpus if g.index == device]
    return gpus


def detect(device: int = 0, environ: Optional[Mapping[str, str]] = None, gpus: Optional[Sequence[GpuInfo]] = None,
           run: Callable = subprocess.run) -> Detected:
    """Ask nvidia-smi once (unless ``gpus`` is given) and match CUDA device ``device``.  Never raises."""
    if gpus is None:
        gpus = query(environ, run=run) or []
    return Detected(gpus=list(gpus), candidates=select(gpus, device, environ), device=device)


def describe(g: GpuInfo) -> str:
    """One line on what 0.1.3 does on this GPU."""
    if g.tested:
        return "RTX 5090: the card YASAKANI is tested and tuned on"
    if g.arch == "old":
        return "compute capability %s: below 8.6, the released engine has no code for it" % g.cc_text
    if g.compute_cap == (7, 0):
        small = g.total_mib is not None and g.total_mib < QUALITY_MIN_TOTAL_MIB
        return ("Tesla V100: runs on this fork's Volta code (the tensor-core instructions are emulated), but not optimized or tested "
                "(only the RTX 5090 is); speed and operation are not guaranteed" + ("; fastest model only" if small else ""))
    if g.arch == "no-code":
        return "compute capability %s: the released engine has no code for it (it is built for 8.6, 8.9 and 12.x)" % g.cc_text
    if g.total_mib is not None and g.total_mib < MIN_TOTAL_MIB:
        return "%d MiB: too small, the released models need a 16 GB card" % g.total_mib
    small = g.total_mib is not None and g.total_mib < QUALITY_MIN_TOTAL_MIB
    return ("runs since 0.1.3, but not optimized or tested (only the RTX 5090 is); speed and operation are not guaranteed"
            + ("; fastest model only" if small else ""))


def untested_note(det: Detected) -> Optional[str]:
    """The line printed at start on any GPU that is not the RTX 5090."""
    if not det.untested:
        return None
    g = det.gpu
    what = "%s (compute capability %s)" % (g.name, g.cc_text) if g else "this PC's GPUs (%s)" % ", ".join(sorted({x.name for x in det.candidates}))
    note = ("note: %s - YASAKANI runs on it since 0.1.3, but it is not optimized or tested there (it is tested on the RTX 5090 only); "
            "speed and operation are not guaranteed." % what)
    if g and g.arch in ("old", "no-code"):
        note += "  The released engine has no code for compute capability %s, so it will probably not start." % g.cc_text
    return note


@dataclass
class ContextPlan:
    """What :func:`plan_context` decided.  ``max_ctx`` None = the model's own context (unchanged); ``env`` = engine settings to add."""
    max_ctx: Optional[int] = None
    env: Dict[str, str] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)
    free_mib: Optional[int] = None


def overhead_mib(weights_bytes: int) -> int:
    return OVERHEAD_MIB["quality"] if weights_bytes / MIB > QUALITY_WEIGHTS_MIB else OVERHEAD_MIB["fastest"]


def fit_tokens(free_mib: float, files_bytes: int, overhead: int) -> int:
    """How many tokens of context fit: (free - files - overhead - margin) / bytes per token (0 when nothing fits)."""
    room = free_mib - files_bytes / MIB - overhead - MARGIN_MIB
    return max(0, int(room * MIB // KV_BYTES_PER_TOKEN))


def choose(free_mib: float, weights_bytes: int, draft_bytes: int, head_bytes: int, model_max: int) -> Tuple[int, bool]:
    """The rule: (context, head_off).  context is ``model_max`` when it fits, else a multiple of CONTEXT_STEP, 0 when even
    MIN_CONTEXT does not fit.  Under HEAD_OFF_BELOW the draft model's output head is left out when that gives more room."""
    over = overhead_mib(weights_bytes)

    def ctx(files):
        n = fit_tokens(free_mib, files, over)
        return model_max if n >= model_max else (n // CONTEXT_STEP) * CONTEXT_STEP

    with_head = ctx(weights_bytes + draft_bytes + head_bytes)
    if with_head >= min(HEAD_OFF_BELOW, model_max):
        return with_head, False
    without = ctx(weights_bytes + draft_bytes)
    if without > with_head:
        return without, True
    return with_head, False


def plan_context(pack, det: Detected) -> ContextPlan:
    """The context for ``pack`` on the detected GPU.  Unchanged (None) on cards with FULL_CONTEXT_MIN_TOTAL_MIB or more, and when
    nvidia-smi gave no answer.  Raises YasakaniError when not even MIN_CONTEXT tokens fit."""
    plan = ContextPlan()
    g = det.worst()
    if g is None or g.total_mib is None or g.free_mib is None:
        if len(det.candidates) > 1:
            plan.notes.append("note: this PC has %d GPUs and their memory could not be read; the context stays %d tokens"
                              % (len(det.candidates), pack.max_ctx))
        return plan
    if g.total_mib >= FULL_CONTEXT_MIN_TOTAL_MIB:
        return plan
    plan.free_mib = g.free_mib
    sizes = [p.stat().st_size for p in (pack.weights, pack.draft, pack.draft_head)]
    n, head_off = choose(g.free_mib, sizes[0], sizes[1], sizes[2], pack.max_ctx)
    if len(det.candidates) > 1:
        plan.notes.append("note: this PC has %d GPUs and CUDA's device %d cannot be matched with nvidia-smi's list, so the context is "
                          "chosen for the smallest of them (set CUDA_DEVICE_ORDER=PCI_BUS_ID to match them)" % (len(det.candidates), det.device))
    if n < MIN_CONTEXT:
        raise YasakaniError(
            "the model does not fit in the free GPU memory (%d of %d MiB free; it needs about %d MiB with a context of %d tokens).  "
            "Close other programs that use the GPU%s, or give --max-ctx to try anyway."
            % (g.free_mib, g.total_mib, int((sizes[0] + sizes[1]) / MIB + overhead_mib(sizes[0]) + MARGIN_MIB + MIN_CONTEXT * KV_BYTES_PER_TOKEN / MIB),
               MIN_CONTEXT, "" if sizes[0] / MIB <= QUALITY_WEIGHTS_MIB else ", or use the fastest model"))
    if head_off:
        plan.env["YSKN_BDR_SHEAD"] = "0"
        plan.notes.append("note: to leave room for the context, the output head of the draft model (%s, %.1f GiB) is not loaded; "
                          "each pass takes %s longer (measured on the RTX 5090)" % (pack.draft_head.name, sizes[2] / (1 << 30), HEAD_SLOWER))
    if n < pack.max_ctx:
        plan.max_ctx = n
        plan.notes.append("context: %d tokens, chosen from the free GPU memory (%d of %d MiB free); --max-ctx sets it yourself"
                          % (n, g.free_mib, g.total_mib))
    return plan
