# SPDX-License-Identifier: Apache-2.0
"""0.1.3: which GPU, what 0.1.3 does on it, and the context that fits its memory (yasakani/gpu.py and the quickstart's copy of the rule).

nvidia-smi is never run for real here: its output is given as text, or a stand-in program prints it.  The memory sizes of the cards are
what nvidia-smi reports for them (MiB); the rule's numbers are those of the published model files.
"""
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from tests import support
from yasakani import gpu as G
from yasakani._native import YasakaniError

ROOT = Path(__file__).resolve().parent.parent
FASTEST = (10311344128, 1164969472, 1411010584)       # the published files: weights, draft model, its output head
QUALITY = (12365242368, 1164969472, 1411010584)

# index, name, compute capability, memory.total (MiB) -> what the quickstart does, whether the quality model fits
CARDS = [
    ("NVIDIA GeForce RTX 5090", "12.0", 32607, "tested", True),
    ("NVIDIA GeForce RTX 4090", "8.9", 24564, "other", True),
    ("NVIDIA GeForce RTX 3090 Ti", "8.6", 24564, "other", True),
    ("NVIDIA GeForce RTX 3090", "8.6", 24576, "other", True),
    ("NVIDIA GeForce RTX 5080", "12.0", 16303, "other", False),
    ("NVIDIA GeForce RTX 5070 Ti", "12.0", 16303, "other", False),
    ("NVIDIA GeForce RTX 5060 Ti", "12.0", 16311, "other", False),
    ("NVIDIA GeForce RTX 4080 SUPER", "8.9", 16376, "other", False),
    ("NVIDIA GeForce RTX 4080", "8.9", 16376, "other", False),
    ("NVIDIA GeForce RTX 4070 Ti SUPER", "8.9", 16376, "other", False),
    ("NVIDIA GeForce RTX 4060 Ti", "8.9", 16380, "other", False),       # the 16 GB card
    ("NVIDIA GeForce RTX 4060 Ti", "8.9", 8188, "small", False),        # the 8 GB card of the same name
    ("NVIDIA GeForce RTX 5070", "12.0", 12227, "small", False),
    ("NVIDIA GeForce RTX 3060", "8.6", 12288, "small", False),
    ("NVIDIA GeForce RTX 5090 Laptop GPU", "12.0", 24463, "other", True),
    ("NVIDIA GeForce RTX 5090 D", "12.0", 32607, "other", True),
    ("NVIDIA RTX PRO 6000 Blackwell Workstation Edition", "12.0", 97887, "other", True),
    ("NVIDIA RTX A6000", "8.6", 49140, "other", True),
    ("Tesla V100-PCIE-32GB", "7.0", 32768, "other", True),
    ("NVIDIA GeForce RTX 2080 Ti", "7.5", 11264, "old", False),
    ("NVIDIA GeForce GTX 1080 Ti", "6.1", 11264, "old", False),
    ("NVIDIA A100 80GB PCIe", "8.0", 81920, "old", True),
    ("NVIDIA H100 PCIe", "9.0", 81559, "nocode", True),
    ("NVIDIA B200", "10.0", 183359, "nocode", True),
]


def smi_line(i, name, cc, total, used=1000, uuid=None):
    return "%d, %s, %s, %s, %d, %d, 616.92" % (i, uuid or "GPU-%08d-0000-0000-0000-000000000000" % i, name, cc, total, used)


class FakeFile:
    def __init__(self, name, size):
        self.name, self._size = name, size

    def stat(self):
        return os.stat_result((0, 0, 0, 0, 0, 0, self._size, 0, 0, 0))


class FakePack:
    def __init__(self, sizes, max_ctx=262144):
        self.weights, self.draft, self.draft_head = (FakeFile(n, s) for n, s in zip(("w.yskn", "bdraft2_k8.yskn", "bdraft2_k8.shead"), sizes))
        self.max_ctx = max_ctx


def det_of(*rows, environ=None):
    gpus = G.parse("\n".join(rows) + "\n")
    return G.detect(0, environ=environ or {}, gpus=gpus)


def load_quickstart():
    """quickstart/quickstart.py of the repository (or the file YASAKANI_QUICKSTART names); None when it is not there (the engine zip)."""
    for p in (os.environ.get("YASAKANI_QUICKSTART"), ROOT.parent / "quickstart" / "quickstart.py"):
        if p and Path(p).is_file():
            spec = importlib.util.spec_from_file_location("yasakani_quickstart_under_test", str(p))
            m = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(m)
            return m
    return None


class ParseTests(unittest.TestCase):
    def test_rows(self):
        gs = G.parse(smi_line(0, "NVIDIA GeForce RTX 5090", "12.0", 32607, 2644) + "\n" + smi_line(1, "NVIDIA GeForce RTX 3090", "8.6", 24576, 15))
        self.assertEqual(len(gs), 2)
        self.assertEqual((gs[0].index, gs[0].compute_cap, gs[0].total_mib, gs[0].used_mib, gs[0].free_mib), (0, (12, 0), 32607, 2644, 29963))
        self.assertEqual((gs[1].name, gs[1].cc_text, gs[1].driver), ("NVIDIA GeForce RTX 3090", "8.6", "616.92"))
        self.assertTrue(gs[0].uuid.startswith("GPU-"))

    def test_missing_values_and_noise(self):
        gs = G.parse("0, [N/A], NVIDIA GeForce RTX 4090, [N/A], 24564, [N/A], 581.57\nNo devices were found\n\n")
        self.assertEqual(len(gs), 1)
        g = gs[0]
        self.assertEqual((g.uuid, g.compute_cap, g.used_mib, g.free_mib, g.arch), (None, None, None, None, "unknown"))
        self.assertEqual(G.parse("No devices were found\n"), [])
        self.assertEqual(G.parse(""), [])

    def test_query_falls_back_without_compute_cap(self):
        calls = []

        def run(cmd, **kw):
            calls.append(cmd[1])
            if "compute_cap" in cmd[1]:
                return subprocess.CompletedProcess(cmd, 2, stdout='Field "compute_cap" is not a valid field to query.\n', stderr="")
            return subprocess.CompletedProcess(cmd, 0, stdout="0, GPU-1, NVIDIA GeForce RTX 3090, 24576, 500, 470.10\n", stderr="")
        gs = G.query({"YASAKANI_NVIDIA_SMI": "/x/nvidia-smi"}, run=run)
        self.assertEqual(len(calls), 2)
        self.assertEqual((gs[0].name, gs[0].compute_cap, gs[0].total_mib), ("NVIDIA GeForce RTX 3090", None, 24576))

    def test_query_without_nvidia_smi(self):
        self.assertIsNone(G.query({"YASAKANI_NVIDIA_SMI": "none"}))
        self.assertIsNone(G.query({"YASAKANI_NVIDIA_SMI": ""}))

        def boom(cmd, **kw):
            raise FileNotFoundError(cmd[0])
        self.assertIsNone(G.query({"YASAKANI_NVIDIA_SMI": "/nonexistent/nvidia-smi"}, run=boom))
        with mock.patch.object(G.shutil, "which", return_value=None):
            self.assertIsNone(G.query({}))
        self.assertFalse(G.detect(0, environ={"YASAKANI_NVIDIA_SMI": "none"}).known)


class DecisionTableTests(unittest.TestCase):
    def test_cards(self):
        for name, cc, total, qs, quality in CARDS:
            g = G.parse(smi_line(0, name, cc, total))[0]
            with self.subTest(card=name, total=total):
                self.assertEqual(g.tested, qs == "tested")
                self.assertEqual(g.arch, {"old": "old", "nocode": "no-code"}.get(qs, "ok"))
                line = G.describe(g)
                if qs == "tested":
                    self.assertIn("tested and tuned", line)
                elif qs == "other":
                    self.assertIn("not optimized or tested", line)
                    self.assertEqual("fastest model only" in line, not quality)
                elif qs == "small":
                    self.assertIn("too small", line)
                else:
                    self.assertIn("no code", line)

    def test_untested_note(self):
        self.assertIsNone(G.untested_note(det_of(smi_line(0, "NVIDIA GeForce RTX 5090", "12.0", 32607))))
        self.assertIsNone(G.untested_note(G.Detected()))                         # nothing known: no note
        n = G.untested_note(det_of(smi_line(0, "NVIDIA GeForce RTX 4080", "8.9", 16376)))
        self.assertIn("RTX 4080 (compute capability 8.9)", n)
        self.assertIn("not optimized or tested", n)
        self.assertIn("RTX 5090 only", n)
        n = G.untested_note(det_of(smi_line(0, "NVIDIA GeForce RTX 2080 Ti", "7.5", 11264)))
        self.assertIn("will probably not start", n)


class SelectTests(unittest.TestCase):
    TWO = [smi_line(0, "NVIDIA GeForce RTX 3090", "8.6", 24576, 300, "GPU-aaaa1111-0000"),
           smi_line(1, "NVIDIA GeForce RTX 5090", "12.0", 32607, 2000, "GPU-bbbb2222-0000")]

    def gpus(self):
        return G.parse("\n".join(self.TWO))

    def test_one_gpu(self):
        gs = self.gpus()[:1]
        self.assertEqual(G.select(gs, 0, {}), gs)
        self.assertEqual(G.select(gs, 1, {}), [])
        self.assertEqual(G.select([], 0, {}), [])

    def test_two_gpus(self):
        gs = self.gpus()
        self.assertEqual(len(G.select(gs, 0, {})), 2)                           # CUDA's default order: not known which
        self.assertEqual(G.select(gs, 1, {"CUDA_DEVICE_ORDER": "PCI_BUS_ID"})[0].index, 1)
        self.assertEqual(G.select(gs, 0, {"CUDA_VISIBLE_DEVICES": "GPU-bbbb2222"})[0].index, 1)     # a unique UUID prefix
        self.assertEqual(G.select(gs, 0, {"CUDA_VISIBLE_DEVICES": "1", "CUDA_DEVICE_ORDER": "PCI_BUS_ID"})[0].index, 1)
        self.assertEqual(len(G.select(gs, 0, {"CUDA_VISIBLE_DEVICES": "1"})), 2)
        self.assertEqual(G.select(gs, 1, {"CUDA_VISIBLE_DEVICES": "1"}), [])    # only one GPU is visible
        self.assertEqual(G.select(gs, 0, {"CUDA_VISIBLE_DEVICES": ""}), [])

    def test_two_gpus_plan_for_the_smaller(self):
        det = G.detect(0, environ={}, gpus=self.gpus())
        self.assertTrue(det.untested)                                          # one of them is not an RTX 5090
        w = det.worst()
        self.assertEqual((w.total_mib, w.free_mib, w.compute_cap), (24576, 24276, (8, 6)))
        plan = G.plan_context(FakePack(QUALITY), det)
        self.assertIsNone(plan.max_ctx)                                         # the quality model fits both at 262,144
        self.assertTrue(any("2 GPUs" in n for n in plan.notes))
        det = G.detect(0, environ={"CUDA_DEVICE_ORDER": "PCI_BUS_ID"}, gpus=self.gpus())
        self.assertEqual(det.gpu.index, 0)
        det = G.detect(1, environ={"CUDA_DEVICE_ORDER": "PCI_BUS_ID"}, gpus=self.gpus())
        self.assertTrue(det.tested)


class ContextRuleTests(unittest.TestCase):
    """The rule, with the numbers of the published files (yasakani/gpu.py: choose, plan_context)."""

    def test_choose(self):
        self.assertEqual(G.choose(16300 - 1000, *FASTEST, 262144), (65536, True))      # 16 GB that also drives the display
        self.assertEqual(G.choose(16311 - 0, *FASTEST, 262144), (65536, False))       # 16 GB without a display
        self.assertEqual(G.choose(16303 - 1500, *FASTEST, 262144), (49152, True))
        self.assertEqual(G.choose(16303 - 3000, *FASTEST, 262144)[0], 0)              # does not fit
        self.assertEqual(G.choose(24564 - 1000, *QUALITY, 262144), (262144, False))   # 24 GB: the quality model at full context
        self.assertEqual(G.choose(24564 - 4000, *QUALITY, 262144), (163840, False))
        self.assertEqual(G.choose(24564 - 1000, *FASTEST, 262144), (262144, False))
        self.assertEqual(G.choose(16376 - 800, *QUALITY, 262144)[0], 0)               # 16 GB: the quality model does not fit
        self.assertEqual(G.choose(12288 - 300, *FASTEST, 262144)[0], 0)               # 12 GB: nothing fits
        self.assertEqual(G.choose(20000, *FASTEST, 4096), (4096, False))              # a model whose own context is small

    def test_rule_is_safe_against_the_measured_overhead(self):
        """The chosen context must fit with the overhead measured on the RTX 5090 (2,090 / 2,291 MiB above the files)."""
        for sizes, measured in ((FASTEST, 2090), (QUALITY, 2291)):
            for total in (16303, 16376, 20470, 24564, 24576):
                for used in range(0, 6001, 250):
                    n, head_off = G.choose(total - used, *sizes, 262144)
                    if not n:
                        continue
                    files = sizes[0] + sizes[1] + (0 if head_off else sizes[2])
                    need = files / G.MIB + measured + n * G.KV_BYTES_PER_TOKEN / G.MIB
                    self.assertLessEqual(need + 500, total - used, (sizes, total, used, n, head_off))
                    self.assertEqual(n % G.CONTEXT_STEP if n < 262144 else 0, 0)

    def test_plan_unchanged_on_the_rtx_5090(self):
        for used in (0, 2644, 12000, 20000):                                   # however much else uses it
            plan = G.plan_context(FakePack(QUALITY), det_of(smi_line(0, "NVIDIA GeForce RTX 5090", "12.0", 32607, used)))
            self.assertEqual((plan.max_ctx, plan.env, plan.notes), (None, {}, []))
        plan = G.plan_context(FakePack(QUALITY), G.Detected())                    # nvidia-smi gave nothing
        self.assertEqual((plan.max_ctx, plan.env, plan.notes), (None, {}, []))

    def test_plan_16gb(self):
        plan = G.plan_context(FakePack(FASTEST), det_of(smi_line(0, "NVIDIA GeForce RTX 4080", "8.9", 16376, 1000)))
        self.assertEqual(plan.max_ctx, 81920)
        self.assertEqual(plan.env, {"YSKN_BDR_SHEAD": "0"})
        self.assertTrue(any("bdraft2_k8.shead" in n and "3%" in n for n in plan.notes))
        self.assertTrue(any(n.startswith("context: 81920 tokens") for n in plan.notes))
        plan = G.plan_context(FakePack(FASTEST), det_of(smi_line(0, "NVIDIA GeForce RTX 5060 Ti", "12.0", 16311, 0)))
        self.assertEqual((plan.max_ctx, plan.env), (65536, {}))

    def test_plan_24gb(self):
        plan = G.plan_context(FakePack(QUALITY), det_of(smi_line(0, "NVIDIA GeForce RTX 3090", "8.6", 24576, 1000)))
        self.assertEqual((plan.max_ctx, plan.env, plan.notes), (None, {}, []))
        plan = G.plan_context(FakePack(QUALITY), det_of(smi_line(0, "NVIDIA GeForce RTX 4090", "8.9", 24564, 4000)))
        self.assertEqual((plan.max_ctx, plan.env), (163840, {}))

    def test_plan_does_not_fit(self):
        with self.assertRaises(YasakaniError) as cm:
            G.plan_context(FakePack(QUALITY), det_of(smi_line(0, "NVIDIA GeForce RTX 4080", "8.9", 16376, 500)))
        self.assertIn("does not fit", str(cm.exception))
        self.assertIn("use the fastest model", str(cm.exception))
        with self.assertRaises(YasakaniError) as cm:
            G.plan_context(FakePack(FASTEST), det_of(smi_line(0, "NVIDIA GeForce RTX 3060", "8.6", 12288, 300)))
        self.assertIn("--max-ctx", str(cm.exception))
        self.assertNotIn("fastest model", str(cm.exception))

    def test_plan_unknown_memory(self):
        det = det_of("0, GPU-1, NVIDIA GeForce RTX 4080, 8.9, 16376, [N/A], 581.57")
        self.assertEqual(G.plan_context(FakePack(FASTEST), det).max_ctx, None)


class QuickstartTests(unittest.TestCase):
    """The quickstart's own copy of the decision and the rule (it runs before the package is installed)."""

    @classmethod
    def setUpClass(cls):
        cls.qs = load_quickstart()
        if cls.qs is None:
            raise unittest.SkipTest("quickstart/quickstart.py not found (set YASAKANI_QUICKSTART)")

    def test_version(self):
        self.assertEqual(self.qs.QS_VERSION, "0.1.3")
        self.assertIn("/v0.1.3/", self.qs.ENGINE_URL)

    def test_decision_table(self):
        for name, cc, total, qs, quality in CARDS:
            g = self.qs.parse_gpus(smi_line(0, name, cc, total))[0]
            with self.subTest(card=name, total=total):
                self.assertEqual(self.qs.gpu_verdict(g), qs)
                if qs in ("tested", "other"):
                    self.assertEqual(total >= self.qs.QUALITY_MIN_TOTAL_MIB, quality)
        g = self.qs.parse_gpus("0, GPU-1, NVIDIA GeForce RTX 4090, [N/A], 24564, 900, 470.10")[0]    # an old driver: cc unknown
        self.assertEqual(self.qs.gpu_verdict(g), "other")

    def test_same_as_package(self):
        for k in ("FULL_CONTEXT_MIN_TOTAL_MIB", "QUALITY_MIN_TOTAL_MIB", "MIN_TOTAL_MIB", "KV_BYTES_PER_TOKEN", "CONTEXT_STEP", "MIN_CONTEXT",
                  "HEAD_OFF_BELOW", "MARGIN_MIB"):
            self.assertEqual(getattr(G, k), getattr(self.qs, k), k)
        for name, cc, total, qs, quality in CARDS:
            g = G.parse(smi_line(0, name, cc, total))[0]
            self.assertEqual(g.tested, self.qs.gpu_verdict(self.qs.parse_gpus(smi_line(0, name, cc, total))[0]) == "tested", name)
        for model, sizes in (("speed", FASTEST), ("quality", QUALITY)):
            self.assertEqual(self.qs.MODEL_BYTES[model], sizes)
            self.assertEqual(self.qs.OVERHEAD_MIB[model], G.overhead_mib(sizes[0]))
            for free in range(9000, 32001, 37):
                self.assertEqual(self.qs.estimate_context(free, model), G.choose(free, *sizes, 262144), (model, free))

    def test_pick(self):
        qs = self.qs
        two = qs.parse_gpus(smi_line(0, "NVIDIA GeForce RTX 3060", "8.6", 12288, 300, "GPU-aaaa") + "\n"
                            + smi_line(1, "NVIDIA GeForce RTX 5090", "12.0", 32607, 2000, "GPU-bbbb"))
        g, v, pin = qs.pick_gpu(two, environ={})
        self.assertEqual((g["index"], v, pin), (1, "tested", "GPU-bbbb"))
        g, v, pin = qs.pick_gpu(two[:1], environ={})
        self.assertEqual((g["index"], v, pin), (0, "small", None))
        g, v, pin = qs.pick_gpu(two, environ={"CUDA_VISIBLE_DEVICES": "GPU-aaaa"})       # the user chose one
        self.assertEqual((g["index"], v, pin), (0, "small", None))
        g, v, pin = qs.pick_gpu(two, environ={"CUDA_VISIBLE_DEVICES": "0"})              # not known which: the less able one
        self.assertEqual((g["index"], v, pin), (0, "small", None))
        three = qs.parse_gpus("\n".join([smi_line(0, "NVIDIA GeForce RTX 4080", "8.9", 16376, 100, "GPU-c"),
                                         smi_line(1, "NVIDIA GeForce RTX 3090", "8.6", 24576, 900, "GPU-d"),
                                         smi_line(2, "NVIDIA GeForce RTX 2080 Ti", "7.5", 11264, 0, "GPU-e")]))
        g, v, pin = qs.pick_gpu(three, environ={})
        self.assertEqual((g["index"], v, pin), (1, "other", "GPU-d"))

    def test_query_fallback(self):
        def run(cmd, **kw):
            if "compute_cap" in cmd[1]:
                return subprocess.CompletedProcess(cmd, 2, stdout="", stderr="")
            return subprocess.CompletedProcess(cmd, 0, stdout="0, GPU-1, NVIDIA GeForce RTX 3090, 24576, 500, 470.10\n", stderr="")
        gs = self.qs.query_gpus(run=run)
        self.assertEqual((gs[0]["name"], gs[0]["cc"], gs[0]["total"]), ("NVIDIA GeForce RTX 3090", None, 24576))

    def test_messages(self):
        T = self.qs.T
        self.assertIn("最適化も動作の確認もしていません（確かめたのは RTX 5090 だけです）", T["gpu_other"]["ja"])
        for key in ("no_smi", "no_gpu"):
            self.assertIn("AMD", T[key]["ja"])
            self.assertIn("Mac", T[key]["en"])
        self.assertNotIn("continue_q", T)
        for key, kw in (("gpu_is", dict(g="X", c="8.9", t=16376, d="1")), ("gpu_small", dict(t=8188)), ("gpu_old", dict(c="7.5")),
                        ("gpu_nocode", dict(c="9.0")), ("ctx_auto", dict(n="65,536")), ("ctx_short", dict(u=9000)),
                        ("gpu_pick", dict(n=2, g="X"))):
            for lang in ("ja", "en"):
                T[key][lang].format(**kw)


FAKE_SMI = r'''#!{python}
import json, os, sys
gpus = json.loads(os.environ["FAKE_GPUS"])
q = [a for a in sys.argv[1:] if a.startswith("--query-gpu=")]
if not q:
    sys.exit(0)
fields = q[0].split("=", 1)[1].split(",")
units = "nounits" not in " ".join(sys.argv)
known = set(gpus[0]) if gpus else set()
bad = [f for f in fields if f not in known]
if bad:
    print('Field "%s" is not a valid field to query.' % bad[0])
    sys.exit(2)
for g in gpus:
    print(", ".join(str(g[f]) + (" MiB" if units and f.startswith("memory.") else "") for f in fields))
'''


@unittest.skipUnless(support.tokenizer_dir() is not None and os.name == "posix",
                     "needs the Qwen3.8 tokenizer files (YASAKANI_TEST_MODEL_DIR) and a POSIX test machine")
class CliGpuTests(unittest.TestCase):
    """The command line with a stand-in nvidia-smi and the stub engine library."""

    @classmethod
    def setUpClass(cls):
        cls.dll = str(support.stub_path())
        cls.folder = support.make_real_folder("gpu")
        cls.addClassCleanup(shutil.rmtree, cls.folder.parent, True)
        with open(cls.folder / "bdraft2_k8.shead", "wb") as f:                # a sparse file of the published size: the rule sees it
            f.truncate(FASTEST[2])
        tmp = Path(tempfile.mkdtemp(prefix="fake_smi_"))
        cls.addClassCleanup(shutil.rmtree, tmp, True)
        cls.smi = tmp / "nvidia-smi"
        cls.smi.write_text(FAKE_SMI.replace("{python}", sys.executable), encoding="utf-8")
        cls.smi.chmod(0o755)

    def gpu(self, name, cc, total, used, index=0):
        return dict(index=index, uuid="GPU-%d" % index, name=name, compute_cap=cc, **{"memory.total": total, "memory.used": used},
                    driver_version="616.92")

    def run_cli(self, gpus, *args, timeout=60):
        env = dict(os.environ, PYTHONPATH=str(ROOT), PYTHONUNBUFFERED="1", YASAKANI_NVIDIA_SMI=str(self.smi), FAKE_GPUS=json.dumps(gpus))
        for k in [k for k in env if k.startswith("YSKN_")] + ["YASAKANI_DLL", "YASAKANI_MODEL_DIR", "CUDA_VISIBLE_DEVICES", "CUDA_DEVICE_ORDER"]:
            env.pop(k, None)
        return subprocess.run([sys.executable, "-m", "yasakani", *args], capture_output=True, text=True, timeout=timeout, cwd=str(ROOT),
                              env=env, input="/exit\n")

    def common(self):
        return ["--model-dir", str(self.folder), "--dll", self.dll]

    def test_fake_smi(self):
        r = subprocess.run([str(self.smi), "--query-gpu=name,memory.total", "--format=csv,noheader"], capture_output=True, text=True,
                           env=dict(os.environ, FAKE_GPUS=json.dumps([self.gpu("X", "8.9", 16376, 10)])))
        self.assertEqual(r.stdout.strip(), "X, 16376 MiB")

    def test_rtx_5090_unchanged(self):
        r = self.run_cli([self.gpu("NVIDIA GeForce RTX 5090", "12.0", 32607, 2644)], "chat", *self.common(), "--no-think")
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("(context 262144 tokens)", r.stderr)
        self.assertNotIn("note:", r.stderr)
        self.assertNotIn("context: ", r.stderr)

    def test_16gb_context_and_note(self):
        gpus = [self.gpu("NVIDIA GeForce RTX 4080", "8.9", 16376, 11418)]      # about 4,958 MiB free beside the placeholder weights
        r = self.run_cli(gpus, "chat", *self.common(), "--no-think")
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("NVIDIA GeForce RTX 4080 (compute capability 8.9)", r.stderr)
        self.assertIn("not optimized or tested", r.stderr)
        self.assertIn("bdraft2_k8.shead", r.stderr)
        self.assertIn("context: 98304 tokens", r.stderr)
        self.assertIn("(context 98304 tokens)", r.stderr)
        r = self.run_cli(gpus, "chat", *self.common(), "--no-think", "--max-ctx", "8192")       # given: no rule
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("(context 8192 tokens)", r.stderr)
        self.assertNotIn("context: ", r.stderr)
        self.assertIn("not optimized or tested", r.stderr)

    def test_does_not_fit(self):
        r = self.run_cli([self.gpu("NVIDIA GeForce RTX 4080", "8.9", 16376, 14000)], "chat", *self.common())
        self.assertEqual(r.returncode, 1)
        self.assertIn("does not fit in the free GPU memory", r.stderr)
        self.assertNotIn("loading the model", r.stderr)

    def test_no_nvidia_smi(self):
        env_gpus = [self.gpu("NVIDIA GeForce RTX 4080", "8.9", 16376, 14000)]
        env = dict(os.environ, PYTHONPATH=str(ROOT), YASAKANI_NVIDIA_SMI="none", FAKE_GPUS=json.dumps(env_gpus))
        r = subprocess.run([sys.executable, "-m", "yasakani", "chat", *self.common(), "--no-think"], capture_output=True, text=True,
                           timeout=60, cwd=str(ROOT), env=env, input="/exit\n")
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("(context 262144 tokens)", r.stderr)

    def test_info(self):
        r = self.run_cli([self.gpu("NVIDIA GeForce RTX 3090", "8.6", 24576, 500)], "info", *self.common())
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertIn("NVIDIA GeForce RTX 3090, 24576 MiB, 500 MiB", r.stdout)
        self.assertIn("GPU 0: compute capability 8.6 - runs since 0.1.3, but not optimized or tested", r.stdout)

    def test_selfcheck_exit_codes(self):
        ref = self.folder.parent / "ref_gpu.json"
        args = ["--model-dir", str(self.folder), "--dll", self.dll, "--max-ctx", "4096"]
        g5090 = [self.gpu("NVIDIA GeForce RTX 5090", "12.0", 32607, 2644)]
        g4080 = [self.gpu("NVIDIA GeForce RTX 4080", "8.9", 16376, 1000)]
        r = self.run_cli(g5090, "selfcheck", *args, "--write", str(ref))
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        r = self.run_cli(g4080, "selfcheck", *args, "--reference", str(ref))
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)                 # the same output passes on any GPU
        d = json.loads(ref.read_text(encoding="utf-8"))
        d["cases"][1]["sha256"] = "0" * 64
        ref.write_text(json.dumps(d), encoding="utf-8")
        r = self.run_cli(g5090, "selfcheck", *args, "--reference", str(ref))
        self.assertEqual(r.returncode, 1, r.stdout)                            # on the RTX 5090 a difference is a failure
        self.assertNotIn("Exit code 3", r.stdout)
        r = self.run_cli(g4080, "selfcheck", *args, "--reference", str(ref))
        self.assertEqual(r.returncode, 3, r.stdout)                            # elsewhere it is told apart
        self.assertIn("made on an RTX 5090", r.stdout)
        d["cases"][2]["prompt_sha256"] = "1" * 64
        ref.write_text(json.dumps(d), encoding="utf-8")
        r = self.run_cli(g4080, "selfcheck", *args, "--reference", str(ref))
        self.assertEqual(r.returncode, 1, r.stdout)                            # a tokenizer problem stays a failure


@unittest.skipUnless(support.tokenizer_dir() is not None and os.name == "posix",
                     "needs the Qwen3.8 tokenizer files (YASAKANI_TEST_MODEL_DIR) and a POSIX test machine")
class OpenStackTests(unittest.TestCase):
    """_open_stack passes the rule's context and engine setting to the Engine (in this process, the stub library)."""

    def test_head_off_reaches_the_engine(self):
        from yasakani import cli
        folder = support.make_real_folder("stack")
        self.addCleanup(shutil.rmtree, folder.parent, True)
        with open(folder / "bdraft2_k8.shead", "wb") as f:
            f.truncate(FASTEST[2])
        gpus = G.parse(smi_line(0, "NVIDIA GeForce RTX 5070 Ti", "12.0", 16303, 11300))
        args = cli.build_parser().parse_args(["chat", "--model-dir", str(folder), "--dll", str(support.stub_path())])
        saved = {k: v for k, v in os.environ.items() if k.startswith("YSKN_")}

        def restore():                                                           # the engine settings of this process as before
            for k in [k for k in os.environ if k.startswith("YSKN_")]:
                del os.environ[k]
            os.environ.update(saved)
        self.addCleanup(restore)
        with mock.patch.object(G, "query", return_value=gpus):
            pack, eng, fmt, runner = cli._open_stack(args, warm=False)
        try:
            self.assertEqual(eng.max_ctx, 98304)
            self.assertEqual(eng.profile.env["YSKN_BDR_SHEAD"], "0")
            self.assertEqual(os.environ["YSKN_BDR_SHEAD"], "0")
            self.assertTrue(args.gpu_detected.untested)
        finally:
            eng.close()


if __name__ == "__main__":
    unittest.main()
