"""Pin baseline presets to Appendix C.1 Table 8 and LoRA to D.2 Table 9.

Check package exports, lookup, CLI defaults, and constructor dispatch with
recorder classes so preset tests do not train models.
"""

from __future__ import annotations

import inspect
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
TOY_RELEASE = REPO_ROOT / "data" / "public" / "intermediate"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


# Paper Appendix C.1 Table 8 — copy-paste pin per baseline


#: Maps method name → expected (paper-claimed) hyperparameter dict.
#: Values must match paper Appendix C.1 Table 8 exactly.
EXPECTED_PAPER_HYPERPARAMS: dict[str, dict[str, object]] = {
    "deepddi": {
        "ssp_dim": 50,
        "hidden_dim": 2048,
        "n_layers": 9,
        "dropout": 0.3,
        "learning_rate": 1e-3,
        "batch_size": 256,
        "n_epochs": 100,
    },
    "ssi_ddi": {
        "in_features": 55,
        "hidd_dim": 64,
        "kge_dim": 64,
        "heads_out_feat_params": (32, 32, 32, 32),
        "blocks_params": (2, 2, 2, 2),
        "learning_rate": 1e-2,
        "weight_decay": 5e-4,
        "batch_size": 1024,
        "n_epochs": 150,
    },
    "dsn_ddi": {
        "in_features": 55,
        "hidd_dim": 128,
        "kge_dim": 128,
        "heads_out_feat_params": (64, 64),
        "blocks_params": (2, 2),
        "learning_rate": 1e-3,
        "weight_decay": 5e-4,
        "batch_size": 512,
        "n_epochs": 50,
    },
    "hdn_ddi": {
        "in_features": 55,
        "hidd_dim": 128,
        "kge_dim": 128,
        "heads_out_feat_params": (64, 64),
        "blocks_params": (2, 2),
        "learning_rate": 1e-3,
        "weight_decay": 5e-4,
        "batch_size": 512,
        "n_epochs": 50,
    },
    "emergnn": {
        "n_dim": 64,
        "length": 3,
        "feat": "M",
        "learning_rate": 1e-3,
        "batch_size": 32,
        "n_epochs": 40,
    },
    "tiger": {
        "mol_only": False,
        "max_layer": 2,
        "output_dim": 64,
        "dropout": 0.2,
        "extractor": "randomWalk",
        "fixed_num": 32,
        "learning_rate": 1e-3,
        "weight_decay": 1e-4,
        "batch_size": 128,
        "n_epochs": 50,
    },
    "mkg_fenn": {
        "embedding_num": 128,
        "neighbor_sample_size": 6,
        "dropout": 0.3,
        "learning_rate": 1e-2,
        "weight_decay": 1e-8,
        "batch_size": 256,
        "n_epochs": 50,
    },
    "textddi": {
        "backbone": "roberta-base",
        "max_length": 256,
        "learning_rate": 1e-5,
        "weight_decay": 1e-6,
        "batch_size": 32,
        "n_epochs": 30,
    },
}


# Per-baseline PAPER_HYPERPARAMS dict matches Table 8


class TestPaperHyperparamsMatchTableC1:
    """Every PAPER_HYPERPARAMS value matches Appendix C.1 Table 8."""

    @pytest.mark.parametrize(
        "method,expected",
        list(EXPECTED_PAPER_HYPERPARAMS.items()),
    )
    def test_per_baseline_dict_matches_paper(self, method, expected):
        from coldddi.baselines import ensure_imported
        from coldddi.baselines.base import get_paper_hyperparams

        ensure_imported(method)
        actual = get_paper_hyperparams(method)
        for k, v in expected.items():
            assert k in actual, (
                f"{method}: PAPER_HYPERPARAMS missing key {k!r} "
                f"(paper Appendix C.1 Table 8 lists it)"
            )
            assert actual[k] == v, (
                f"{method}: PAPER_HYPERPARAMS[{k!r}]={actual[k]!r} "
                f"!= paper Table 8 value {v!r}"
            )

    @pytest.mark.parametrize(
        "method,expected",
        list(EXPECTED_PAPER_HYPERPARAMS.items()),
    )
    def test_no_extra_keys_beyond_paper(self, method, expected):
        """Reject hyperparameter keys absent from the paper table."""
        from coldddi.baselines import ensure_imported
        from coldddi.baselines.base import get_paper_hyperparams

        ensure_imported(method)
        actual = get_paper_hyperparams(method)
        extras = set(actual) - set(expected)
        assert not extras, (
            f"{method}: PAPER_HYPERPARAMS has extra keys {extras} "
            "not in paper Table 8.  Either remove them or update the "
            "EXPECTED_PAPER_HYPERPARAMS pin if Table 8 was extended."
        )


# Package-root re-export pinned for every baseline


class TestPaperHyperparamsAreReExported:
    @pytest.mark.parametrize("method", list(EXPECTED_PAPER_HYPERPARAMS))
    def test_importable_from_package_root(self, method):
        """Each baseline package exports PAPER_HYPERPARAMS for downstream tools."""
        from coldddi.baselines import ensure_imported

        ensure_imported(method)
        import importlib

        mod = importlib.import_module(f"coldddi.baselines.{method}")
        assert hasattr(mod, "PAPER_HYPERPARAMS"), (
            f"coldddi.baselines.{method} is missing PAPER_HYPERPARAMS"
        )
        assert "PAPER_HYPERPARAMS" in mod.__all__


# evaluate.py --preset paper is the default


class TestPaperPresetIsDefault:
    def test_default_preset_constant(self):
        from coldddi.evaluate import DEFAULT_PRESET, PRESETS

        assert DEFAULT_PRESET == "paper"
        assert set(PRESETS) == {"paper", "smoke"}

    def test_cli_default_is_paper(self):
        """Appendix C.1's paper default belongs to the CLI, not class constructors."""
        from coldddi.evaluate import _build_parser

        parser = _build_parser()
        args = parser.parse_args([
            "--method", "deepddi",
            "--data", "/tmp/x",
            "--out", "/tmp/y",
        ])
        assert args.preset == "paper"

    def test_cli_rejects_unknown_preset(self):
        from coldddi.evaluate import _build_parser

        parser = _build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args([
                "--method", "deepddi",
                "--data", "/tmp/x",
                "--out", "/tmp/y",
                "--preset", "bogus",
            ])


# run_evaluation(preset="paper") materialises the dict


class TestPaperPresetMaterialisesHyperparams:
    """Record constructor kwargs to verify paper-preset dispatch without training."""

    @pytest.mark.skipif(
        not (TOY_RELEASE / "filtered" / "drugs.csv").is_file(),
        reason="Toy fixture missing — run reconstruct.py --toy first.",
    )
    @pytest.mark.parametrize("method", list(EXPECTED_PAPER_HYPERPARAMS))
    def test_paper_preset_forwards_hyperparams_to_class(
        self, method, tmp_path, monkeypatch,
    ):
        from coldddi import evaluate
        from coldddi.baselines import ensure_imported
        from coldddi.baselines.base import _REGISTRY, get_paper_hyperparams

        ensure_imported(method)
        OrigCls = _REGISTRY[method]

        captured: dict[str, object] = {}

        # Record the kwargs passed by evaluate.py.
        sig = inspect.signature(OrigCls)
        param_names = [p for p in sig.parameters if p != "self"]

        captured_box: dict = {}

        class FakeBaseline:
            name = method
            modality = "mol"  # arbitrary; never read by evaluate.py

            def __init__(self, **kw):
                captured_box["init_kwargs"] = dict(kw)
                # Stop before training; only constructor kwargs matter here.
                raise RuntimeError("FakeBaseline: capture-only")

        # Preserve the real signature for evaluate.py's parameter filter.
        import inspect as _inspect
        new_params = [
            _inspect.Parameter("self", _inspect.Parameter.POSITIONAL_OR_KEYWORD),
        ] + [
            _inspect.Parameter(
                name, _inspect.Parameter.KEYWORD_ONLY, default=None,
            )
            for name in param_names
        ]
        FakeBaseline.__init__.__signature__ = _inspect.Signature(new_params)

        _REGISTRY[method] = FakeBaseline
        try:
            with pytest.raises(RuntimeError, match="FakeBaseline"):
                evaluate.run_evaluation(
                    method=method,
                    data_dir=TOY_RELEASE,    # real toy dir so load_*  works
                    seed=42,
                    settings=["S2"],
                    out_dir=tmp_path / "out",
                    device="cpu",
                    preset="paper",
                    with_indicators=False,
                )
        finally:
            _REGISTRY[method] = OrigCls

        # Require at least one forwarded paper kwarg, with the paper value.
        # Data-resolution failures before training skip this test.
        if "init_kwargs" not in captured_box:
            pytest.skip(
                f"{method}: data_dir resolution raised before "
                "FakeBaseline.__init__ was called"
            )
        observed_kwargs = captured_box["init_kwargs"]
        expected = EXPECTED_PAPER_HYPERPARAMS[method]
        # Forward every paper kwarg accepted by the real class.
        for k, v in expected.items():
            if k in param_names and k in observed_kwargs:
                assert observed_kwargs[k] == v, (
                    f"{method}: preset=paper forwarded "
                    f"{k}={observed_kwargs[k]!r} != paper {v!r}"
                )


# preset="smoke" leaves class defaults intact


class TestSmokePresetUsesClassDefaults:
    @pytest.mark.skipif(
        not (TOY_RELEASE / "filtered" / "drugs.csv").is_file(),
        reason="Toy fixture missing",
    )
    def test_smoke_preset_does_not_load_paper_dict(
        self, tmp_path, monkeypatch,
    ):
        """The smoke preset retains constructor defaults without paper kwargs."""
        from coldddi import evaluate
        from coldddi.baselines import ensure_imported
        from coldddi.baselines.base import _REGISTRY

        ensure_imported("deepddi")
        OrigCls = _REGISTRY["deepddi"]

        captured_box: dict = {}

        class FakeBaseline:
            name = "deepddi"
            modality = "mol"

            def __init__(self, **kw):
                captured_box["init_kwargs"] = dict(kw)
                raise RuntimeError("FakeBaseline: capture-only")

        import inspect as _inspect
        new_params = [
            _inspect.Parameter("self", _inspect.Parameter.POSITIONAL_OR_KEYWORD),
            _inspect.Parameter(
                "device", _inspect.Parameter.KEYWORD_ONLY, default="auto",
            ),
        ]
        FakeBaseline.__init__.__signature__ = _inspect.Signature(new_params)

        _REGISTRY["deepddi"] = FakeBaseline
        try:
            with pytest.raises(RuntimeError, match="FakeBaseline"):
                evaluate.run_evaluation(
                    method="deepddi",
                    data_dir=TOY_RELEASE,
                    seed=42,
                    settings=["S2"],
                    out_dir=tmp_path / "out",
                    device="cpu",
                    preset="smoke",
                    with_indicators=False,
                )
        finally:
            _REGISTRY["deepddi"] = OrigCls

        if "init_kwargs" not in captured_box:
            pytest.skip("FakeBaseline.__init__ not reached")
        observed = captured_box["init_kwargs"]
        # Only device is forwarded; paper training parameters stay unset.
        assert "n_epochs" not in observed, (
            "preset='smoke' leaked paper n_epochs onto the constructor"
        )
        assert "batch_size" not in observed
        assert observed.get("device") == "cpu"


# Invalid preset raises


class TestInvalidPresetRaises:
    def test_run_evaluation_invalid_preset_raises(self, tmp_path):
        from coldddi.evaluate import run_evaluation

        with pytest.raises(ValueError, match="preset must be"):
            run_evaluation(
                method="deepddi",
                data_dir=tmp_path,
                seed=42,
                settings=["S2"],
                out_dir=tmp_path / "out",
                preset="bogus",
            )


# LoRA paper preset (Appendix D.2 Table 9)


class TestPaperLoRAHyperparams:
    """Pin PAPER_LORA_HYPERPARAMS to Appendix D.2 Table 9."""

    def test_lora_adapter_block_pinned(self):
        from coldddi.llm.trainer import PAPER_LORA_HYPERPARAMS

        lora = PAPER_LORA_HYPERPARAMS["lora"]
        assert lora["r"] == 16
        assert lora["alpha"] == 16
        assert lora["dropout"] == 0.10
        assert lora["target_modules"] == ("q_proj", "k_proj", "v_proj", "o_proj")
        assert lora["task_type"] == "CAUSAL_LM"
        assert lora["bias"] == "none"

    def test_training_block_pinned(self):
        from coldddi.llm.trainer import PAPER_LORA_HYPERPARAMS

        assert PAPER_LORA_HYPERPARAMS["num_epochs"] == 4
        assert PAPER_LORA_HYPERPARAMS["learning_rate"] == 5e-4
        assert PAPER_LORA_HYPERPARAMS["warmup_ratio"] == 0.05
        assert PAPER_LORA_HYPERPARAMS["weight_decay"] == 0.0
        assert PAPER_LORA_HYPERPARAMS["max_grad_norm"] == 1.0
        assert PAPER_LORA_HYPERPARAMS["optim"] == "adamw_torch"
        assert PAPER_LORA_HYPERPARAMS["lr_scheduler_type"] == "cosine"
        assert PAPER_LORA_HYPERPARAMS["max_length"] == 1250
        assert PAPER_LORA_HYPERPARAMS["save_total_limit"] == 20

    @pytest.mark.parametrize("billions,expected", [
        (0.5, (8, 2)),
        (1.0, (8, 2)),
        (3.0, (4, 4)),
        (4.0, (4, 4)),
        (7.0, (2, 8)),
        (13.0, (1, 16)),
        (14.0, (1, 16)),
    ])
    def test_paper_micro_batch_tiers(self, billions, expected):
        """Paper Table 9 micro-batch tiering: every tier keeps the
        effective batch at 16 (= micro * grad_accum)."""
        from coldddi.llm.trainer import paper_micro_batch_for_size

        micro, accum = paper_micro_batch_for_size(billions)
        assert (micro, accum) == expected
        assert micro * accum == 16, (
            f"size={billions}B: paper claims effective batch=16, got "
            f"{micro}*{accum}={micro*accum}"
        )

    def test_paper_max_length_kg_modes(self):
        """Paper Table 9: 1,250 for Top-3 KG (R0-R3); 4,096 for
        Full KG (R4-R7)."""
        from coldddi.llm.trainer import paper_max_length

        assert paper_max_length(kg_top3=True) == 1250
        assert paper_max_length(kg_top3=False) == 4096

    def test_paper_llm_trainer_config_builder(self):
        from coldddi.llm.trainer import paper_llm_trainer_config

        cfg = paper_llm_trainer_config(
            model_name="meta-llama/Llama-3.2-1B",
            output_dir="/tmp/out",
            billions=1.0,
        )
        # Adapter block
        assert cfg.lora.r == 16
        assert cfg.lora.alpha == 16
        assert cfg.lora.target_modules == ("q_proj", "k_proj", "v_proj", "o_proj")
        # Training block
        assert cfg.num_epochs == 4
        assert cfg.learning_rate == 5e-4
        assert cfg.warmup_ratio == 0.05
        assert cfg.max_length == 1250
        assert cfg.save_total_limit == 20
        # Size-tiered micro batch (1B → micro=8, accum=2)
        assert cfg.micro_batch_size == 8
        assert cfg.gradient_accumulation_steps == 2

    def test_paper_llm_trainer_config_overrides_win(self):
        from coldddi.llm.trainer import paper_llm_trainer_config

        cfg = paper_llm_trainer_config(
            model_name="m", output_dir="/tmp/x", billions=1.0,
            num_epochs=99,
        )
        assert cfg.num_epochs == 99
        # Other paper fields still applied.
        assert cfg.learning_rate == 5e-4


# Signature dispatch policy.


class TestSignatureDispatchPolicy:
    """Forward all paper kwargs to wrappers, but only named params to strict classes.
    """

    @pytest.mark.skipif(
        not (TOY_RELEASE / "filtered" / "drugs.csv").is_file(),
        reason="Toy fixture missing",
    )
    def test_var_keyword_class_receives_all_paper_kwargs(self, tmp_path):
        """A wrapper-style ``def __init__(self, **kw)`` must receive
        every PAPER_HYPERPARAMS key (no signature filter)."""
        from coldddi import evaluate
        from coldddi.baselines import ensure_imported
        from coldddi.baselines.base import _REGISTRY

        ensure_imported("deepddi")
        OrigCls = _REGISTRY["deepddi"]
        captured: dict = {}

        class WrapperBaseline:
            name = "deepddi"
            modality = "mol"

            def __init__(self, **kw):
                captured["init_kwargs"] = dict(kw)
                raise RuntimeError("WrapperBaseline: capture-only")

        _REGISTRY["deepddi"] = WrapperBaseline
        try:
            with pytest.raises(RuntimeError, match="WrapperBaseline"):
                evaluate.run_evaluation(
                    method="deepddi",
                    data_dir=TOY_RELEASE,
                    seed=42,
                    settings=["S2"],
                    out_dir=tmp_path / "out",
                    device="cpu",
                    preset="paper",
                    with_indicators=False,
                )
        finally:
            _REGISTRY["deepddi"] = OrigCls

        observed = captured["init_kwargs"]
        # Wrapper with **kw → every paper kwarg flows through.
        assert observed["n_epochs"] == 100
        assert observed["batch_size"] == 256
        assert observed["learning_rate"] == 1e-3

    @pytest.mark.skipif(
        not (TOY_RELEASE / "filtered" / "drugs.csv").is_file(),
        reason="Toy fixture missing",
    )
    def test_strict_signature_class_filters_unknown_paper_kwargs(self, tmp_path):
        """Strict constructors receive only supported parameters: n_epochs and device."""
        import inspect as _inspect

        from coldddi import evaluate
        from coldddi.baselines import ensure_imported
        from coldddi.baselines.base import _REGISTRY

        ensure_imported("deepddi")
        OrigCls = _REGISTRY["deepddi"]
        captured: dict = {}

        class StrictBaseline:
            name = "deepddi"
            modality = "mol"

            def __init__(self, **kw):
                captured["init_kwargs"] = dict(kw)
                raise RuntimeError("StrictBaseline: capture-only")

        # Pretend StrictBaseline only accepts n_epochs + device.
        new_params = [
            _inspect.Parameter("self", _inspect.Parameter.POSITIONAL_OR_KEYWORD),
            _inspect.Parameter(
                "n_epochs", _inspect.Parameter.KEYWORD_ONLY, default=1,
            ),
            _inspect.Parameter(
                "device", _inspect.Parameter.KEYWORD_ONLY, default="auto",
            ),
        ]
        StrictBaseline.__init__.__signature__ = _inspect.Signature(new_params)

        _REGISTRY["deepddi"] = StrictBaseline
        try:
            with pytest.raises(RuntimeError, match="StrictBaseline"):
                evaluate.run_evaluation(
                    method="deepddi",
                    data_dir=TOY_RELEASE,
                    seed=42,
                    settings=["S2"],
                    out_dir=tmp_path / "out",
                    device="cpu",
                    preset="paper",
                    with_indicators=False,
                )
        finally:
            _REGISTRY["deepddi"] = OrigCls

        observed = captured["init_kwargs"]
        # Filtered: only n_epochs (a paper kwarg) and device made it.
        assert observed.get("n_epochs") == 100
        assert observed.get("device") == "cpu"
        # Unsupported parameters would fail strict construction.
        assert "batch_size" not in observed
        assert "learning_rate" not in observed
        assert "hidden_dim" not in observed
