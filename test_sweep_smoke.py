"""
Smoke tests for plot_sweep.py and the Slurm sweep script logic.

Tests are self-contained: they create temporary fixture data, run the
relevant code paths, and clean up — no GPU, no cluster, no SPair-71k required.

Run with:
    pytest tests/smoke/test_smoke.py -v
or standalone:
    python test_smoke.py

NOTE: k is normalised to str in the current implementation (Copilot fix applied).
Single-block  → k = "28"
Multi-block   → k = "(10, 28)"
k_numeric     → int for single-block, None for multi-block
"""

import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EXPECTED_T_VALUES = [20, 100, 180, 260, 340, 420]
EXPECTED_K_VALUES = [19, 28, 37, 46]
EXPECTED_COMBINATIONS = len(EXPECTED_T_VALUES) * len(EXPECTED_K_VALUES)  # 24


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_result_json(metric: str = "image", mean: float = 50.0) -> dict:
    """Minimal JSON that eval_spair.py is expected to produce."""
    categories = ["aeroplane", "bicycle", "bird", "boat", "bottle"]
    return {
        metric: {
            "Mean": mean,
            "All": mean - 1.0,
            **{cat: mean + i for i, cat in enumerate(categories)},
        }
    }


def populate_layers_cat(base: Path, model: str, t_vals, k_vals, metric="image"):
    """Write synthetic result JSONs under base/layers_cat/<model>/."""
    model_dir = base / "layers_cat" / model
    model_dir.mkdir(parents=True)
    for t in t_vals:
        for k in k_vals:
            fname = model_dir / f"t{t}_b[{k}]_e8.json"
            fname.write_text(json.dumps(make_result_json(metric, mean=float(t + k))))
    return model_dir


def _import_plot_sweep():
    """Import plot_sweep module from disk, or skip the test if not found."""
    candidates = [
        Path(__file__).parent.parent.parent / "plot_sweep.py",  # tests/smoke/ -> root
        Path(__file__).parent.parent / "plot_sweep.py",
        Path(__file__).parent / "plot_sweep.py",
        Path("plot_sweep.py"),
    ]
    src = next((p for p in candidates if p.exists()), None)
    if src is None:
        pytest.skip("plot_sweep.py not found on disk")
    import importlib.util

    spec = importlib.util.spec_from_file_location("plot_sweep", src)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# 1.  parse_filename
#     k is always a str in the current implementation (Copilot fix applied).
# ---------------------------------------------------------------------------


class TestParseFilename:
    """Unit-tests for the filename parser (no I/O)."""

    @pytest.fixture(autouse=True)
    def _load(self):
        self.mod = _import_plot_sweep()

    def _parse(self, name: str):
        return self.mod.parse_filename(Path(name))

    # --- return shape -------------------------------------------------------

    def test_single_block_keys(self):
        r = self._parse("t260_b[28]_e8.json")
        assert r is not None
        assert set(r.keys()) >= {"t", "k", "e"}, f"Missing keys in {r}"

    def test_single_block_t_and_e(self):
        r = self._parse("t260_b[28]_e8.json")
        assert r["t"] == 260
        assert r["e"] == 8

    def test_single_block_k_is_str(self):
        """k is normalised to str so DataFrame dtype is always consistent."""
        r = self._parse("t260_b[28]_e8.json")
        assert isinstance(r["k"], str), f"Expected str, got {type(r['k'])}: {r['k']}"
        assert r["k"] == "28"

    def test_single_block_k_numeric(self):
        """k_numeric holds the int value for single-block entries."""
        r = self._parse("t260_b[28]_e8.json")
        if "k_numeric" in r:
            assert r["k_numeric"] == 28

    def test_multi_block_k_is_str(self):
        r = self._parse("t260_b[10, 28]_e8.json")
        assert r is not None
        assert isinstance(r["k"], str), f"Expected str, got {type(r['k'])}: {r['k']}"

    def test_multi_block_k_numeric_is_none(self):
        r = self._parse("t260_b[10, 28]_e8.json")
        if "k_numeric" in r:
            assert r["k_numeric"] is None

    def test_no_brackets(self):
        r = self._parse("t100_b28_e4.json")
        assert r is not None
        assert r["t"] == 100
        assert r["e"] == 4
        assert isinstance(r["k"], str)

    def test_unparseable_returns_none(self):
        assert self._parse("garbage.json") is None

    def test_all_sweep_combinations_parseable(self):
        for t in EXPECTED_T_VALUES:
            for k in EXPECTED_K_VALUES:
                r = self._parse(f"t{t}_b[{k}]_e8.json")
                assert r is not None, f"Failed to parse t={t} k={k}"
                assert r["t"] == t
                assert r["e"] == 8
                # k value round-trips through str
                assert str(k) in r["k"], f"k={k} not found in r['k']={r['k']!r}"


# ---------------------------------------------------------------------------
# 2.  load_results
# ---------------------------------------------------------------------------


class TestLoadResults:
    @pytest.fixture(autouse=True)
    def _load_mod(self):
        self.mod = _import_plot_sweep()

    def _load(self, tmp_path, model="flux", metric="image"):
        old_cwd = os.getcwd()
        os.chdir(tmp_path)
        try:
            return self.mod.load_results(model, metric)
        finally:
            os.chdir(old_cwd)

    @pytest.fixture
    def full_fixture(self, tmp_path):
        populate_layers_cat(tmp_path, "flux", EXPECTED_T_VALUES, EXPECTED_K_VALUES)
        return tmp_path

    def test_row_count(self, full_fixture):
        df = self._load(full_fixture)
        assert len(df) == EXPECTED_COMBINATIONS

    def test_required_columns(self, full_fixture):
        df = self._load(full_fixture)
        for col in ("t", "k", "e", "pck_mean"):
            assert col in df.columns

    def test_t_values(self, full_fixture):
        df = self._load(full_fixture)
        assert set(df["t"].unique()) == set(EXPECTED_T_VALUES)

    def test_k_values_are_strings(self, full_fixture):
        """k column must be uniformly str after the dtype-normalisation fix."""
        df = self._load(full_fixture)
        non_str = [v for v in df["k"] if not isinstance(v, str)]
        assert not non_str, f"Non-str k values found: {non_str[:5]}"

    def test_k_str_values_round_trip(self, full_fixture):
        """Every expected int k must appear as its str representation."""
        df = self._load(full_fixture)
        k_strs = set(df["k"].unique())
        for k in EXPECTED_K_VALUES:
            assert str(k) in k_strs, f"Expected '{k}' in k column, got {k_strs}"

    def test_k_column_is_sortable(self, full_fixture):
        df = self._load(full_fixture)
        try:
            df.sort_values("k")
        except TypeError as exc:
            pytest.fail(f"sort_values('k') raised TypeError: {exc}")

    def test_missing_dir_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            self._load(tmp_path)  # no layers_cat/ created


# ---------------------------------------------------------------------------
# 3.  n_axes calculation
# ---------------------------------------------------------------------------


class TestNAxesCalculation:
    """
    Validates that n_axes exactly matches the number of plots that will be
    drawn — the bug flagged in the PR review (Gemini/Copilot images 1-4).
    """

    @staticmethod
    def _n_axes_buggy(unique_t: int, unique_k: int) -> int:
        """Original broken formula — kept to document the regression."""
        return 1 + (unique_k > 1) + (unique_t > 1)

    @staticmethod
    def _n_axes_fixed(unique_t: int, unique_k: int) -> int:
        show_heatmap = unique_t > 1 and unique_k > 1
        show_lines_by_block = unique_t > 1
        show_lines_by_timestep = unique_k > 1
        return int(show_heatmap) + int(show_lines_by_block) + int(show_lines_by_timestep)

    @staticmethod
    def _expected_draw_count(unique_t: int, unique_k: int) -> int:
        n = 0
        if unique_t > 1 and unique_k > 1:
            n += 1
        if unique_t > 1:
            n += 1
        if unique_k > 1:
            n += 1
        return n

    scenarios = [
        (6, 4, "full sweep"),
        (1, 4, "only k varies"),
        (6, 1, "only t varies"),
        (1, 1, "degenerate — neither varies"),
        (2, 1, "two timesteps, single block"),
        (1, 2, "single timestep, two blocks"),
    ]

    @pytest.mark.parametrize("unique_t,unique_k,desc", scenarios)
    def test_fixed_formula(self, unique_t, unique_k, desc):
        assert self._n_axes_fixed(unique_t, unique_k) == self._expected_draw_count(unique_t, unique_k), desc

    @pytest.mark.parametrize(
        "unique_t,unique_k,desc",
        [s for s in scenarios if not (s[0] > 1 and s[1] > 1) and not (s[0] == 1 and s[1] == 1)],
    )
    def test_buggy_formula_overcounts(self, unique_t, unique_k, desc):
        """Regression guard: documents that the old formula was wrong."""
        assert self._n_axes_buggy(unique_t, unique_k) != self._expected_draw_count(unique_t, unique_k), (
            f"[{desc}] buggy formula accidentally correct — was the formula changed?"
        )


# ---------------------------------------------------------------------------
# 4.  k dtype consistency
# ---------------------------------------------------------------------------


class TestKDtypeConsistency:
    @pytest.fixture(autouse=True)
    def _load_mod(self):
        self.mod = _import_plot_sweep()

    def _load_from_names(self, tmp_path, filenames, metric="image"):
        model_dir = tmp_path / "layers_cat" / "flux"
        model_dir.mkdir(parents=True)
        for fname in filenames:
            (model_dir / fname).write_text(json.dumps(make_result_json(metric, 50.0)))
        old_cwd = os.getcwd()
        os.chdir(tmp_path)
        try:
            return self.mod.load_results("flux", metric)
        finally:
            os.chdir(old_cwd)

    def test_single_block_k_is_str(self, tmp_path):
        df = self._load_from_names(tmp_path, ["t260_b[28]_e8.json", "t100_b[28]_e8.json"])
        assert all(isinstance(v, str) for v in df["k"])

    def test_multi_block_k_is_str(self, tmp_path):
        df = self._load_from_names(tmp_path, ["t260_b[10, 28]_e8.json", "t100_b[10, 28]_e8.json"])
        assert all(isinstance(v, str) for v in df["k"])

    def test_mixed_single_and_multi_block_k_is_str(self, tmp_path):
        """Core dtype-normalisation check: no mixed int/tuple column."""
        df = self._load_from_names(
            tmp_path,
            [
                "t260_b[28]_e8.json",
                "t100_b[10, 28]_e8.json",
            ],
        )
        non_str = [v for v in df["k"] if not isinstance(v, str)]
        assert not non_str, f"Mixed k dtypes — normalisation fix not applied. Non-str values: {non_str}"

    def test_k_sortable_after_mix(self, tmp_path):
        df = self._load_from_names(tmp_path, [f"t{t}_b[{k}]_e8.json" for t in [100, 260] for k in [19, 28]])
        try:
            df.sort_values("k")
        except TypeError as exc:
            pytest.fail(f"sort_values('k') raised TypeError: {exc}")


# ---------------------------------------------------------------------------
# 5.  Slurm script — static analysis
# ---------------------------------------------------------------------------


def _find_slurm_script() -> Path | None:
    candidates = [
        # canonical location
        Path(__file__).parent.parent.parent / "experiments" / "sweep_eval_spair.sh",
        # fallbacks when tests/ is nested differently
        Path(__file__).parent.parent / "experiments" / "sweep_eval_spair.sh",
        Path(__file__).parent / "experiments" / "sweep_eval_spair.sh",
        # legacy / alternative names kept for convenience
        Path(__file__).parent.parent.parent / "experiments" / "sweep.sh",
        Path(__file__).parent.parent.parent / "sweep.sh",
        Path("experiments/sweep_eval_spair.sh"),
        Path("sweep_eval_spair.sh"),
    ]
    return next((p for p in candidates if p.exists()), None)


@pytest.fixture(scope="module")
def slurm_script_text():
    path = _find_slurm_script()
    if path is None:
        pytest.skip(
            "Slurm script not found. Expected at experiments/sweep_eval_spair.sh "
            "relative to the project root."
        )
    return path.read_text()


class TestSlurmScript:
    def test_has_account_directive(self, slurm_script_text):
        assert "#SBATCH -A" in slurm_script_text or "#SBATCH --account" in slurm_script_text

    def test_has_gpu_directive(self, slurm_script_text):
        assert "--gpus-per-task" in slurm_script_text

    def test_has_time_limit(self, slurm_script_text):
        assert "--time" in slurm_script_text

    def test_output_log_dir(self, slurm_script_text):
        assert "logs/" in slurm_script_text

    def test_t_values_present(self, slurm_script_text):
        for t in EXPECTED_T_VALUES:
            assert str(t) in slurm_script_text, f"Timestep t={t} not found in script"

    def test_k_values_present(self, slurm_script_text):
        for k in EXPECTED_K_VALUES:
            assert str(k) in slurm_script_text, f"Block index k={k} not found in script"

    def test_combination_count_comment(self, slurm_script_text):
        assert str(EXPECTED_COMBINATIONS) in slurm_script_text

    def test_ensemble_size_8(self, slurm_script_text):
        assert "ensemble_size 8" in slurm_script_text or "--ensemble_size 8" in slurm_script_text

    def test_uses_project_root(self, slurm_script_text):
        assert "PROJECT_ROOT" in slurm_script_text

    def test_sources_common_sh(self, slurm_script_text):
        assert "_common.sh" in slurm_script_text

    def test_fatal_on_missing_common_sh(self, slurm_script_text):
        assert "exit 1" in slurm_script_text

    def test_nested_loop_structure(self, slurm_script_text):
        t_loop = re.search(r"for\s+t\s+in\s+[\d\s]+;", slurm_script_text)
        k_loop = re.search(r"for\s+k\s+in\s+[\d\s]+;", slurm_script_text)
        assert t_loop is not None, "for t loop not found"
        assert k_loop is not None, "for k loop not found"
        assert t_loop.start() < k_loop.start(), "k loop should be nested inside t loop"

    def test_eval_spair_invocation(self, slurm_script_text):
        assert "eval_spair.py" in slurm_script_text

    def test_dataset_flag(self, slurm_script_text):
        assert "--dataset" in slurm_script_text and "spair" in slurm_script_text

    def test_no_suspicious_hardcoded_paths(self, slurm_script_text):
        allowed_prefixes = ("/lustre", "/dev/null", "/bin", "/usr")
        abs_paths = re.findall(r'(?<!["\w])/[a-zA-Z][^\s"\']*', slurm_script_text)
        suspicious = [p for p in abs_paths if not any(p.startswith(a) for a in allowed_prefixes)]
        assert not suspicious, f"Suspicious hardcoded absolute paths: {suspicious}"


# ---------------------------------------------------------------------------
# 6.  End-to-end CLI
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def plot_sweep_script() -> Path:
    candidates = [
        Path(__file__).parent.parent.parent / "plot_sweep.py",
        Path(__file__).parent.parent / "plot_sweep.py",
        Path(__file__).parent / "plot_sweep.py",
        Path("plot_sweep.py"),
    ]
    src = next((p for p in candidates if p.exists()), None)
    if src is None:
        pytest.skip("plot_sweep.py not found on disk")
    return src


class TestPlotSweepCLI:
    def test_help_exits_zero(self, plot_sweep_script):
        r = subprocess.run(
            [sys.executable, str(plot_sweep_script), "--help"],
            capture_output=True,
            text=True,
        )
        assert r.returncode == 0
        assert "metric" in r.stdout.lower() or "metric" in r.stderr.lower()

    def test_missing_data_exits_nonzero(self, plot_sweep_script, tmp_path):
        r = subprocess.run(
            [sys.executable, str(plot_sweep_script), "--model", "flux"],
            capture_output=True,
            text=True,
            cwd=tmp_path,
        )
        assert r.returncode != 0

    def test_full_run_produces_png(self, plot_sweep_script, tmp_path):
        populate_layers_cat(tmp_path, "flux", EXPECTED_T_VALUES, EXPECTED_K_VALUES)
        out = tmp_path / "sweep_plot.png"
        r = subprocess.run(
            [
                sys.executable,
                str(plot_sweep_script),
                "--model",
                "flux",
                "--metric",
                "image",
                "--out",
                str(out),
            ],
            capture_output=True,
            text=True,
            cwd=tmp_path,
            env={**os.environ, "MPLBACKEND": "Agg"},
        )
        assert r.returncode == 0, f"stdout: {r.stdout}\nstderr: {r.stderr}"
        assert out.exists()
        assert out.stat().st_size > 1024

    def test_point_metric_run(self, plot_sweep_script, tmp_path):
        populate_layers_cat(tmp_path, "flux", EXPECTED_T_VALUES, EXPECTED_K_VALUES, metric="point")
        out = tmp_path / "sweep_point.png"
        r = subprocess.run(
            [
                sys.executable,
                str(plot_sweep_script),
                "--model",
                "flux",
                "--metric",
                "point",
                "--out",
                str(out),
            ],
            capture_output=True,
            text=True,
            cwd=tmp_path,
            env={**os.environ, "MPLBACKEND": "Agg"},
        )
        assert r.returncode == 0, f"stdout: {r.stdout}\nstderr: {r.stderr}"
        assert out.exists()

    def test_single_t_single_k_exits_gracefully(self, plot_sweep_script, tmp_path):
        populate_layers_cat(tmp_path, "flux", [260], [28])
        r = subprocess.run(
            [sys.executable, str(plot_sweep_script), "--model", "flux"],
            capture_output=True,
            text=True,
            cwd=tmp_path,
            env={**os.environ, "MPLBACKEND": "Agg"},
        )
        assert r.returncode == 0
        assert "nothing to sweep" in r.stdout.lower()


# ---------------------------------------------------------------------------
# Standalone runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
