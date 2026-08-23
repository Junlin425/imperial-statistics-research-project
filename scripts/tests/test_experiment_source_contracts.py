import unittest
from pathlib import Path


SCRIPTS_DIR = Path(__file__).resolve().parents[1]

INFERENCE_SCRIPTS = [
    "08_abc_manual_sv.py",
    "11_abc_auto_sv.py",
    "12_abc_wasserstein_sv.py",
    "15_manual_abc_synthetic.py",
    "16_auto_abc_synthetic.py",
    "17_wasserstein_abc_synthetic.py",
]


class FairInferenceConfigurationTests(unittest.TestCase):

    def test_all_inference_scripts_share_budget_and_acceptance_rule(self):
        for filename in INFERENCE_SCRIPTS:
            with self.subTest(filename=filename):
                source = (SCRIPTS_DIR / filename).read_text(
                    encoding="utf-8",
                )
                self.assertIn("N_SIMULATIONS = 10_000", source)
                self.assertIn("ACCEPTANCE_FRACTION = 0.05", source)
                self.assertIn("RANDOM_SEED = 42", source)

    def test_refactored_runners_use_shared_experiment_modules(self):
        expected_tokens = {
            "08_abc_manual_sv.py": [
                "run_manual_rejection_abc",
                "save_manual_abc_result",
                "script_output",
            ],
            "15_manual_abc_synthetic.py": [
                "run_manual_rejection_abc",
                "save_manual_abc_result",
                "script_output",
            ],
            "12_abc_wasserstein_sv.py": [
                "run_wasserstein_rejection_abc",
                "save_wasserstein_abc_result",
                "script_output",
            ],
            "17_wasserstein_abc_synthetic.py": [
                "run_wasserstein_rejection_abc",
                "save_wasserstein_abc_result",
                "script_output",
            ],
        }
        for filename, tokens in expected_tokens.items():
            source = (SCRIPTS_DIR / filename).read_text(encoding="utf-8")
            for token in tokens:
                with self.subTest(filename=filename, token=token):
                    self.assertIn(token, source)

    def test_wasserstein_scripts_no_longer_use_pilot_threshold(self):
        for filename in [
            "12_abc_wasserstein_sv.py",
            "17_wasserstein_abc_synthetic.py",
        ]:
            with self.subTest(filename=filename):
                source = (SCRIPTS_DIR / filename).read_text(
                    encoding="utf-8",
                )
                self.assertNotIn("Pilot Run", source)
                self.assertNotIn("np.percentile", source)


class LearnedSummaryMetricTests(unittest.TestCase):

    def test_auto_scripts_use_the_shared_sv_core(self):
        training_source = (SCRIPTS_DIR / "auto_summary.py").read_text(
            encoding="utf-8"
        )
        experiment_source = (
            SCRIPTS_DIR / "auto_abc_experiment.py"
        ).read_text(encoding="utf-8")

        for token in [
            "from sv_abc_core import",
            "is_valid_return_series",
            "sample_prior",
            "simulate_sv",
        ]:
            with self.subTest(module="auto_summary.py", token=token):
                self.assertIn(token, training_source)

        for token in [
            "from sv_abc_core import",
            "sample_prior",
            "simulate_sv",
        ]:
            with self.subTest(module="auto_abc_experiment.py", token=token):
                self.assertIn(token, experiment_source)

        expected_calls = {
            "10_train_auto_summary_sv.py": "train_auto_summary_model",
            "11_abc_auto_sv.py": "run_auto_rejection_abc",
            "16_auto_abc_synthetic.py": "run_auto_rejection_abc",
        }
        for filename, function_name in expected_calls.items():
            source = (SCRIPTS_DIR / filename).read_text(encoding="utf-8")
            with self.subTest(filename=filename):
                self.assertIn(function_name, source)
                self.assertIn('if __name__ == "__main__":', source)

    def test_training_script_saves_per_parameter_metrics(self):
        runner_source = (
            SCRIPTS_DIR / "10_train_auto_summary_sv.py"
        ).read_text(encoding="utf-8")
        training_source = (SCRIPTS_DIR / "auto_summary.py").read_text(
            encoding="utf-8"
        )

        self.assertIn("mean_absolute_error", training_source)
        self.assertIn("root_mean_squared_error", training_source)
        self.assertIn("auto_summary_validation_metrics.csv", runner_source)
        for parameter in ["alpha", "beta", "sigma_eta"]:
            self.assertIn(f'"{parameter}"', training_source)

    def test_ablation_runner_uses_the_formal_shared_settings(self):
        source = (
            SCRIPTS_DIR / "28_learned_summary_ablation.py"
        ).read_text(encoding="utf-8")

        for token in [
            "TRAINING_SIMULATIONS = 5_000",
            "TRAINING_SERIES_LENGTH = 4_000",
            "INFERENCE_SIMULATIONS = 10_000",
            "ACCEPTANCE_FRACTION = 0.05",
            "RANDOM_SEED = 42",
            "N_ESTIMATORS = 300",
            "run_learned_summary_ablation",
            "save_ablation_outputs",
            'if __name__ == "__main__":',
        ]:
            with self.subTest(token=token):
                self.assertIn(token, source)

    def test_temporal_sensitivity_runner_uses_the_approved_design(self):
        source = (
            SCRIPTS_DIR / "29_temporal_order_sensitivity.py"
        ).read_text(encoding="utf-8")

        for token in [
            "BLOCK_LENGTHS = (1, 5, 20, 100)",
            "N_PERMUTATIONS = 100",
            "SCALE_SIMULATIONS = 2_000",
            "evaluate_temporal_sensitivity",
            "summarise_temporal_sensitivity",
            "save_temporal_sensitivity_outputs",
            "estimate_prior_predictive_summary_scale",
            'if __name__ == "__main__":',
        ]:
            with self.subTest(token=token):
                self.assertIn(token, source)


class RepeatedConvergenceTests(unittest.TestCase):

    def test_convergence_script_uses_ten_seeds_and_saves_raw_results(self):
        source = (
            SCRIPTS_DIR / "21_convergence_study.py"
        ).read_text(encoding="utf-8")

        self.assertIn("CONVERGENCE_SEEDS = list(range(10))", source)
        self.assertIn("evaluate_convergence_prefixes", source)
        self.assertIn("convergence_study_raw.csv", source)
        self.assertIn("convergence_study.csv", source)


class ABCSMCRunnerContractTests(unittest.TestCase):
    def test_synthetic_runner_uses_the_approved_shared_configuration(self):
        source = (
            SCRIPTS_DIR / "24_abc_smc_synthetic.py"
        ).read_text(encoding="utf-8")

        for token in [
            "N_PARTICLES = 500",
            "N_PILOT = 2_000",
            "MAX_POPULATIONS = 5",
            "EPSILON_QUANTILE = 0.50",
            "MAX_ATTEMPTS_PER_POPULATION = 50_000",
            "RANDOM_SEED = 42",
            "MIN_KERNEL_SCALE_FRACTION = 0.01",
            "run_abc_smc",
            "save_abc_smc_result",
            "Path(__file__).resolve().parent",
            '"abc_smc_synthetic_sv"',
        ]:
            with self.subTest(token=token):
                self.assertIn(token, source)

    def test_real_runner_uses_the_same_approved_configuration(self):
        source = (
            SCRIPTS_DIR / "25_abc_smc_sv.py"
        ).read_text(encoding="utf-8")

        for token in [
            "N_PARTICLES = 500",
            "N_PILOT = 2_000",
            "MAX_POPULATIONS = 5",
            "EPSILON_QUANTILE = 0.50",
            "MAX_ATTEMPTS_PER_POPULATION = 50_000",
            "RANDOM_SEED = 42",
            "MIN_KERNEL_SCALE_FRACTION = 0.01",
            "run_abc_smc",
            "save_abc_smc_result",
            "Path(__file__).resolve().parent",
            '"sp500_returns.csv"',
            '"abc_smc_sv"',
        ]:
            with self.subTest(token=token):
                self.assertIn(token, source)


class PosteriorPredictiveRunnerContractTests(unittest.TestCase):
    def test_numbered_ppc_scripts_use_the_shared_complete_runner(self):
        for filename in [
            "20_posterior_predictive_check.py",
            "23_posterior_predictive_pvalue.py",
        ]:
            with self.subTest(filename=filename):
                source = (SCRIPTS_DIR / filename).read_text(encoding="utf-8")
                self.assertIn("from posterior_predictive_utils import", source)
                self.assertIn("run_posterior_predictive_analysis", source)
                self.assertIn("DEFAULT_REPLICATIONS", source)
                self.assertIn("DEFAULT_SEED", source)
                self.assertIn('if __name__ == "__main__":', source)

    def test_script_20_no_longer_contains_stale_posterior_means(self):
        source = (
            SCRIPTS_DIR / "20_posterior_predictive_check.py"
        ).read_text(encoding="utf-8")
        for stale_literal in ["-0.6088", "-0.3982", "-0.9783"]:
            with self.subTest(stale_literal=stale_literal):
                self.assertNotIn(stale_literal, source)


if __name__ == "__main__":
    unittest.main()
