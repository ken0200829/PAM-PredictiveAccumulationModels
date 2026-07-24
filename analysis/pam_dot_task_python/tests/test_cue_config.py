import math
import unittest

from pam_dot_task_python.config import (
    CUE_FORMULATION_VERSION,
    CUE_EFFECT_PRIOR_COVERAGE_Z,
    CUE_EFFECT_PRIOR_VARIANCES,
    CUE_EFFECT_STRONG_TRANSFORMED,
    CUE_MODEL_SPECS,
    CUE_PRIOR_VERSION,
    cue_model_spec,
    cue_registry_digest,
    cue_registry_manifest,
    ddm_prior,
)


class CueModelRegistryTests(unittest.TestCase):
    EXPECTED_EFFECTS = {
        "cue_history_w": ("b_H_w",),
        "cue_parallel_w": ("b_H_w", "gamma_w"),
        "cue_parallel_vbias": ("b_H_w", "gamma_v0"),
        "cue_parallel_w_vbias": ("b_H_w", "gamma_w", "gamma_v0"),
        "cue_integrated_w": ("b_w",),
        "cue_integrated_vbias": ("b_v",),
        "cue_integrated_w_vbias": ("b_w", "b_v"),
    }

    def test_registry_model_ids_have_exact_effect_sets(self):
        self.assertEqual(CUE_FORMULATION_VERSION, "cue-locus-0.2.0")
        self.assertIn("candidate", CUE_PRIOR_VERSION)
        self.assertEqual(set(CUE_MODEL_SPECS), set(self.EXPECTED_EFFECTS))
        for model_id, effects in self.EXPECTED_EFFECTS.items():
            with self.subTest(model_id=model_id):
                spec = cue_model_spec(model_id)
                self.assertEqual(spec.model_id, model_id)
                self.assertEqual(spec.response_effects, effects)

    def test_registry_is_immutable(self):
        with self.assertRaises(TypeError):
            CUE_MODEL_SPECS["mutant"] = CUE_MODEL_SPECS["cue_history_w"]

    def test_registry_digest_is_deterministic_and_versioned(self):
        manifest = cue_registry_manifest()
        self.assertEqual(
            manifest["formulation_version"], CUE_FORMULATION_VERSION
        )
        self.assertEqual(manifest["prior_version"], CUE_PRIOR_VERSION)
        self.assertEqual(set(manifest["models"]), set(self.EXPECTED_EFFECTS))
        first = cue_registry_digest()
        second = cue_registry_digest()
        self.assertEqual(first, second)
        self.assertEqual(len(first), 64)

    def test_prior_names_are_generated_from_the_registry(self):
        for model_id, effects in self.EXPECTED_EFFECTS.items():
            with self.subTest(model_id=model_id):
                prior = ddm_prior(model_id)
                self.assertEqual(
                    prior.names,
                    ("log_a_a", "log_a_v") + effects + ("b_c", "Ter_logit"),
                )
                self.assertEqual(prior.free_names, prior.names)

    def test_effect_prior_widths_put_strong_effect_at_central_95_percent(self):
        for parameter, strong_value in CUE_EFFECT_STRONG_TRANSFORMED.items():
            with self.subTest(parameter=parameter):
                standard_deviation = math.sqrt(CUE_EFFECT_PRIOR_VARIANCES[parameter])
                self.assertAlmostEqual(
                    standard_deviation * CUE_EFFECT_PRIOR_COVERAGE_Z,
                    strong_value,
                    places=15,
                )
        for model_id, effects in self.EXPECTED_EFFECTS.items():
            prior = ddm_prior(model_id)
            for parameter in effects:
                if parameter in CUE_EFFECT_PRIOR_VARIANCES:
                    with self.subTest(model_id=model_id, parameter=parameter):
                        self.assertEqual(
                            prior.variances[prior.names.index(parameter)],
                            CUE_EFFECT_PRIOR_VARIANCES[parameter],
                        )

    def test_legacy_and_cue_registry_are_not_aliased(self):
        self.assertIsNone(cue_model_spec("ddm_w_c"))
        self.assertIsNotNone(cue_model_spec("cue_integrated_w"))
        self.assertNotEqual(
            ddm_prior("ddm_w_c").names,
            ddm_prior("cue_integrated_w").names,
        )


if __name__ == "__main__":
    unittest.main()
