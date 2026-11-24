"""Protocol tests intentionally require only Python's standard library."""

import json
import os
from pathlib import Path
import random
import subprocess
import sys
import unittest

from foresight.research.diagnostic_protocol import (
    ALLOWED_VARIANTS,
    deterministic_bin_permutation,
    make_event_donor_map,
    validate_variants,
)


class EventDonorMapTests(unittest.TestCase):
    def setUp(self):
        # Unequal sequence sizes prevent a one-to-one cross-sequence derangement.
        self.ids = [f"long:sequence:{i}" for i in range(17)] + ["short:1", "third:8"]

    def test_all_donors_are_existing_foreign_sequence_windows(self):
        mapping = make_event_donor_map(self.ids, 42)
        self.assertEqual(set(mapping), set(self.ids))
        for recipient, donor in mapping.items():
            with self.subTest(recipient=recipient):
                self.assertIn(donor, self.ids)
                self.assertNotEqual(recipient.rsplit(":", 1)[0], donor.rsplit(":", 1)[0])

    def test_inventory_order_and_global_rng_do_not_change_mapping(self):
        expected = make_event_donor_map(self.ids, 42)
        reordered = self.ids.copy()
        random.Random(13).shuffle(reordered)
        rng_state = random.getstate()
        self.assertEqual(make_event_donor_map(reordered, 42), expected)
        self.assertEqual(random.getstate(), rng_state)
        self.assertEqual(list(make_event_donor_map(reordered, 42)), list(expected))

    def test_full_inventory_mapping_is_reused_for_each_batch_size(self):
        mapping = make_event_donor_map(self.ids, 42)
        expected = [mapping[sample_id] for sample_id in self.ids]
        for batch_size in (1, 2, 7, 100):
            with self.subTest(batch_size=batch_size):
                actual = []
                for start in range(0, len(self.ids), batch_size):
                    actual.extend(mapping[sample_id] for sample_id in self.ids[start:start + batch_size])
                self.assertEqual(actual, expected)
        # A recipient subset does not constrain its donor to the same subset.
        self.assertNotIn(mapping[self.ids[0]], self.ids[:5])

    def test_seed_changes_donors_when_multiple_candidates_exist(self):
        self.assertNotEqual(make_event_donor_map(self.ids, 42), make_event_donor_map(self.ids, 43))

    def test_single_window_in_each_sequence_is_valid(self):
        self.assertEqual(make_event_donor_map(["a:1", "b:2"], 42), {"a:1": "b:2", "b:2": "a:1"})

    def test_invalid_inventories_fail_explicitly(self):
        cases = [[], "a:1", ["a:1", "a:2"], ["a:1", "b:1", "a:1"],
                 ["a", "b:1"], [":1", "b:1"], ["a:", "b:1"], [1, "b:1"]]
        for sample_ids in cases:
            with self.subTest(sample_ids=sample_ids), self.assertRaises(ValueError):
                make_event_donor_map(sample_ids, 42)

    def test_reproducible_across_process_hash_seeds_without_torch_import(self):
        root = Path(__file__).resolve().parents[1]
        program = (
            "import json,sys; "
            "from foresight.research.diagnostic_protocol import make_event_donor_map; "
            "assert 'torch' not in sys.modules and 'numpy' not in sys.modules; "
            "print(json.dumps(make_event_donor_map(['a:1','a:2','b:1','c:1'],42),sort_keys=True))"
        )
        outputs = []
        for hash_seed in ("1", "87"):
            env = {**os.environ, "PYTHONHASHSEED": hash_seed, "PYTHONPATH": str(root)}
            outputs.append(subprocess.check_output([sys.executable, "-c", program], cwd=root, env=env, text=True))
        self.assertEqual(json.loads(outputs[0]), json.loads(outputs[1]))


class BinPermutationTests(unittest.TestCase):
    def test_nonidentity_valid_permutation(self):
        for bins in (2, 3, 5, 15):
            for seed in range(30):
                with self.subTest(bins=bins, seed=seed):
                    order = deterministic_bin_permutation(bins, seed, "sequence:100")
                    self.assertEqual(sorted(order), list(range(bins)))
                    self.assertNotEqual(order, list(range(bins)))

    def test_one_bin_is_identity(self):
        self.assertEqual(deterministic_bin_permutation(1, 42, "sequence:100"), [0])

    def test_repeatability_and_no_global_rng_side_effect(self):
        rng_state = random.getstate()
        expected = deterministic_bin_permutation(5, 42, "sequence:100")
        self.assertEqual(deterministic_bin_permutation(5, 42, "sequence:100"), expected)
        self.assertEqual(random.getstate(), rng_state)

    def test_sample_ids_produce_different_orders(self):
        orders = {tuple(deterministic_bin_permutation(5, 42, f"sequence:{i}")) for i in range(30)}
        self.assertGreater(len(orders), 1)

    def test_invalid_inputs(self):
        for bins in (0, -1, True, 2.5):
            with self.subTest(bins=bins), self.assertRaises(ValueError):
                deterministic_bin_permutation(bins, 42, "sequence:100")
        for seed in (True, "42", 4.2):
            with self.subTest(seed=seed), self.assertRaises(ValueError):
                deterministic_bin_permutation(5, seed, "sequence:100")
            with self.subTest(seed=seed), self.assertRaises(ValueError):
                make_event_donor_map(["a:1", "b:1"], seed)


class VariantValidationTests(unittest.TestCase):
    def test_canonical_order(self):
        self.assertEqual(validate_variants(list(reversed(ALLOWED_VARIANTS))), ALLOWED_VARIANTS)
        self.assertEqual(validate_variants(["off", "correct", "zero"]), ("correct", "zero", "off"))

    def test_correct_only_is_valid(self):
        self.assertEqual(validate_variants(["correct"]), ("correct",))

    def test_invalid_variants(self):
        for variants in ([], "correct", ["off"], ["correct", "correct"],
                         ["correct", "unknown"], ["correct", None]):
            with self.subTest(variants=variants), self.assertRaises(ValueError):
                validate_variants(variants)


if __name__ == "__main__":
    unittest.main()
