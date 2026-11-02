"""Run stage 2B unit checks without an external pytest installation."""
from pathlib import Path
import tempfile
from fecd.tests.test_box_ops import test_base_k1_equals_k8_and_edge_clipping,test_prompt_variants_are_fixed
from fecd.tests.test_sam_ensemble import test_multi_box_max_and_empty,test_population_stats_and_threshold_votes
from fecd.tests.test_cache_schema import test_atomic_roundtrip_config_and_duplicate_names,test_corrupt_cache_errors,test_tampered_metadata_and_stale_lock
from fecd.tests.test_fecd_dataset import test_synchronized_flip_rotation,test_box_coordinates_follow_discrete_geometry

def main():
 tests=[test_base_k1_equals_k8_and_edge_clipping,test_prompt_variants_are_fixed,test_multi_box_max_and_empty,test_population_stats_and_threshold_votes,test_synchronized_flip_rotation,test_box_coordinates_follow_discrete_geometry]
 for test in tests:test();print(test.__name__+': PASS')
 with tempfile.TemporaryDirectory() as tmp:
  test_atomic_roundtrip_config_and_duplicate_names(Path(tmp));test_corrupt_cache_errors(Path(tmp));test_tampered_metadata_and_stale_lock(Path(tmp));print('cache schema/tamper/lock tests: PASS')
 print(f'{len(tests)+3} checks passed')
if __name__=='__main__':main()
