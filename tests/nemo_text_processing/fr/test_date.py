# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pytest
from parameterized import parameterized

from ..utils import CACHE_DIR, parse_test_case_file

try:
    from nemo_text_processing.inverse_text_normalization.inverse_normalize import InverseNormalizer

    PYNINI_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    PYNINI_AVAILABLE = False


class TestDate:
    inverse_normalizer = (
        InverseNormalizer(lang='fr', cache_dir=CACHE_DIR, overwrite_cache=False) if PYNINI_AVAILABLE else None
    )

    @parameterized.expand(parse_test_case_file('fr/data_inverse_text_normalization/test_cases_date.txt'))
    @pytest.mark.skipif(
        not PYNINI_AVAILABLE,
        reason="`pynini` not installed, please install via nemo_text_processing/pynini_install.sh",
    )
    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_denorm(self, test_input, expected):
        pred = self.inverse_normalizer.inverse_normalize(test_input, verbose=False)
        assert pred == expected

    testing_class_weights = {
        'fr': {
            'cardinal': 100,
            'ordinal': 100,
            'decimal': 100,
            'measure': 100,
            'date': 1.01,
            'word': 1.1,
            'time': 100,
            'money': 100,
            'electronic': 100,
            'telephone': 100,
            'whitelist': 100,
        }
    }
    excluded_inverse_normalizer_fr = (
        (
            InverseNormalizer(
                lang='fr',
                cache_dir=CACHE_DIR,
                overwrite_cache=False,
                language_excluded_classes={'fr': {'date': True}},
                language_class_weights=testing_class_weights,
            )
        )
        if PYNINI_AVAILABLE
        else None
    )

    @parameterized.expand(parse_test_case_file('fr/data_inverse_text_normalization/test_cases_date.txt'))
    @pytest.mark.skipif(
        not PYNINI_AVAILABLE,
        reason="`pynini` not installed, please install via nemo_text_processing/pynini_install.sh",
    )
    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_excluded_denorm(self, test_input, expected):
        pred = self.excluded_inverse_normalizer_fr.inverse_normalize(test_input, verbose=False)
        assert pred != expected
