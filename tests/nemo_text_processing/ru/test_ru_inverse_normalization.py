# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
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


class TestRuInverseNormalize:

    normalizer = InverseNormalizer(lang='ru', cache_dir=CACHE_DIR) if PYNINI_AVAILABLE else None

    @parameterized.expand(parse_test_case_file('ru/data_inverse_text_normalization/test_cases_cardinal.txt'))
    @pytest.mark.skipif(
        not PYNINI_AVAILABLE,
        reason="`pynini` not installed, please install via nemo_text_processing/pynini_install.sh",
    )
    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_denorm_cardinal(self, test_input, expected):
        pred = self.normalizer.inverse_normalize(test_input, verbose=False)
        assert expected == pred

    testing_class_weights = {
        'ru': {
            'cardinal': 1.01,
            'ordinal': 100,
            'decimal': 100,
            'measure': 100,
            'date': 100,
            'word': 1.1,
            'time': 100,
            'money': 100,
            'electronic': 100,
            'telephone': 100,
            'whitelist': 100,
        }
    }
    '''
    excluded_inverse_normalizer_ru = (
        (
            InverseNormalizer(
                lang='ru',
                cache_dir=CACHE_DIR,
                overwrite_cache=False,
                language_excluded_classes={'ru': {'cardinal': True}},
                language_class_weights=testing_class_weights,
            )
        )
        if PYNINI_AVAILABLE
        else None
    )
    @parameterized.expand(parse_test_case_file('ru/data_inverse_text_normalization/test_cases_cardinal.txt'))
    @pytest.mark.skipif(
        not PYNINI_AVAILABLE,
        reason="`pynini` not installed, please install via nemo_text_processing/pynini_install.sh",
    )
    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_excluded_denorm_cardinal(self, test_input, expected):
        pred = self.excluded_inverse_normalizer_ru.inverse_normalize(test_input, verbose=False)
        examples_that_are_the_same = ['ноль', 'я одна', 'девять']
        if expected in examples_that_are_the_same:
            assert pred == expected
        else:
            assert pred != expected
    '''

    @parameterized.expand(parse_test_case_file('ru/data_inverse_text_normalization/test_cases_ordinal.txt'))
    @pytest.mark.skipif(
        not PYNINI_AVAILABLE,
        reason="`pynini` not installed, please install via nemo_text_processing/pynini_install.sh",
    )
    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_denorm_ordinal(self, test_input, expected):
        pred = self.normalizer.inverse_normalize(test_input, verbose=False)
        assert expected == pred

    testing_class_weights['ru']['ordinal'] = 1.01
    testing_class_weights['ru']['cardinal'] = 100
    excluded_inverse_normalizer_ru = (
        (
            InverseNormalizer(
                lang='ru',
                cache_dir=CACHE_DIR,
                overwrite_cache=False,
                language_excluded_classes={'ru': {'ordinal': True}},
                language_class_weights=testing_class_weights,
            )
        )
        if PYNINI_AVAILABLE
        else None
    )
    '''
    @parameterized.expand(parse_test_case_file('ru/data_inverse_text_normalization/test_cases_ordinal.txt'))
    @pytest.mark.skipif(
        not PYNINI_AVAILABLE,
        reason="`pynini` not installed, please install via nemo_text_processing/pynini_install.sh",
    )
    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_excluded_denorm_ordinal(self, test_input, expected):
        pred = self.excluded_inverse_normalizer_ru.inverse_normalize(test_input, verbose=False)
        assert expected != pred

    # @parameterized.expand(parse_test_case_file('ru_data_inverse_text_normalization/test_cases_ordinal_hard.txt'))
    # @pytest.mark.run_only_on('CPU')
    # @pytest.mark.unit
    # def test_denorm_ordinal_hard(self, test_input, expected):
    #     pred = self.normalizer.inverse_normalize(test_input, verbose=False)
    #     assert expected == pred

    @parameterized.expand(parse_test_case_file('ru/data_inverse_text_normalization/test_cases_decimal.txt'))
    @pytest.mark.skipif(
        not PYNINI_AVAILABLE,
        reason="`pynini` not installed, please install via nemo_text_processing/pynini_install.sh",
    )
    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_denorm_decimal(self, test_input, expected):
        pred = self.normalizer.inverse_normalize(test_input, verbose=False)
        assert expected == pred

    testing_class_weights['ru']['decimal'] = 1.01
    testing_class_weights['ru']['ordinal'] = 100
    excluded_inverse_normalizer_ru = (
        (
            InverseNormalizer(
                lang='ru',
                cache_dir=CACHE_DIR,
                overwrite_cache=False,
                language_excluded_classes={'ru': {'decimal': True}},
                language_class_weights=testing_class_weights,
            )
        )
        if PYNINI_AVAILABLE
        else None
    )

    @parameterized.expand(parse_test_case_file('ru/data_inverse_text_normalization/test_cases_decimal.txt'))
    @pytest.mark.skipif(
        not PYNINI_AVAILABLE,
        reason="`pynini` not installed, please install via nemo_text_processing/pynini_install.sh",
    )
    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_excluded_denorm_decimal(self, test_input, expected):
        pred = self.excluded_inverse_normalizer_ru.inverse_normalize(test_input, verbose=False)
        assert expected != pred

    @parameterized.expand(parse_test_case_file('ru/data_inverse_text_normalization/test_cases_electronic.txt'))
    @pytest.mark.skipif(
        not PYNINI_AVAILABLE,
        reason="`pynini` not installed, please install via nemo_text_processing/pynini_install.sh",
    )
    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_denorm_electronic(self, test_input, expected):
        pred = self.normalizer.inverse_normalize(test_input, verbose=False)
        assert expected == pred

    testing_class_weights['ru']['electronic'] = 1.01
    testing_class_weights['ru']['decimal'] = 100
    excluded_inverse_normalizer_ru = (
        (
            InverseNormalizer(
                lang='ru',
                cache_dir=CACHE_DIR,
                overwrite_cache=False,
                language_excluded_classes={'ru': {'electronic': True}},
                language_class_weights=testing_class_weights,
            )
        )
        if PYNINI_AVAILABLE
        else None
    )

    @parameterized.expand(parse_test_case_file('ru/data_inverse_text_normalization/test_cases_electronic.txt'))
    @pytest.mark.skipif(
        not PYNINI_AVAILABLE,
        reason="`pynini` not installed, please install via nemo_text_processing/pynini_install.sh",
    )
    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_excluded_denorm_electronic(self, test_input, expected):
        pred = self.excluded_inverse_normalizer_ru.inverse_normalize(test_input, verbose=False)
        assert expected != pred

    @parameterized.expand(parse_test_case_file('ru/data_inverse_text_normalization/test_cases_date.txt'))
    @pytest.mark.skipif(
        not PYNINI_AVAILABLE,
        reason="`pynini` not installed, please install via nemo_text_processing/pynini_install.sh",
    )
    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_denorm_date(self, test_input, expected):
        pred = self.normalizer.inverse_normalize(test_input, verbose=False)
        assert expected == pred

    testing_class_weights['ru']['date'] = 1.01
    testing_class_weights['ru']['electronic'] = 100
    excluded_inverse_normalizer_ru = (
        (
            InverseNormalizer(
                lang='ru',
                cache_dir=CACHE_DIR,
                overwrite_cache=False,
                language_excluded_classes={'ru': {'date': True}},
                language_class_weights=testing_class_weights,
            )
        )
        if PYNINI_AVAILABLE
        else None
    )

    @parameterized.expand(parse_test_case_file('ru/data_inverse_text_normalization/test_cases_date.txt'))
    @pytest.mark.skipif(
        not PYNINI_AVAILABLE,
        reason="`pynini` not installed, please install via nemo_text_processing/pynini_install.sh",
    )
    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_excluded_denorm_date(self, test_input, expected):
        pred = self.excluded_inverse_normalizer_ru.inverse_normalize(test_input, verbose=False)
        assert expected != pred

    @parameterized.expand(parse_test_case_file('ru/data_inverse_text_normalization/test_cases_measure.txt'))
    @pytest.mark.skipif(
        not PYNINI_AVAILABLE,
        reason="`pynini` not installed, please install via nemo_text_processing/pynini_install.sh",
    )
    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_denorm_measure(self, test_input, expected):
        pred = self.normalizer.inverse_normalize(test_input, verbose=False)
        assert expected == pred

    testing_class_weights['ru']['measure'] = 1.01
    testing_class_weights['ru']['date'] = 100
    excluded_inverse_normalizer_ru = (
        (
            InverseNormalizer(
                lang='ru',
                cache_dir=CACHE_DIR,
                overwrite_cache=False,
                language_excluded_classes={'ru': {'measure': True}},
                language_class_weights=testing_class_weights,
            )
        )
        if PYNINI_AVAILABLE
        else None
    )
    
    @parameterized.expand(parse_test_case_file('ru/data_inverse_text_normalization/test_cases_measure.txt'))
    @pytest.mark.skipif(
        not PYNINI_AVAILABLE,
        reason="`pynini` not installed, please install via nemo_text_processing/pynini_install.sh",
    )
    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_excluded_denorm_measure(self, test_input, expected):
        pred = self.excluded_inverse_normalizer_ru.inverse_normalize(test_input, verbose=False)
        assert expected != pred

    
    @parameterized.expand(parse_test_case_file('ru/data_inverse_text_normalization/test_cases_money.txt'))
    @pytest.mark.skipif(
        not PYNINI_AVAILABLE,
        reason="`pynini` not installed, please install via nemo_text_processing/pynini_install.sh",
    )
    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_denorm_money(self, test_input, expected):
        pred = self.normalizer.inverse_normalize(test_input, verbose=False)
        assert expected == pred

    testing_class_weights['ru']['money'] = 1.01
    testing_class_weights['ru']['measure'] = 100
    excluded_inverse_normalizer_ru = (
        (
            InverseNormalizer(
                lang='ru',
                cache_dir=CACHE_DIR,
                overwrite_cache=False,
                language_excluded_classes={'ru': {'money': True}},
                language_class_weights=testing_class_weights,
            )
        )
        if PYNINI_AVAILABLE
        else None
    )

    @parameterized.expand(parse_test_case_file('ru/data_inverse_text_normalization/test_cases_money.txt'))
    @pytest.mark.skipif(
        not PYNINI_AVAILABLE,
        reason="`pynini` not installed, please install via nemo_text_processing/pynini_install.sh",
    )
    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_excluded_denorm_money(self, test_input, expected):
        pred = self.excluded_inverse_normalizer_ru.inverse_normalize(test_input, verbose=False)
        assert expected != pred

    @parameterized.expand(parse_test_case_file('ru/data_inverse_text_normalization/test_cases_time.txt'))
    @pytest.mark.skipif(
        not PYNINI_AVAILABLE,
        reason="`pynini` not installed, please install via nemo_text_processing/pynini_install.sh",
    )
    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_denorm_time(self, test_input, expected):
        pred = self.normalizer.inverse_normalize(test_input, verbose=False)
        assert expected == pred

    testing_class_weights['ru']['time'] = 1.01
    testing_class_weights['ru']['money'] = 100
    excluded_inverse_normalizer_ru = (
        (
            InverseNormalizer(
                lang='ru',
                cache_dir=CACHE_DIR,
                overwrite_cache=False,
                language_excluded_classes={'ru': {'time': True}},
                language_class_weights=testing_class_weights,
            )
        )
        if PYNINI_AVAILABLE
        else None
    )

    @parameterized.expand(parse_test_case_file('ru/data_inverse_text_normalization/test_cases_time.txt'))
    @pytest.mark.skipif(
        not PYNINI_AVAILABLE,
        reason="`pynini` not installed, please install via nemo_text_processing/pynini_install.sh",
    )
    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_excluded_denorm_time(self, test_input, expected):
        pred = self.excluded_inverse_normalizer_ru.inverse_normalize(test_input, verbose=False)
        assert expected != pred

    @parameterized.expand(parse_test_case_file('ru/data_inverse_text_normalization/test_cases_whitelist.txt'))
    @pytest.mark.skipif(
        not PYNINI_AVAILABLE,
        reason="`pynini` not installed, please install via nemo_text_processing/pynini_install.sh",
    )
    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_denorm_whitelist(self, test_input, expected):
        pred = self.normalizer.inverse_normalize(test_input, verbose=False)
        assert expected == pred

    testing_class_weights['ru']['whitelist'] = 1.01
    testing_class_weights['ru']['time'] = 100
    excluded_inverse_normalizer_ru = (
        (
            InverseNormalizer(
                lang='ru',
                cache_dir=CACHE_DIR,
                overwrite_cache=False,
                language_excluded_classes={'ru': {'whitelist': True}},
                language_class_weights=testing_class_weights,
            )
        )
        if PYNINI_AVAILABLE
        else None
    )

    @parameterized.expand(parse_test_case_file('ru/data_inverse_text_normalization/test_cases_whitelist.txt'))
    @pytest.mark.skipif(
        not PYNINI_AVAILABLE,
        reason="`pynini` not installed, please install via nemo_text_processing/pynini_install.sh",
    )
    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_excluded_denorm_whitelist(self, test_input, expected):
        pred = self.excluded_inverse_normalizer_ru.inverse_normalize(test_input, verbose=False)
        assert expected != pred

    @parameterized.expand(parse_test_case_file('ru/data_inverse_text_normalization/test_cases_word.txt'))
    @pytest.mark.skipif(
        not PYNINI_AVAILABLE,
        reason="`pynini` not installed, please install via nemo_text_processing/pynini_install.sh",
    )
    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_denorm_word(self, test_input, expected):
        pred = self.normalizer.inverse_normalize(test_input, verbose=False)
        assert expected == pred

    testing_class_weights['ru']['word'] = 1.01
    testing_class_weights['ru']['whitelist'] = 100
    excluded_inverse_normalizer_ru = (
        (
            InverseNormalizer(
                lang='ru',
                cache_dir=CACHE_DIR,
                overwrite_cache=False,
                language_excluded_classes={'ru': {'word': True}},
                language_class_weights=testing_class_weights,
            )
        )
        if PYNINI_AVAILABLE
        else None
    )

    @parameterized.expand(parse_test_case_file('ru/data_inverse_text_normalization/test_cases_exclude_word.txt'))
    @pytest.mark.skipif(
        not PYNINI_AVAILABLE,
        reason="`pynini` not installed, please install via nemo_text_processing/pynini_install.sh",
    )
    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_excluded_denorm_word(self, test_input, expected):
        pred = self.excluded_inverse_normalizer_ru.inverse_normalize(test_input, verbose=False)
        assert expected == pred
'''
