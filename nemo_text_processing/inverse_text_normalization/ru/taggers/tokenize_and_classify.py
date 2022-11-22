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

import os
from typing import Dict, Optional

import pynini
from nemo_text_processing.inverse_text_normalization.en.taggers.punctuation import PunctuationFst
from nemo_text_processing.inverse_text_normalization.en.taggers.word import WordFst
from nemo_text_processing.inverse_text_normalization.ru.taggers.cardinal import CardinalFst
from nemo_text_processing.inverse_text_normalization.ru.taggers.date import DateFst
from nemo_text_processing.inverse_text_normalization.ru.taggers.decimals import DecimalFst
from nemo_text_processing.inverse_text_normalization.ru.taggers.electronic import ElectronicFst
from nemo_text_processing.inverse_text_normalization.ru.taggers.measure import MeasureFst
from nemo_text_processing.inverse_text_normalization.ru.taggers.money import MoneyFst
from nemo_text_processing.inverse_text_normalization.ru.taggers.ordinal import OrdinalFst
from nemo_text_processing.inverse_text_normalization.ru.taggers.telephone import TelephoneFst
from nemo_text_processing.inverse_text_normalization.ru.taggers.time import TimeFst
from nemo_text_processing.inverse_text_normalization.ru.taggers.whitelist import WhiteListFst
from nemo_text_processing.text_normalization.en.graph_utils import (
    GraphFst,
    delete_extra_space,
    delete_space,
    generator_main,
)
from nemo_text_processing.text_normalization.ru.taggers.tokenize_and_classify import ClassifyFst as TNClassifyFst
from pynini.lib import pynutil

from nemo.utils import logging


class ClassifyFst(GraphFst):
    """
    Final class that composes all other classification grammars. This class can process an entire sentence, that is lower cased.
    For deployment, this grammar will be compiled and exported to OpenFst Finate State Archiv (FAR) File. 
    More details to deployment at NeMo/tools/text_processing_deployment.

    Args:
        cache_dir: path to a dir with .far grammar file. Set to None to avoid using cache.
        overwrite_cache: set to True to overwrite .far files
        excluded_classes: A dictionary of semiotic class and a boolean indicating if that class should be excluded
            from tagging a word(s) in the given text. By default if None no class will be excluded, a class is
            only excluded if explicitly stated.
        class_weights: The weight to be applied to each of the semiotic class,
            a lower weight gives higher priority to the given class.
            By default the following weights are applied if no weight is given for a class:
            'cardinal': 1.1, 'ordinal': 1.1, 'decimal': 1.1, 'measure': 1.1, 'date': 1.09,
            'word': 100,'time': 1.1,'money': 1.1,'electronic': 1.1,'telephone': 1.1, 'whitelist': 1.01
    """

    def __init__(
        self,
        cache_dir: str = None,
        overwrite_cache: bool = False,
        excluded_classes: Optional[Dict[str, bool]] = None,
        class_weights: Optional[Dict[str, float]] = None,
    ):
        super().__init__(name="tokenize_and_classify", kind="classify")

        if excluded_classes is None:
            excluded_classes = {}

        classes = [
            'cardinal',
            'ordinal',
            'decimal',
            'measure',
            'date',
            'word',
            'time',
            'money',
            'whitelist',
            'electronic',
            'telephone',
        ]
        for excluded_class in excluded_classes:
            if excluded_class not in classes:
                logging.info(f"The class {excluded_class} is not being excluded as the class does not exist.")

        if class_weights is None:
            class_weights = {}
        default_class_weights = {
            'cardinal': 1.1,
            'ordinal': 1.1,
            'decimal': 1.1,
            'measure': 1.1,
            'date': 1.09,
            'word': 100,
            'time': 1.1,
            'money': 1.1,
            'electronic': 1.1,
            'telephone': 1.1,
            'whitelist': 1.01,
        }
        for _class, weight in default_class_weights.items():
            if _class not in class_weights:
                class_weights[_class] = weight

        for class_weight in class_weights:
            if class_weight not in classes:
                logging.info(
                    f"The class {class_weight} does not have a custom class weight applied to it as the class does not exist."
                )

        far_file = None
        if cache_dir is not None and cache_dir != "None":
            os.makedirs(cache_dir, exist_ok=True)
            far_file = os.path.join(cache_dir, "_ru_itn.far")
        if not overwrite_cache and far_file and os.path.exists(far_file):
            self.fst = pynini.Far(far_file, mode="r")["tokenize_and_classify"]
            logging.info(f"ClassifyFst.fst was restored from {far_file}.")
        else:
            logging.info(f"Creating ClassifyFst grammars. This might take some time...")
            empty_fst = pynini.string_map([])
            tn_classify = TNClassifyFst(
                input_case='cased', deterministic=False, cache_dir=cache_dir, overwrite_cache=True
            )
            cardinal = CardinalFst(tn_cardinal=tn_classify.cardinal)
            cardinal_graph = cardinal.fst

            ordinal = OrdinalFst(tn_ordinal=tn_classify.ordinal)
            ordinal_graph = ordinal.fst

            decimal = DecimalFst(tn_decimal=tn_classify.decimal)
            decimal_graph = decimal.fst

            measure_graph = MeasureFst(tn_measure=tn_classify.measure).fst
            date_graph = DateFst(tn_date=tn_classify.date).fst
            word_graph = WordFst().fst
            time_graph = TimeFst(tn_time=tn_classify.time).fst
            money_graph = MoneyFst(tn_money=tn_classify.money).fst
            whitelist_graph = WhiteListFst().fst
            punct_graph = PunctuationFst().fst
            electronic_graph = ElectronicFst(tn_electronic=tn_classify.electronic).fst
            telephone_graph = TelephoneFst(tn_telephone=tn_classify.telephone).fst

            class_graphs = {
                'cardinal': cardinal_graph,
                'ordinal': ordinal_graph,
                'decimal': decimal_graph,
                'measure': measure_graph,
                'date': date_graph,
                'word': word_graph,
                'time': time_graph,
                'money': money_graph,
                'electronic': electronic_graph,
                'telephone': telephone_graph,
                'whitelist': whitelist_graph,
            }
            for _class in class_graphs.keys():
                is_empty_graph = excluded_classes.get(_class, False)
                if is_empty_graph:
                    class_graphs[_class] = empty_fst

            classify = (
                pynutil.add_weight(class_graphs['whitelist'], class_weights['whitelist'])
                | pynutil.add_weight(class_graphs['time'], class_weights['time'])
                | pynutil.add_weight(class_graphs['date'], class_weights['date'])
                | pynutil.add_weight(class_graphs['decimal'], class_weights['decimal'])
                | pynutil.add_weight(class_graphs['measure'], class_weights['measure'])
                | pynutil.add_weight(class_graphs['cardinal'], class_weights['cardinal'])
                | pynutil.add_weight(class_graphs['ordinal'], class_weights['ordinal'])
                | pynutil.add_weight(class_graphs['money'], class_weights['money'])
                | pynutil.add_weight(class_graphs['telephone'], class_weights['telephone'])
                | pynutil.add_weight(class_graphs['electronic'], class_weights['electronic'])
                | pynutil.add_weight(class_graphs['word'], class_weights['word'])
            )

            punct = pynutil.insert("tokens { ") + pynutil.add_weight(punct_graph, weight=1.1) + pynutil.insert(" }")
            token = pynutil.insert("tokens { ") + classify + pynutil.insert(" }")
            token_plus_punct = (
                pynini.closure(punct + pynutil.insert(" ")) + token + pynini.closure(pynutil.insert(" ") + punct)
            )

            graph = token_plus_punct + pynini.closure(pynutil.add_weight(delete_extra_space, 1.1) + token_plus_punct)

            graph = delete_space + graph + delete_space
            self.fst = graph.optimize()

            if far_file:
                generator_main(far_file, {"tokenize_and_classify": self.fst})
                logging.info(f"ClassifyFst grammars are saved to {far_file}.")
