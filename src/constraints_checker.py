import numpy
import torch
import pm4py
import pandas
import random
import string
import re
import os
import csv
import requests
import transformers
import torch.nn as nn

from peft.mapping import get_peft_model
from peft.tuners.lora.config import LoraConfig
from transformers.models.auto.configuration_auto import AutoConfig
from transformers.models.auto.tokenization_auto import AutoTokenizer
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, BitsAndBytesConfig
from pyxdameraulevenshtein import normalized_damerau_levenshtein_distance
from transformers.models.gpt2.configuration_gpt2 import GPT2Config
from datasets import Dataset
from Declare4Py.Utils.Declare.Checkers import TemplateConstraintChecker
from Declare4Py.Utils.Declare.TraceStates import TraceState
from Declare4Py.D4PyEventLog import D4PyEventLog
from pm4py.objects.conversion.log import converter as log_converter
from scipy.optimize import linear_sum_assignment
from enum import Enum
from peft import PeftModel, PeftConfig


def satisfies(sequence, constraint, detailed=False, completed=True):
    case_concept_name = []
    concept_name = []
    timestamp = []
    for token in sequence.split(" "):
        if not token == "":
            case_concept_name.append(0)
            concept_name.append(token.replace("_", " "))
            timestamp.append('2024-07-17 09:00:00')

    log_data = {'case:concept:name': case_concept_name, 'concept:name': concept_name, 'time:timestamp': timestamp}
    data_frame = pm4py.format_dataframe(pandas.DataFrame(log_data), case_id='case:concept:name', activity_key='concept:name')
    pm4py_event_log = log_converter.apply(data_frame)
    # Converting pm4py Event Log as Declare Event Log
    event_log = D4PyEventLog(case_name="case:concept:name")
    event_log.log = pm4py.convert_to_event_log(pm4py_event_log)
    event_log.log_length = len(event_log.log)
    event_log.timestamp_key = event_log.log._properties['pm4py:param:timestamp_key']
    event_log.activity_key = event_log.log._properties['pm4py:param:activity_key']
    # Building constraint for checker
    consider_vacuity = False
    rules = {"vacuous_satisfaction": consider_vacuity, "activation": constraint['condition'][0]}
    if constraint['template'].supports_cardinality:
        rules["n"] = constraint['n']

    if constraint['template'].is_binary:
        rules["correlation"] = constraint['condition'][1]

    rules["time"] = constraint['condition'][-1]
    # Checking constraint
    complete_result = (TemplateConstraintChecker(event_log.get_log()[0], completed, constraint['activities'], rules).get_template(constraint['template'])()).state
    if detailed:
        return complete_result

    else:
        return complete_result == TraceState.SATISFIED


def check_constraints(sequence, constraints, detailed, completed):
    if detailed:
        result = []
        counter = 0

    clean_sequence = " ".join(_extract_activities(get_string_between(DELIM_SOS, DELIM_EOS, sequence)))
    if clean_sequence.replace(" ", "") == "":
        if detailed:
            return [TraceState.POSSIBLY_SATISFIED] * len(constraints), len(constraints)

        else:
            return True

    for constraint in constraints:
        if detailed:
            result_on_input = satisfies(clean_sequence, constraint, detailed=True, completed=completed)
            result.append(result_on_input)
            if result_on_input == TraceState.SATISFIED:
                counter += 1

        else:
            result_on_input = satisfies(clean_sequence, constraint)
            if not result_on_input:
                return False

    if detailed:
        return result, counter

    else:
        return True
