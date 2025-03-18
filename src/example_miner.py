import pm4py
import pandas

from pm4py.objects.conversion.log import converter as log_converter
from Declare4Py.D4PyEventLog import D4PyEventLog
from Declare4Py.ProcessModels.DeclareModel import DeclareModel
from src.support.declare.DeclareMiner import DeclareMiner


decl_min_support = 0.8
decl_max_support = 0.9
decl_itemsets_support = 0.9
decl_max_declare_cardinality = 3

data_frame = pandas.read_csv(dataset_file_name, sep=",")
data_frame = pm4py.format_dataframe(data_frame, case_id="Case ID", activity_key="Activity", timestamp_key="Start Timestamp")    # Your log

event_log = D4PyEventLog(case_name="case:concept:name")
event_log.log = pm4py.convert_to_event_log(log_converter.apply(data_frame))
event_log.log_length = len(event_log.log)
event_log.timestamp_key = event_log.log._properties["pm4py:param:timestamp_key"]
event_log.activity_key = event_log.log._properties["pm4py:param:activity_key"]
discovery = DeclareMiner(log=event_log, consider_vacuity=True, min_support=decl_min_support, max_support=decl_max_support, itemsets_support=decl_itemsets_support, max_declare_cardinality=decl_max_declare_cardinality)
discovered_model: DeclareModel = discovery.run()
discovered_model.to_file(decl_path)
