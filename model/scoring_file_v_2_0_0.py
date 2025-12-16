# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------
import json
import logging
import os
import pickle
import numpy as np
import pandas as pd
import joblib

import azureml.automl.core
from azureml.automl.core.shared import logging_utilities, log_server
from azureml.telemetry import INSTRUMENTATION_KEY

from inference_schema.schema_decorators import input_schema, output_schema
from inference_schema.parameter_types.numpy_parameter_type import NumpyParameterType
from inference_schema.parameter_types.pandas_parameter_type import PandasParameterType
from inference_schema.parameter_types.standard_py_parameter_type import StandardPythonParameterType


data_sample = PandasParameterType(pd.DataFrame({"total_items": pd.Series([0.0], dtype="float32"), "bonsai": pd.Series([0.0], dtype="float32"), "other": pd.Series([0.0], dtype="float32"), "plastic_pots_trays": pd.Series([0.0], dtype="float32"), "single_flower_vase": pd.Series([0.0], dtype="float32"), "decorative_sand": pd.Series([0.0], dtype="float32"), "saucers_mats": pd.Series([0.0], dtype="float32"), "books": pd.Series([0.0], dtype="float32"), "suiban": pd.Series([0.0], dtype="float32"), "bonsai_seeds": pd.Series([0.0], dtype="float32"), "for_bonsai_classes": pd.Series([0.0], dtype="float32"), "bonsai_soil": pd.Series([0.0], dtype="float32"), "bonsai_tools": pd.Series([0.0], dtype="float32"), "bonsai_pots": pd.Series([0.0], dtype="float32"), "bonsai_decorations": pd.Series([0.0], dtype="float32"), "lucky_bag": pd.Series([0.0], dtype="float32"), "moss": pd.Series([0.0], dtype="float32"), "moss_bonsai": pd.Series([0.0], dtype="float32"), "chemicals_fertilizer": pd.Series([0.0], dtype="float32"), "wire": pd.Series([0.0], dtype="float32"), "decorative_stones": pd.Series([0.0], dtype="float32"), "accessories": pd.Series([0.0], dtype="float32"), "max_item_long_cm": pd.Series([0.0], dtype="float32"), "sum_item_volume_cm3": pd.Series([0.0], dtype="float64")}))
input_sample = StandardPythonParameterType({'data': data_sample})
method_sample = StandardPythonParameterType("predict")
sample_global_params = StandardPythonParameterType({"method": method_sample})

result_sample = NumpyParameterType(np.array([0.0]))
output_sample = StandardPythonParameterType({'Results':result_sample})

try:
    log_server.enable_telemetry(INSTRUMENTATION_KEY)
    log_server.set_verbosity('INFO')
    logger = logging.getLogger('azureml.automl.core.scoring_script_v2')
except:
    pass


def get_model_root(model_root: str):
    root_contents = os.listdir(model_root)
    logger.info(f"List model root dir: {os.listdir(model_root)}")
    if len(root_contents) == 1:
        root_file_path = os.path.join(model_root, root_contents[0])
        return root_file_path if os.path.isdir(root_file_path) else model_root
    else:
        raise Exception("Unexpected. root must contain a model file or a mlflow model directory")


def init():
    global model
    # This name is model.id of model that we want to deploy deserialize the model file back
    # into a sklearn model
    model_root = get_model_root(os.getenv('AZUREML_MODEL_DIR'))
    model_path = os.path.join(model_root, 'model.pkl')
    path = os.path.normpath(model_path)
    path_split = path.split(os.sep)
    log_server.update_custom_dimensions({'model_name': path_split[-3], 'model_version': path_split[-2]})
    try:
        logger.info("Loading model from path.")
        model = joblib.load(model_path)
        logger.info("Loading successful.")
    except Exception as e:
        logging_utilities.log_traceback(e, logger)
        raise

@input_schema('GlobalParameters', sample_global_params, convert_to_provided_type=False)
@input_schema('Inputs', input_sample)
@output_schema(output_sample)
def run(Inputs, GlobalParameters={"method": "predict"}):
    data = Inputs['data']
    if GlobalParameters.get("method", None) == "predict_proba":
        result = model.predict_proba(data)
    elif GlobalParameters.get("method", None) == "predict":
        result = model.predict(data)
    else:
        raise Exception(f"Invalid predict method argument received. GlobalParameters: {GlobalParameters}")
    if isinstance(result, pd.DataFrame):
        result = result.values
    return {'Results':result.tolist()}
