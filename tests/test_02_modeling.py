
import json
import pytest
import re
import sys
import sys
import os
import ast
import pandas as pd
import numpy as np
from unittest.mock import MagicMock, patch, ANY

NOTEBOOK_PATH = "02_Modeling.ipynb"

@pytest.fixture
def notebook():
    with open(NOTEBOOK_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

# --- helper to extract code ---
def get_function_def(notebook, function_name):
    """Finds the cell defining the function and extracts ONLY the function definition using AST."""
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            source = "".join(cell["source"])
            if f"def {function_name}" in source:
                # Parse the cell source
                try:
                    tree = ast.parse(source)
                    for node in tree.body:
                        if isinstance(node, ast.FunctionDef) and node.name == function_name:
                            # Re-construct the function code
                            # This loses comments but keeps logic
                            return ast.unparse(node)
                except:
                    continue
    return None

def get_safety_check_code(notebook, check_type="training"):
    """Finds the cell with the safety check."""
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            source = "".join(cell["source"])
            if "COST SAFETY CHECK" in source:
                if check_type == "training" and "xgb.fit" in source:
                    return source
                if check_type == "transform" and "transformer.transform" in source:
                    return source
    return None

# --- Existing Structural Tests ---

def test_notebook_structure_valid_json(notebook):
    """Verify the notebook is valid JSON and has cells."""
    assert "cells" in notebook
    assert isinstance(notebook["cells"], list)
    assert len(notebook["cells"]) > 0


def test_setup_block_exists(notebook):
    """Verify that the new safe setup block exists and imports key modules."""
    found = False
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            source = "".join(cell["source"])
            if "import sagemaker" in source and "import awswrangler as wr" in source:
                found = True
                assert "read_parquet" in source, "Setup block should load data from S3 using read_parquet"
                break
    assert found, "Could not find the new safe setup block with S3 loading"


def test_training_logs_suppressed(notebook):
    """Verify xgb.fit is called with logs=False."""
    found = False
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            source = "".join(cell["source"])
            if "xgb.fit(" in source:
                found = True
                clean_source = re.sub(r'\s+', '', source)
                assert "logs=False" in clean_source, "xgb.fit must be called with logs=False"
    assert found, "Could not find xgb.fit call"

# --- NEW: Function Logic Tests ---

def test_make_model_frame_logic(notebook):
    """Test the data cleaning function make_model_frame."""
    # Use AST extraction to get ONLY the function, not the cell's execution code
    code = get_function_def(notebook, "make_model_frame")
    assert code, "Could not find make_model_frame function definition"
    
    # We need to setup the global variables it relies on: num_features, cat_features, label_col
    exec_globals = {
        "pd": pd,
        "num_features": ["num1", "num2"],
        "cat_features": ["cat1"],
        "label_col": "target"
    }
    
    # Execute the definition
    exec(code, exec_globals)
    make_model_frame = exec_globals["make_model_frame"]
    
    # Create sample dirty data
    df = pd.DataFrame({
        "num1": [1.0, "2.0", None], # Mixed types and missing
        "num2": [10, 20, 30],
        "cat1": ["A", None, "C"],   # Missing category
        "target": ["0", "1", "0"],  # String target
        "extra": [1, 2, 3]          # Extra column should be dropped
    })
    
    # Run function
    cleaned = make_model_frame(df)
    
    # Assertions
    assert "extra" not in cleaned.columns
    assert cleaned["num1"].isnull().sum() == 0 # Should fill NA
    assert cleaned["cat1"].iloc[1] == "UNK"    # Should fill NA with UNK
    assert pd.api.types.is_numeric_dtype(cleaned["num1"])
    assert pd.api.types.is_integer_dtype(cleaned["target"])
    assert cleaned.shape == (3, 4) # 2 num + 1 cat + 1 target


def test_transformer_recreation(notebook):
    """Verify that the transformer is recreated after the training/safety check block."""
    found = False
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            source = "".join(cell["source"])
            if "check_s3_prefix_has_contents" in source and "xgb.fit" in source:
                if "# --- FIX: Re-instantiate Transformer ---" in source:
                    found = True
                    break
    assert found, "Training safety check cell must recreate the transformer object"


def test_transform_output_consistency(notebook):
    """Verify that the retrieval cell uses transform_output to find files, not a hardcoded 'batch-output' path."""
    found = False
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            source = "".join(cell["source"])
            if "s3.list_objects_v2" in source and "transform_output" in source:
                # Check for the fix pattern
                if "urlparse(transform_output)" in source or "parsed_out.path" in source:
                    found = True
                    # Ensure the bad pattern is gone
                    if 'out_prefix = f"{prefix}batch-output/"' in source:
                        assert False, "Both old (bad) and new (good) logic found. Remove hardcoded batch-output."
                    break
    assert found, "Retrieval cell should determine S3 prefix from transform_output variable"

def test_classification_metrics_logic(notebook):
    """Test the metrics calculation function."""
    code = get_function_def(notebook, "classification_metrics")
    assert code, "Could not find classification_metrics function definition"
    
    # Needs sklearn functions imported
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
    exec_globals = {
        "accuracy_score": accuracy_score,
        "precision_score": precision_score,
        "recall_score": recall_score,
        "f1_score": f1_score,
        "roc_auc_score": roc_auc_score
    }
    
    exec(code, exec_globals)
    classification_metrics = exec_globals["classification_metrics"]
    
    y_true = [0, 1, 1, 0]
    y_pred = [0, 1, 0, 0]
    y_score = [0.1, 0.9, 0.4, 0.2]
    
    metrics = classification_metrics(y_true, y_pred, y_score)
    
    assert metrics["accuracy"] == 0.75
    assert "roc_auc" in metrics
    assert metrics["precision"] == 1.0 # 1 TP, 0 FP
    assert metrics["recall"] == 0.5    # 1 TP, 1 FN

# --- NEW: Transform Safety Test ---

def test_transform_safety_check_logic(notebook):
    """Test the batch transform safety check logic."""
    code = get_safety_check_code(notebook, "transform")
    assert code, "Could not find Transform Safety Check cell"
    
    # Mock modules
    with patch.dict(sys.modules, {"boto3": MagicMock(), "sagemaker": MagicMock(), "awswrangler": MagicMock()}):
        mock_boto_mod = sys.modules["boto3"]
        sys.modules["awswrangler"].s3.to_csv = MagicMock()
        
        # Scenario: Output Exists (Skip)
        mock_s3 = MagicMock()
        mock_boto_mod.client.return_value = mock_s3
        mock_s3.list_objects_v2.return_value = {"KeyCount": 1} # Exists
        
        mock_transformer = MagicMock()
        mock_test_xgb = MagicMock() # Mock the dataframe
        
        exec_globals = {
            "transform_output": "s3://bucket/path/output",
            "prefix": "prefix/",
            "bucket": "bucket",
            "test_xgb": mock_test_xgb,
            "transformer": mock_transformer,
            "wr": sys.modules["awswrangler"], # code uses 'wr' alias
            "urlparse": sys.modules["urllib.parse"].urlparse if "urllib.parse" in sys.modules else __import__("urllib.parse").parse.urlparse,
            "print": MagicMock()
        }
        
        # We need to inject the helper function because we are running cells in isolation
        exec_globals["check_s3_prefix_has_contents"] = lambda b, p: True
        
        exec(code, exec_globals)
        
        # Assertions
        exec_globals["print"].assert_any_call("Found existing transform output in s3://bucket/path/output. Skipping Transform.")
        mock_transformer.transform.assert_not_called()
        
        # Scenario: Output Missing (Run)
        exec_globals["check_s3_prefix_has_contents"] = lambda b, p: False
        exec(code, exec_globals)
        
        mock_transformer.transform.assert_called()

# --- Existing Training Logic Test (Refined) ---

def test_training_safety_check_logic(notebook):
    """Functionally test the safety check logic."""
    code = get_safety_check_code(notebook, "training")
    assert code, "Could not find Training Cost Safety Check cell"
    assert "xgb_model.create()" in code, "Must call xgb_model.create() to register model when skipping training"
    
    with patch.dict(sys.modules, {"boto3": MagicMock(), "sagemaker": MagicMock(), "sagemaker.model": MagicMock()}):
        mock_boto_mod = sys.modules["boto3"]
        mock_s3 = MagicMock()
        mock_boto_mod.client.return_value = mock_s3

        # Scenario: Skip
        mock_s3.list_objects_v2.return_value = {
            "KeyCount": 5, 
            "Contents": [{"Key": "bucket/output/model.tar.gz", "LastModified": "2023-01-01"}]
        }
        
        mock_estimator = MagicMock()
        exec_globals = {
            "output_path": "s3://fake-bucket/output/",
            "xgb_image": "fake-image",
            "role": "fake-role",
            "sm_sess": MagicMock(),
            "train_input": "fake-train",
            "val_input": "fake-val",
            "s3_train": "s3://bucket/train/data.csv",
            "s3_val": "s3://bucket/val/data.csv",
            "xgb": mock_estimator,
            "transform_output": "s3://bucket/output/transform", # Needed for transformer recreation
            "Model": MagicMock(),
            "TrainingInput": MagicMock(),
            "os": os,
            "urlparse": __import__("urllib.parse").parse.urlparse,
            "print": MagicMock()
        }
        sys.modules['sagemaker.model'].Model = MagicMock()
        
        exec(code, exec_globals)
        
        exec_globals["print"].assert_any_call("Found existing training artifacts in s3://fake-bucket/output/. Skipping Training to save cost.")
        mock_estimator.fit.assert_not_called()


def test_model_estimator_defined(notebook):
    """Verify that the Sagemaker Estimator (xgb) is defined with hyperparameters."""
    found_estimator = False
    found_hyperparameters = False
    found_output_path = False
    
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            source = "".join(cell["source"])
            
            # check for Estimator init
            if "sagemaker.estimator.Estimator" in source and "xgb =" in source:
                found_estimator = True
                
            # check if output_path is defined
            if "output_path =" in source:
                found_output_path = True

            # check for explicit set_hyperparameters call
            if "xgb.set_hyperparameters" in source:
                found_hyperparameters = True
                
    assert found_output_path, "output_path variable must be defined"
    assert found_estimator, "xgb Estimator must be defined"
    assert found_hyperparameters, "xgb hyperparameters must be set (e.g. via set_hyperparameters)"

def test_model_estimator_defined(notebook):
    """Verify that the Sagemaker Estimator (xgb) is defined with hyperparameters."""
    found_hyperparameters = False
    
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            source = "".join(cell["source"])
            
            # check for Estimator init
            if "sagemaker.estimator.Estimator" in source and "xgb =" in source:
                found_estimator = True
                if "hyperparameters={" in source.replace(" ", ""): # simple check if passed in constructor
                    found_hyperparameters = True
                
            # check if output_path is defined
            if "output_path =" in source:
                found_output_path = True

            # check for explicit set_hyperparameters call
            if "xgb.set_hyperparameters" in source:
                found_hyperparameters = True
                
    assert found_output_path, "output_path variable must be defined"
    assert found_estimator, "xgb Estimator must be defined"
    assert found_hyperparameters, "xgb hyperparameters must be set (e.g. via set_hyperparameters)"


