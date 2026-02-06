
import unittest
from unittest.mock import MagicMock, patch
import sys
import os

# --- MOCK BOTO3 BEFORE IMPORT ---
# We mock boto3 here so the test can run in an environment where boto3 is not installed.
mock_boto3 = MagicMock()
sys.modules["boto3"] = mock_boto3

# Now we can safely import it (it will use the mock)
import boto3

# --- Logic to Test ---

def get_latest_model_artifact(s3_client, bucket, prefix):
    """
    Finds the latest model.tar.gz in the specified S3 prefix.
    """
    resp = s3_client.list_objects_v2(Bucket=bucket, Prefix=prefix)
    contents = sorted(resp.get('Contents', []), key=lambda x: x['LastModified'], reverse=True)
    for c in contents:
        if c['Key'].endswith('/output/model.tar.gz'):
            return f"s3://{bucket}/{c['Key']}"
    return None

def deploy_logic(sm_client, endpoint_name, create_fn):
    """
    Idempotent deployment logic: checks if endpoint exists, else creates it.
    create_fn is a callback to perform the actual creation if needed.
    """
    try:
        resp = sm_client.describe_endpoint(EndpointName=endpoint_name)
        return f"Endpoint {endpoint_name} already exists. Status: {resp['EndpointStatus']}"
    except sm_client.exceptions.ClientError:
        create_fn()
        return f"Creating new endpoint: {endpoint_name}..."

def monitor_schedule_logic(sm_client, schedule_name, create_fn):
    """
    Idempotent schedule logic.
    """
    try:
        sm_client.describe_monitoring_schedule(MonitoringScheduleName=schedule_name)
        return f"Schedule {schedule_name} already exists."
    except Exception as e: # Catching generic for mock simplified behavior
        create_fn()
        return f"Creating Monitoring Schedule: {schedule_name}"


class TestDeploymentNotebook(unittest.TestCase):
    
    def setUp(self):
        self.mock_s3 = MagicMock()
        self.mock_sm = MagicMock()
        self.bucket = "test-bucket"
        self.prefix = "test-prefix"

    def test_get_latest_model_artifact_found(self):
        # Setup mock response with unsorted dates
        self.mock_s3.list_objects_v2.return_value = {
            'Contents': [
                {'Key': 'test-prefix/old/output/model.tar.gz', 'LastModified': '2023-01-01'},
                {'Key': 'test-prefix/new/output/model.tar.gz', 'LastModified': '2023-01-02'},
                {'Key': 'test-prefix/garbage.txt', 'LastModified': '2023-01-03'}
            ]
        }
        
        result = get_latest_model_artifact(self.mock_s3, self.bucket, self.prefix)
        self.assertEqual(result, f"s3://{self.bucket}/test-prefix/new/output/model.tar.gz")

    def test_get_latest_model_artifact_none(self):
        self.mock_s3.list_objects_v2.return_value = {}
        result = get_latest_model_artifact(self.mock_s3, self.bucket, self.prefix)
        self.assertIsNone(result)

    def test_deploy_logic_exists(self):
        # Mock successful describe
        self.mock_sm.describe_endpoint.return_value = {'EndpointStatus': 'InService'}
        mock_create = MagicMock()
        
        msg = deploy_logic(self.mock_sm, "test-ep", mock_create)
        
        self.assertIn("already exists", msg)
        mock_create.assert_not_called()

    def test_deploy_logic_creates(self):
        # Mock ClientError (not found)
        # We simulate the exception structure
        error_response = {'Error': {'Code': 'ValidationException', 'Message': 'Could not find endpoint'}}
        # Define ClientError class on the exception object of the mock
        self.mock_sm.exceptions.ClientError = type('ClientError', (Exception,), {})
        self.mock_sm.describe_endpoint.side_effect = self.mock_sm.exceptions.ClientError
        
        mock_create = MagicMock()
        msg = deploy_logic(self.mock_sm, "test-ep", mock_create)
        
        self.assertIn("Creating new endpoint", msg)
        mock_create.assert_called_once()

    def test_monitor_schedule_logic_creates(self):
        # Simulating generic exception triggering creation (simulates ResourceNotFound)
        self.mock_sm.describe_monitoring_schedule.side_effect = Exception("Not Found")
        
        mock_create = MagicMock()
        msg = monitor_schedule_logic(self.mock_sm, "test-sched", mock_create)
        
        self.assertIn("Creating Monitoring Schedule", msg)
        mock_create.assert_called_once()

if __name__ == '__main__':
    unittest.main()
