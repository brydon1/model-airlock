import pytest
from typer.testing import CliRunner
from main import app
import json

runner = CliRunner()

@pytest.fixture
def valid_config(tmp_path):
    """Creates a temporary valid JSON config file."""
    config_data = {
        "model_name": "test-model",
        "author_email": "tester@pathrobotics.com",
        "version": "1.0.0",
        "framework": "PyTorch",
        "input_tensors": [{"name": "in", "dims": [1, 3, 224, 224], "dtype": "float32"}],
        "output_tensors": [{"name": "out", "dims": [1, 10], "dtype": "float32"}]
    }
    file_path = tmp_path / "model_config.json"
    file_path.write_text(json.dumps(config_data))
    return file_path

@pytest.fixture
def valid_model_file(tmp_path):
    """Creates a temporary fake PyTorch file (just header bytes)."""
    p = tmp_path / "model.pt"
    # Write the PyTorch 'magic bytes' (PK..) to fool the validator
    with open(p, "wb") as f:
        f.write(b'\x50\x4b\x03\x04\x00\x00') 
    return p

def test_deploy_dry_run_success(valid_config, valid_model_file):
    """Test the full happy path with dry-run."""
    result = runner.invoke(app, [
        "deploy",
        "--config", str(valid_config),
        "--model-file", str(valid_model_file),
        "--bucket", "test-bucket",
        "--dry-run"
    ])
    assert result.exit_code == 0
    assert "Schema Validation Passed" in result.stdout
    assert "Dry Run: Skipping S3 Upload" in result.stdout


def test_deploy_magic_byte_fail(valid_config, tmp_path):
    """Ensure we catch fake files."""
    fake_model = tmp_path / "fake.pt"
    fake_model.write_text("This is just text, not a model")
    
    result = runner.invoke(app, [
        "deploy",
        "--config", str(valid_config),
        "--model-file", str(fake_model),
        "--bucket", "test-bucket",
        "--dry-run"
    ])
    assert result.exit_code == 1
    assert "File Signature Mismatch" in result.stdout