#import pytest
#from pathlib import Path
# Import the functions from your main.py file
from main import (
    validate_pytorch_static_analysis,
    validate_onnx_static_analysis,
    validate_pickle_static_analysis
)

# --- PyTorch Tests ---

def test_pytorch_valid_zip_format(tmp_path):
    """Test modern PyTorch save format (Zip based, starts with PK)."""
    d = tmp_path / "models"
    d.mkdir()
    p = d / "model.pt"
    
    # Write 'PK' magic bytes followed by dummy data
    p.write_bytes(b'PK\x03\x04\x00\x00') 
    
    valid, msg = validate_pytorch_static_analysis(p)
    assert valid is True
    assert "Zip format" in msg

def test_pytorch_valid_legacy_pickle(tmp_path):
    """Test legacy PyTorch save format (starts with PROTO opcode 0x80)."""
    d = tmp_path / "models"
    d.mkdir()
    p = d / "legacy.pt"
    
    # Write 0x80 (PROTO) followed by dummy data
    p.write_bytes(b'\x80\x02\x8a\nl')
    
    valid, msg = validate_pytorch_static_analysis(p)
    assert valid is True
    assert "Legacy Pickle" in msg

def test_pytorch_invalid_header(tmp_path):
    """Test a file that is neither PK zip nor 0x80 pickle."""
    d = tmp_path / "models"
    d.mkdir()
    p = d / "bad.pt"
    
    # Write garbage header
    p.write_bytes(b'\x00\x01\x02\x03')
    
    valid, msg = validate_pytorch_static_analysis(p)
    assert valid is False
    assert "Invalid PyTorch header" in msg


# --- ONNX Tests ---

def test_onnx_valid_header(tmp_path):
    """Test ONNX file starting with 0x08 (ir_version)."""
    d = tmp_path / "models"
    d.mkdir()
    p = d / "model.onnx"
    
    # Write 0x08 followed by dummy data
    p.write_bytes(b'\x08\x01\x12\x04')
    
    valid, msg = validate_onnx_static_analysis(p)
    assert valid is True
    assert "valid ONNX" in msg

def test_onnx_invalid_header(tmp_path):
    """Test ONNX file with incorrect start byte."""
    d = tmp_path / "models"
    d.mkdir()
    p = d / "bad.onnx"
    
    # Write garbage start byte
    p.write_bytes(b'\x00\x00\x00\x00')
    
    valid, msg = validate_onnx_static_analysis(p)
    assert valid is False
    assert "does not appear to start with ONNX" in msg


# --- Pickle Tests ---

def test_pickle_valid_structure(tmp_path):
    """Test Pickle with valid Start (0x80) and End (.)."""
    d = tmp_path / "models"
    d.mkdir()
    p = d / "model.pkl"
    
    # Construct a byte stream: [Start 0x80] ... [Content] ... [End 0x2E]
    content = b'\x80' + b'some_data' + b'.'
    p.write_bytes(content)
    
    valid, msg = validate_pickle_static_analysis(p)
    assert valid is True
    assert "valid Pickle" in msg

def test_pickle_invalid_start(tmp_path):
    """Test Pickle that ends correctly but starts wrong."""
    d = tmp_path / "models"
    d.mkdir()
    p = d / "bad_start.pkl"
    
    # Starts with 'X' instead of 0x80, but ends with '.'
    content = b'X' + b'some_data' + b'.'
    p.write_bytes(content)
    
    valid, msg = validate_pickle_static_analysis(p)
    assert valid is False
    assert "must start with PROTO" in msg

def test_pickle_invalid_end(tmp_path):
    """Test Pickle that starts correctly but ends wrong."""
    d = tmp_path / "models"
    d.mkdir()
    p = d / "bad_end.pkl"
    
    # Starts with 0x80, but ends with 'X' instead of '.'
    content = b'\x80' + b'some_data' + b'X'
    p.write_bytes(content)
    
    valid, msg = validate_pickle_static_analysis(p)
    assert valid is False
    assert "must end with STOP" in msg