import typer
import json
import boto3
from botocore.exceptions import NoCredentialsError
from rich.console import Console
from model_schema import ModelManifest
from pathlib import Path

app = typer.Typer()
console = Console()

def validate_math_constraints(manifest: ModelManifest):
    """
    Mock 'Sanity Check': Ensure input volume matches expected hardware constraints.
    """
    MAX_TENSOR_VOLUME = 10_000_000 # Example constraint
    
    for tensor in manifest.input_tensors:
        volume = 1
        for d in tensor.dims:
            volume *= d
        
        if volume > MAX_TENSOR_VOLUME:
            return False, f"Tensor {tensor.name} exceeds max volume ({volume} > {MAX_TENSOR_VOLUME})"
            
    return True, "Math validation passed."

def validate_model_static_analysis(model_file: Path) -> tuple[bool, str]:
    """
    Validate the model file using static analysis (headers/footers).
    """
    suffix = model_file.suffix.lower()

    if suffix == '.pt' or suffix == '.pth':
        return validate_pytorch_static_analysis(model_file)

    elif suffix == '.onnx':
        return validate_onnx_static_analysis(model_file)

    elif suffix == '.pkl':
        return validate_pickle_static_analysis(model_file)

    else:
        return False, f"Unsupported model file type: {suffix}"

def validate_pytorch_static_analysis(model_file: Path):
    """
    Validate the PyTorch model file using static analysis.
    """
    with open(model_file, 'rb') as f:
        header = f.read(2)
        # Case A: Modern PyTorch (Zip file) -> Starts with PK
        if header == b'PK':
            return True, "Valid PyTorch model (Zip format)"
        # Case B: Legacy PyTorch (Raw Pickle) -> Starts with PROTO (0x80)
        elif header[0] == 0x80:
            return True, "Valid PyTorch model (Legacy Pickle format)"
        else:
            return False, "Invalid PyTorch header (expected PK or 0x80)"

def validate_onnx_static_analysis(model_file: Path):
    """
    Validate the ONNX model file using static analysis.
    """
    with open(model_file, 'rb') as f:
        header = f.read(1)
        # ONNX usually starts with Field 1 (ir_version), which is 0x08 (Varint)
        # Note: This is a heuristic, not a guarantee, as protobuf is schema-based.
        if header != b'\x08':
                return False, "File does not appear to start with ONNX ir_version (0x08)"
    return True, "Model file appears to be a valid ONNX model"

def validate_pickle_static_analysis(model_file: Path):
    """
    Validate the Pickle model file using static analysis.
    """
    with open(model_file, 'rb') as f:
        # 1. Check Start: Should be PROTO opcode (0x80) for modern pickles
        start_byte = f.read(1)
        if start_byte != b'\x80':
            return False, "Pickle file must start with PROTO opcode (0x80)"
        
        # 2. Check End: Should be STOP opcode (.)
        # efficiently seek to the last byte without reading the whole file
        f.seek(-1, 2) 
        last_byte = f.read(1)
        if last_byte != b'.': # 0x2E
            return False, "Pickle file must end with STOP opcode (.)"
            
    return True, "Model file is a valid Pickle model"

@app.command()
def deploy(
    model_file: Path = typer.Option(..., help="Path to the actual model binary (.pt, .onnx, .pkl)"),
    config: Path = typer.Option(..., help="Path to the model_config.json"),
    bucket: str = typer.Option(..., help="Target AWS S3 Bucket"),
    dry_run: bool = typer.Option(False, help="Validate without uploading")
):
    """
    Validates a research model configuration and uploads it to S3 (Simulated).
    """
    console.print(f"[bold blue]🚀 Starting Model Airlock for {config}[/bold blue]")

    # 0. Check if Model File Exists
    if not model_file.exists():
        console.print(f"[bold red]❌ Model file not found:[/bold red] {model_file}")
        raise typer.Exit(code=1)

    # 1. Load and Parse JSON
    try:
        with open(config, 'r') as f:
            data = json.load(f)
    except Exception as e:
        console.print(f"[bold red]❌ Error reading file:[/bold red] {e}")
        raise typer.Exit(code=1)

    # 2. Schema Validation (Pydantic)
    try:
        manifest = ModelManifest(**data)
        console.print("[green]✅ Schema Validation Passed[/green]")
    except Exception as e:
        console.print(f"[bold red]❌ Schema Validation Failed:[/bold red] \n{e}")
        raise typer.Exit(code=1)

    # 3. Math/Logic Validation
    is_valid, message = validate_math_constraints(manifest)
    if not is_valid:
        console.print(f"[bold red]❌ Logic Check Failed:[/bold red] {message}")
        raise typer.Exit(code=1)
    else:
        console.print(f"[green]✅ {message}[/green]")

    # 4. Model Validation
    is_valid, message = validate_model_static_analysis(model_file)
    if not is_valid:
        console.print(f"[bold red]❌ Model Validation Failed:[/bold red] {message}")
        raise typer.Exit(code=1)
    else:
        console.print(f"[green]✅ {message}[/green]")

    # 5. Upload to AWS S3
    if dry_run:
        console.print("[yellow]⚠️ Dry Run: Skipping S3 Upload[/yellow]")
        return

    s3 = boto3.client('s3')
    try:
        # Define S3 paths (Key)
        # We use the version in the folder structure: s3://bucket/model_name/v1.0.0/model.pt
        base_path = f"{manifest.model_name}/{manifest.version}"
        
        # Upload Model
        console.print(f"uploading {model_file.name}...")
        s3.upload_file(str(model_file), bucket, f"{base_path}/{model_file.name}")
        
        # Upload Config
        console.print(f"uploading {config.name}...")
        s3.upload_file(str(config), bucket, f"{base_path}/{config.name}")

        console.print(f"[bold green]🎉 Deployment Complete: s3://{bucket}/{base_path}[/bold green]")
    except NoCredentialsError:
        console.print("[bold red]❌ AWS Credentials not found.[/bold red]")
    except Exception as e:
        console.print(f"[bold red]❌ Upload Failed:[/bold red] {e}")

if __name__ == "__main__":
    app()