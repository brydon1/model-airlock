from pydantic import BaseModel, Field, validator
from typing import List, Optional

class TensorConfig(BaseModel):
    name: str
    dims: List[int]
    dtype: str

    @validator('dims')
    def check_dimensions(cls, v):
        if any(d <= 0 for d in v):
            raise ValueError("Dimensions must be positive integers")
        return v

class ModelManifest(BaseModel):
    """
    Defines the contract for a valid research model before deployment.
    """
    model_name: str
    version: str = Field(..., pattern=r"^\d+\.\d+\.\d+$")  # Regex for SemVer
    framework: str
    input_tensors: List[TensorConfig]
    output_tensors: List[TensorConfig]
    
    # Metadata for the researcher
    author_email: str
    experiment_id: Optional[str] = None