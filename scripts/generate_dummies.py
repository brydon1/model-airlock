import torch
import onnx
from onnx import helper, TensorProto
import joblib

def create_pytorch_dummy():
    # Creates a tiny PyTorch model and saves it
    print("Generating model.pt...")
    model = torch.nn.Linear(10, 10)
    torch.save(model, "model.pt")

def create_pickle_dummy():
    # Creates a dummy dictionary and pickles it (Sklearn style)
    print("Generating model.pkl...")
    data = {"model": "dummy_sklearn_regressor", "coefficients": [0.1, 0.5]}
    joblib.dump(data, "model.pkl")

def create_onnx_dummy():
    # Creates a valid (but empty) ONNX graph
    print("Generating model.onnx...")
    
    # Define inputs/outputs for the graph
    input_tensor = helper.make_tensor_value_info('input', TensorProto.FLOAT, [1, 10])
    output_tensor = helper.make_tensor_value_info('output', TensorProto.FLOAT, [1, 10])
    
    # Create a node (Matrix multiplication)
    node_def = helper.make_node(
        'MatMul', 
        ['input', 'input'], 
        ['output'], 
    )

    # Create the graph and model
    graph_def = helper.make_graph(
        [node_def], 
        'test-model', 
        [input_tensor], 
        [output_tensor]
    )
    model_def = helper.make_model(graph_def, producer_name='onnx-example')
    
    onnx.save(model_def, "model.onnx")

if __name__ == "__main__":
    try:
        create_pytorch_dummy()
        create_pickle_dummy()
        create_onnx_dummy()
        print("\n✅ Success! Created model.pt, model.pkl, and model.onnx")
    except ImportError as e:
        print(f"\n❌ Missing library: {e}")
        print("Run: pip install torch onnx joblib")