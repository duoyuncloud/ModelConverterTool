import os
import glob
import onnx

# Find all model.onnx files in outputs/*/model.onnx
onnx_files = glob.glob('outputs/*/model.onnx')
if not onnx_files:
    print('No ONNX output files found in outputs/.')
    exit(1)

# Get the most recently modified ONNX file
onnx_files.sort(key=os.path.getmtime, reverse=True)
latest_onnx = onnx_files[0]
print(f'Validating latest ONNX file: {latest_onnx}')

# Load and check the ONNX model
try:
    model = onnx.load(latest_onnx)
    onnx.checker.check_model(model)
    print('ONNX model is valid!')
    print('\nModel graph summary:')
    print(onnx.helper.printable_graph(model.graph))
except Exception as e:
    print(f'ONNX model validation failed: {e}')
    exit(1) 