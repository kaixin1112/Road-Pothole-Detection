import torch
import sys

def convert_pt_to_onnx(weights_path, onnx_path, img_size=640, batch_size=1):
    # Load PyTorch model
    model = torch.load(weights_path, map_location='cpu')['model'].float()  # .pt -> model
    model.eval()

    # Dummy input for tracing
    dummy_input = torch.zeros((batch_size, 3, img_size, img_size))

    # Export to ONNX
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        verbose=False,
        opset_version=12,
        input_names=['images'],
        output_names=['output'],
        dynamic_axes={'images': {0: 'batch'}, 'output': {0: 'batch'}}
    )
    print(f"ONNX model saved to: {onnx_path}")

if __name__ == "__main__":

    pt_file = 'best.pt'
    onnx_file = 'best.onnx'

    convert_pt_to_onnx(pt_file, onnx_file)
