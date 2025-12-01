"""
Export the AlphaZeroNet to TorchScript for C++/LibTorch consumption.
Usage:
    PYTHONPATH=. python export_model.py
The output file alphazero_traced.pt will be written in the repository root.
"""

import torch

from src.models.alphazero import AlphaZeroNet


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AlphaZeroNet(input_planes=119, channels=256, blocks=40).to(device)
    model.eval()

    example_input = torch.randn(1, 119, 8, 8, device=device)
    print("Tracing model on device:", device)
    traced_script_module = torch.jit.trace(model, example_input)
    traced_script_module.save("alphazero_traced.pt")
    print("Exported alphazero_traced.pt")


if __name__ == "__main__":
    main()
