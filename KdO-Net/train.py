# Code:

from core import run_cnn


if __name__ == '__main__':

    run_cnn.to_network([
        "--run_mode=train",
        "--output_dim=32",
        "--batch_size=128"
    ])
