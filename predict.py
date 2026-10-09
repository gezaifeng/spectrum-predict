import argparse
from pathlib import Path

import numpy as np
import torch

from runtime import build_model, prediction, preprocess


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    config = checkpoint['config']
    net = build_model(config)
    net.load_state_dict(checkpoint['model'], strict=True)
    net.eval()
    x = preprocess(np.load(args.input, allow_pickle=False), config).unsqueeze(0)
    with torch.no_grad():
        pred, _ = prediction(net, x)
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, pred[0].numpy())


if __name__ == '__main__':
    main()
