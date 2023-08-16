# utf-8
import json
import torch

if __name__ == '__main__':
    with open('./out/graphs.json', 'r') as fp:
        f = json.load(fp)
    torch.save(f, './out/graphs.pkl')

