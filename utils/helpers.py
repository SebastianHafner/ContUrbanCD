import torch
from torch import Tensor
from itertools import combinations
from pathlib import Path
from utils.experiment_manager import CfgNode
import json
from typing import Sequence, Tuple


def get_aoi_ids(cfg: CfgNode) -> Sequence[str]:
    aoi_ids = list(cfg.DATASET.TRAIN_IDS)
    return aoi_ids


def get_edges(n: int, edge_type: str) -> Sequence[Tuple[int, int]]:
    # edges are the timestamp combinations
    if edge_type == 'adjacent':
        edges = [(t1, t1 + 1) for t1 in range(n - 1)]
    elif edge_type == 'cyclic':
        edges = [(t1, t1 + 1) for t1 in range(n - 1)]
        edges.append((0, n - 1))
    elif edge_type == 'dense':
        edges = list(combinations(range(n), 2))
    elif edge_type == 'firstlast':
        edges = [(0, n - 1)]
    else:
        raise Exception('Unkown edge type!')
    return edges


def get_ch(seg: Tensor, edges: Sequence[Tuple[int, int]]) -> Tensor:
    ch = [torch.ne(seg[:, t1], seg[:, t2]) for t1, t2 in edges]
    ch = torch.stack(ch).transpose(0, 1)
    return ch


def load_json(file: Path) -> dict:
    with open(str(file)) as f:
        d = json.load(f)
    return d


def write_json(file: Path, data: dict) -> None:
    with open(str(file), 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
