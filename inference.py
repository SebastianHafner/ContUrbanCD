import torch
from utils import parsers, experiment_manager, helpers, datasets
from utils.experiment_manager import CfgNode
from pathlib import Path
import numpy as np
from model import model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPS = 1e-6


def inference(cfg: CfgNode, edge_type: str = 'dense', run_type: str = 'test'):
    print(cfg.NAME)
    net = model.load_model(cfg, device)
    net.eval()

    tile_size = cfg.AUGMENTATION.CROP_SIZE
    edges = helpers.get_edges(cfg.DATALOADER.TIMESERIES_LENGTH, edge_type)

    pred_folder = Path(cfg.PATHS.OUTPUT) / 'inference' / cfg.NAME
    pred_folder.mkdir(exist_ok=True)

    for aoi_id in list(cfg.DATASET.TRAIN_IDS):
        print(aoi_id)

        ds = datasets.create_eval_dataset(cfg, run_type, site=aoi_id, tiling=tile_size)
        o_seg = np.empty((1, ds.T, 1, ds.m, ds.n), dtype=np.uint8)

        for index in range(len(ds)):
            item = ds.__getitem__(index)
            x, y_seg = item['x'].to(device).unsqueeze(0), item['y'].to(device).unsqueeze(0)
            i, j = item['i'], item['j']
            o_seg_tile = net.module.inference(x, edges)
            o_seg[:, :, :, i:i + tile_size, j:j + tile_size] = o_seg_tile

        np.save(pred_folder / f'{cfg.NAME}_{aoi_id}.npy', o_seg)


if __name__ == '__main__':
    args = parsers.inference_argument_parser().parse_known_args()[0]
    cfg = experiment_manager.setup_cfg(args)
    inference(cfg, edge_type=args.edge_type)
