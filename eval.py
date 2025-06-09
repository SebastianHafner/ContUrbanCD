import torch

from utils import datasets, parsers, experiment_manager, helpers, evaluation
from utils.experiment_manager import CfgNode
from model import model

from pathlib import Path

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def assessment(cfg: CfgNode, edge_type: str = 'dense', run_type: str = 'test'):
    print(cfg.NAME)
    net = model.load_model(cfg, device)
    m = evaluation.run_quantitative_evaluation(net, cfg, device, run_type, enable_mti=True, mti_edge_setting=edge_type)

    data = {}
    for attr in ['seg_cont', 'seg_fl', 'ch_cont', 'ch_fl']:
        f1 = evaluation.f1_score(getattr(m, f'TP_{attr}'), getattr(m, f'FP_{attr}'), getattr(m, f'FN_{attr}'))
        iou = evaluation.iou(getattr(m, f'TP_{attr}'), getattr(m, f'FP_{attr}'), getattr(m, f'FN_{attr}'))
        data[attr] = {'f1': f1, 'iou': iou}
    eval_folder = Path(cfg.PATHS.OUTPUT) / 'evaluation'
    eval_folder.mkdir(exist_ok=True)
    helpers.write_json(eval_folder / f'{cfg.NAME}_{edge_type}.json', data)


if __name__ == '__main__':
    args = parsers.inference_argument_parser().parse_known_args()[0]
    cfg = experiment_manager.setup_cfg(args)
    assessment(cfg, edge_type=args.edge_type)

