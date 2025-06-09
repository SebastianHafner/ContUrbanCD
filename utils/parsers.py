import argparse


def training_argument_parser():
    # https://docs.python.org/3/library/argparse.html#the-add-argument-methodsq
    parser = argparse.ArgumentParser(description="Experiment Args")
    parser.add_argument('-c', "--config-file", dest='config_file', required=True, help="path to config file")
    parser.add_argument('-o', "--output-dir", dest='output_dir', required=True, help="path to output directory")
    parser.add_argument('-d', "--dataset-dir", dest='dataset_dir', default="", required=True,
                        help="path to output directory")
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    return parser


def inference_argument_parser():
    # https://docs.python.org/3/library/argparse.html#the-add-argument-method
    parser = argparse.ArgumentParser(description="Experiment Args")
    parser.add_argument('-c', "--config-file", dest='config_file', required=True, help="path to config file")
    parser.add_argument('-e', "--edge-type", dest='edge_type', default='dense', help="mrf edge type")
    parser.add_argument('-o', "--output-dir", dest='output_dir', required=True, help="path to output directory")
    parser.add_argument('-d', "--dataset-dir", dest='dataset_dir', default="", required=True,
                        help="path to output directory")

    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    return parser


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
