import argparse


def device_type(v):
    if v.lower() == "cpu":
        return v.lower()
    return int(v)


def str2bool(v):
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Unsupported value encountered.")


class DictAction(argparse.Action):
    def __init__(self, option_strings, dest, nargs=None, **kwargs):
        if nargs is not None:
            raise ValueError("nargs not allowed")
        super(DictAction, self).__init__(option_strings, dest, **kwargs)

    def __call__(self, parser, namespace, values, option_string=None):
        input_dict = eval(f"dict({values})")
        output_dict = getattr(namespace, self.dest)
        for k in input_dict:
            output_dict[k] = input_dict[k]
        setattr(namespace, self.dest, output_dict)


def get_parser():
    parser = argparse.ArgumentParser(
        description="Spatial Temporal Graph Convolution Network for Skeleton-based Action Recognition"
    )

    parser.add_argument("--work-dir", "-w", default="./work_dir/temp", help="the work folder for storing results")
    parser.add_argument("--model-saved-name", type=str, default="runs", help="the name of saved model when training")
    parser.add_argument(
        "--config", "-c", default="./config/nturgbd-cross-view/test_bone.yaml", help="path to the configuration file"
    )

    parser.add_argument(
        "--phase",
        "-p",
        default="train",
        help="must be one of ['train', 'test', 'best', 'resume', 'model']"
        + "\n model: model analysis, also runs complexity analysis",
    )
    parser.add_argument(
        "--save-score", type=str2bool, default=False, help="if ture, the classification score will be stored"
    )

    parser.add_argument("--seed", type=int, default=1, help="random seed for pytorch")
    parser.add_argument("--log-interval", type=int, default=100, help="the interval for printing messages (#iteration)")
    parser.add_argument("--save-interval", type=int, default=1, help="the interval for storing models (#iteration)")
    parser.add_argument("--save-epoch", type=int, default=55, help="the start epoch to save model (#iteration)")
    parser.add_argument(
        "--eval-epoch", type=int, default=1,
        help="from this epoch onward, run validation every epoch regardless of --eval-interval"
    )
    parser.add_argument(
        "--eval-interval", type=int, default=10,
        help="run validation every N epochs before --eval-epoch is reached "
        "(milestone epochs in save-epoch/step and the final epoch are always evaluated)"
    )
    parser.add_argument(
        "--show-topk", type=int, default=[1, 2, 5], nargs="+", help="which Top K accuracy will be shown"
    )

    parser.add_argument("--feeder", default="feeders.feeder_ntu.Feeder", help="data loader will be used")
    parser.add_argument("--num-worker", type=int, default=4, help="the number of worker for data loader")
    parser.add_argument(
        "--train-feeder-args", action=DictAction, default=dict(), help="the arguments of data loader for training"
    )
    parser.add_argument(
        "--test-feeder-args", action=DictAction, default=dict(), help="the arguments of data loader for test"
    )
    parser.add_argument(
        "--val-feeder-args",
        action=DictAction,
        default=dict(),
        help="the arguments of data loader for per-epoch validation (single-clip, fast)",
    )

    parser.add_argument("--model", default=None, help="the model will be used")
    parser.add_argument("--model-args", action=DictAction, default=dict(), help="the arguments of model")
    parser.add_argument(
        "--weights", default=None, help="the weights for network initialization, pre-trained to transer learning"
    )
    parser.add_argument(
        "--ignore-weights",
        type=str,
        default=[],
        nargs="+",
        help="the name of weights which will be ignored in the initialization",
    )

    parser.add_argument("--loss", default="torch.nn.CrossEntropyLoss", help="the loss will be used")
    parser.add_argument(
        "--loss-args", action=DictAction, default={"label_smoothing": 0.1}, help="the arguments of loss"
    )

    parser.add_argument("--base-lr", type=float, default=0.01, help="initial learning rate")
    parser.add_argument(
        "--step", type=int, default=[20, 40, 60], nargs="+", help="the epoch where optimizer reduce the learning rate"
    )
    parser.add_argument(
        "--param-group-lr-mult", default=[],
        help="list of {patterns: [str,...], lr_mult: float}; any parameter whose "
             "name contains one of `patterns` is scheduled at base_lr * lr_mult "
             "instead of base_lr. Unmatched parameters keep lr_mult=1.0. Purely "
             "config-driven -- model-specific parameter names "
             "belong in the YAML config, not here."
    )
    parser.add_argument(
        "--param-group-wd-mult", default=[],
        help="list of {patterns: [str,...], wd_mult: float}; any parameter whose "
             "name contains one of `patterns` is optimized at weight_decay * "
             "wd_mult instead of weight_decay (e.g. wd_mult=0.0 excludes a "
             "parameter from weight decay entirely). Unmatched parameters keep "
             "wd_mult=1.0. Independent of --param-group-lr-mult (a parameter can "
             "match rules in both). Purely config-driven -- model-specific "
             "parameter names belong in the YAML config, not here."
    )
    parser.add_argument("--device", type=device_type, default=[0], nargs="+",
                        help="the indexes of GPUs for training or testing")
    parser.add_argument("--optimizer", default="SGD", help="type of optimizer")
    parser.add_argument("--nesterov", type=str2bool, default=True, help="use nesterov or not")
    parser.add_argument("--batch-size", "-b", type=int, default=64, help="training batch size")
    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=None,
        help="test batch size; defaults to --batch-size when not set",
    )
    parser.add_argument("--start-epoch", type=int, default=0, help="start training from which epoch")
    parser.add_argument("--num-epoch", type=int, default=80, help="stop training in which epoch")
    parser.add_argument("--weight-decay", type=float, default=0.0005, help="weight decay for optimizer")
    parser.add_argument("--lr-decay-rate", type=float, default=0.1, help="decay rate for learning rate")
    parser.add_argument("--warm-up-epoch", type=int, default=5, help="number of warm-up epochs")

    return parser
