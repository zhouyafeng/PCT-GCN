
import os

import contextlib
import json
import logging
import os.path
import pickle
import shutil
import subprocess
import sys
import time
from collections import OrderedDict

import yaml
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
from torch.utils.tensorboard import SummaryWriter

import resource


from utils.common import init_seed, import_class  # noqa
from utils.parser import get_parser  # noqa
from utils.logger import log_file  # noqa
from utils.model_complexity import ModelComplexityAnalyzer  # noqa


os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")


try:
    _rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (min(2048, _rlimit[1]), _rlimit[1]))
except ValueError:
    pass


class JsonLogger:

    def __init__(self, log_path: str):
        os.makedirs(os.path.dirname(log_path) or "./logs", exist_ok=True)
        self._fp = open(log_path, "a", buffering=1)

    def log(self, data: dict):
        self._fp.write(json.dumps(data, ensure_ascii=False) + "\n")

    def close(self):
        self._fp.close()


def resolve_device(device_arg) -> torch.device:
    device_id = device_arg[0] if isinstance(device_arg, (list, tuple)) else device_arg
    if isinstance(device_id, str) and device_id.lower() == "cpu":
        return torch.device("cpu")
    if not torch.cuda.is_available():
        logging.warning("CUDA is not available, falling back to CPU.")
        return torch.device("cpu")
    return torch.device("cuda", int(device_id))


def gpu_mem_mb(device) -> int:
    if torch.cuda.is_available() and device.type == "cuda":
        return torch.cuda.memory_allocated(device) // (1024 * 1024)
    return 0


def _atomic_copy(src: str, dst: str) -> None:
    tmp = dst + ".tmp"
    shutil.copyfile(src, tmp)
    os.replace(tmp, dst)


def _load_ckpt(path: str, device: torch.device) -> dict:
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except (EOFError, RuntimeError, pickle.UnpicklingError) as e:
        raise RuntimeError(
            f"Checkpoint file is corrupted (likely truncated by an interrupted save): {path}\n"
            f"  Underlying error: {e}"
        ) from e


class Processor:

    def __init__(self, arg):
        self.arg = arg
        self.global_step = 0
        self.lr = arg.base_lr
        self.best_acc = 0
        self.best_acc_epoch = 0
        self.output_device = resolve_device(arg.device)

        self._last_ckpt = os.path.join(arg.work_dir, f"{arg.model_saved_name}_last.pt")
        self._best_ckpt = os.path.join(arg.work_dir, f"{arg.model_saved_name}_best.pt")
        save_epoch = [arg.save_epoch] if isinstance(arg.save_epoch, int) else list(arg.save_epoch)
        if arg.step:
            save_epoch = sorted(set(save_epoch) | set(arg.step))
        self.save_epoch = save_epoch

        self.save_arg()
        self.load_model()
        self.model = self.model.to(self.output_device)
        if self.output_device.type == "cuda" and isinstance(arg.device, list) and len(arg.device) > 1:
            self.model = nn.DataParallel(self.model, device_ids=arg.device, output_device=self.output_device)

        if arg.phase != "model":
            self._setup_writers()
            if arg.phase not in ["test", "best"]:
                self.load_optimizer()
            self.load_data()

        if arg.phase == "resume":
            self._resume_from_checkpoint()

    def start(self):
        self._run_complexity_analysis()

        if self.arg.phase in ["train", "resume"]:
            self.__start_time = time.time()
            if self.global_step == 0 and self.arg.start_epoch > 0:
                self.global_step = self.arg.start_epoch * len(self.data_loader["train"])

            training_skipped = self.arg.start_epoch >= self.arg.num_epoch
            if training_skipped:
                logging.info(
                    f"All {self.arg.num_epoch} epochs already complete "
                    f"(resumed from epoch {self.arg.start_epoch}). Skipping to final test."
                )

            try:
                for epoch in range(self.arg.start_epoch, self.arg.num_epoch):
                    self.train(epoch)
                    self._save_checkpoint(epoch)
                    is_last_epoch = epoch + 1 == self.arg.num_epoch
                    is_milestone = epoch + 1 in self.save_epoch
                    do_eval = (
                        is_last_epoch
                        or is_milestone
                        or epoch + 1 >= self.arg.eval_epoch
                        or (epoch + 1) % self.arg.eval_interval == 0
                    )
                    if do_eval:
                        self.eval(epoch, loader_name=["val"])
                        if self.best_acc_epoch == epoch + 1:
                            _atomic_copy(self._last_ckpt, self._best_ckpt)
                    if is_milestone:
                        snapshot = os.path.join(self.arg.work_dir, f"{self.arg.model_saved_name}_epoch{epoch + 1}.pt")
                        _atomic_copy(self._last_ckpt, snapshot)
            except KeyboardInterrupt:
                logging.info("Interrupted. Checkpoint already saved for the last completed epoch.")
                return

            self._test_best()
            self._log_train_summary(training_skipped)

        elif self.arg.phase == "test":
            if self.arg.weights is None:
                raise ValueError("Please appoint --weights.")
            logging.info(f"Model:   {self.arg.model}.")
            logging.info(f"Weights: {self.arg.weights}.")
            self.eval(epoch=0, loader_name=["test"], save_score_stem=self.arg.weights.replace(".pt", ""))

        elif self.arg.phase == "best":
            self.__start_time = time.time()
            self._test_best()
            self._log_train_summary(final_test_only=True)

        logging.info("Done.\n")
        if hasattr(self, "json_logger"):
            self.json_logger.close()

    def train(self, epoch: int):
        self.model.train()
        logging.info(f"Training epoch: {epoch + 1}")
        loader = self.data_loader["train"]
        self.lr = self.adjust_learning_rate(epoch)

        loss_values, acc_values = [], []
        top2_values, top5_values = [], []
        data_times, model_times = [], []
        timer = dict(dataloader=0.001, model=0.001, statistics=0.001)
        component_values, component_weights = {}, {}

        self.train_writer.add_scalar("epoch", epoch, self.global_step)
        self.record_time()

        num_classes = self.arg.model_args.get("num_class", 60)

        for _, (data, label, _) in enumerate(tqdm(loader, ncols=40)):
            self.global_step += 1
            data = data.float().to(self.output_device)
            label = label.long().to(self.output_device)
            data_t = self.split_time()
            timer["dataloader"] += data_t
            data_times.append(data_t)

            output = self.model(data)
            loss = self.loss(output, label)
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
            self.optimizer.step()

            logits = output[0] if isinstance(output, tuple) else output
            loss_item = loss.item()
            loss_values.append(loss_item)

            for name, (raw_value, weight) in getattr(self.loss, "last_components", {}).items():
                component_values.setdefault(name, []).append(raw_value)
                component_weights[name] = weight
            model_t = self.split_time()
            timer["model"] += model_t
            model_times.append(model_t)

            with torch.no_grad():
                max_k = min(5, num_classes)
                _, pred = logits.topk(max_k, dim=1, largest=True, sorted=True)
                correct = pred.eq(label.view(-1, 1).expand_as(pred))
                top1 = correct[:, :1].any(1).float().mean().item()
                top2 = correct[:, :min(2, num_classes)].any(1).float().mean().item()
                top5 = correct[:, :max_k].any(1).float().mean().item()

            acc_values.append(top1)
            top2_values.append(top2)
            top5_values.append(top5)

            self.train_writer.add_scalar("acc", top1, self.global_step)
            self.train_writer.add_scalar("loss", loss_item, self.global_step)
            self.train_writer.add_scalar("lr", self.lr, self.global_step)
            timer["statistics"] += self.split_time()

        proportion = {k: f"{int(round(v * 100 / sum(timer.values()))):02d}%" for k, v in timer.items()}
        logging.info(
            f"\tMean training loss: {np.mean(loss_values):.4f}.  "
            f"Mean training acc: {np.mean(acc_values)*100:.2f}%."
            f'\tTime consumption: [Data]{proportion["dataloader"]}, [Network]{proportion["model"]}'
        )
        loss_breakdown = {
            name: round(float(np.mean(values)), 5) for name, values in component_values.items()
        }
        if len(loss_breakdown) > 1:
            breakdown_str = "  ".join(
                f"{name}={loss_breakdown[name]:.4f}(x{component_weights[name]:g})"
                for name in loss_breakdown
            )
            logging.info(f"\tLoss breakdown: {breakdown_str}")
        self.json_logger.log(
            {
                "mode": "train",
                "epoch": epoch + 1,
                "lr": round(self.lr, 6),
                "memory": gpu_mem_mb(self.output_device),
                "data_time": round(float(np.mean(data_times)), 5),
                "top1_acc": round(float(np.mean(acc_values)), 5),
                "top2_acc": round(float(np.mean(top2_values)), 5),
                "top5_acc": round(float(np.mean(top5_values)), 5),
                "loss": round(float(np.mean(loss_values)), 5),
                "loss_components": loss_breakdown,
                "loss_weights": component_weights,
                "time": round(float(np.mean(model_times)), 5),
            }
        )

    def eval(self, epoch: int, loader_name=("test",), save_score_stem=None):
        self.model.eval()
        logging.info(f"Eval epoch: {epoch + 1}")

        with contextlib.ExitStack() as stack:
            f_w = f_r = None
            if save_score_stem is not None:
                f_w = stack.enter_context(open(f"{save_score_stem}_wrong.txt", "w"))
                f_r = stack.enter_context(open(f"{save_score_stem}_right.txt", "w"))

            for ln in loader_name:
                loss_values = []
                score_frags = []
                all_preds, all_labels = [], []
                num_clips = getattr(self.data_loader[ln].dataset, "num_clips", 1)

                for _, (data, label, index) in enumerate(tqdm(self.data_loader[ln], ncols=40)):
                    with torch.no_grad():
                        data = data.float().to(self.output_device)
                        label = label.long().to(self.output_device)
                        logits, loss = self._eval_step(data, label, num_clips)

                    score_frags.append(logits.cpu())
                    loss_values.append(loss.item())
                    _, pred = torch.max(logits, 1)
                    all_preds.append(pred.cpu())
                    all_labels.append(label.cpu())

                    if f_w is not None:
                        predict = pred.cpu().numpy().tolist()
                        true = label.cpu().numpy().tolist()
                        for i, x in enumerate(predict):
                            f_r.write(f"{x},{true[i]}\n")
                            if x != true[i]:
                                f_w.write(f"{index[i]},{x},{true[i]}\n")

                score = torch.softmax(torch.cat(score_frags, dim=0), dim=1).numpy()
                loss = float(np.mean(loss_values))

                dataset = self.data_loader[ln].dataset
                raw_sample_name = getattr(dataset, "sample_name", None)
                sample_name = (
                    raw_sample_name
                    if raw_sample_name is not None and len(raw_sample_name) == len(score)
                    else np.arange(len(score))
                )

                topk_accs = {k: dataset.top_k(score, k) for k in self.arg.show_topk}
                top1_acc = topk_accs[1] if 1 in topk_accs else dataset.top_k(score, 1)

                all_preds_np = torch.cat(all_preds).numpy()
                all_labels_np = torch.cat(all_labels).numpy()
                classes = np.unique(all_labels_np)
                per_class = [
                    np.mean(all_preds_np[all_labels_np == c] == c)
                    for c in classes
                    if np.any(all_labels_np == c)
                ]
                mean_cls_acc = float(np.mean(per_class)) if per_class else 0.0

                if self.arg.phase in ["train", "resume"]:
                    self.val_writer.add_scalar("loss", loss, self.global_step)
                    self.val_writer.add_scalar("acc", top1_acc, self.global_step)

                self.json_logger.log(
                    {
                        "mode": ln,
                        "loader_name": list(loader_name),
                        "epoch": epoch + 1,
                        "lr": round(self.lr, 6),
                        **{f"top{k}_acc": round(v, 5) for k, v in topk_accs.items()},
                        "mean_class_accuracy": round(mean_cls_acc, 5),
                        "loss": round(loss, 5),
                    }
                )

                score_dict = dict(zip(sample_name, score))
                topk_str = "  ".join(f"Top{k}: {100 * v:.2f}%" for k, v in topk_accs.items())
                logging.info(
                    f"\t[{ln}] loss: {loss:.4f}  {topk_str}"
                    f"  | best: {100 * self.best_acc:.2f}% @ ep{self.best_acc_epoch}"
                )

                if save_score_stem is not None:
                    with open(f"{save_score_stem}_score.pkl", "wb") as f:
                        pickle.dump(score_dict, f)

                if ln == "val" and top1_acc > self.best_acc:
                    self.best_acc = top1_acc
                    self.best_acc_epoch = epoch + 1

    def save_arg(self):
        os.makedirs(self.arg.work_dir, exist_ok=True)
        self.arg.output_device = str(self.output_device)
        with open(os.path.join(self.arg.work_dir, "config.yaml"), "w") as f:
            f.write(f"# command line: {' '.join(sys.argv)}\n\n")
            yaml.dump(vars(self.arg), f)

    def load_model(self):
        Model = import_class(self.arg.model)
        logging.info(Model)
        self.model = Model(**self.arg.model_args)
        logging.info(self.model)
        Loss = import_class(self.arg.loss)
        self.loss = Loss(**self.arg.loss_args).to(self.output_device)
        logging.info(f"Loss:\n{self.loss}")
        if self.arg.weights:
            self._load_weights(self.arg.weights)

    def load_data(self):
        Feeder = import_class(self.arg.feeder)
        pin = self.output_device.type == "cuda"
        nw = self.arg.num_worker
        pw = nw > 0
        mp_ctx = "fork" if nw > 0 else None

        self.data_loader = {}
        self.data_loader["val"] = torch.utils.data.DataLoader(
            dataset=Feeder(**self._build_val_feeder_args()),
            batch_size=self.arg.test_batch_size,
            shuffle=False,
            num_workers=nw,
            drop_last=False,
            pin_memory=pin,
            persistent_workers=pw,
            multiprocessing_context=mp_ctx,
        )
        self.data_loader["test"] = self.data_loader["val"]

        if self.arg.phase in ["train", "resume"]:
            self.data_loader["train"] = torch.utils.data.DataLoader(
                dataset=Feeder(**self.arg.train_feeder_args),
                batch_size=self.arg.batch_size,
                shuffle=True,
                num_workers=nw,
                drop_last=True,
                pin_memory=pin,
                persistent_workers=pw,
                multiprocessing_context=mp_ctx,
            )

    def _build_val_feeder_args(self) -> dict:
        val_feeder_args = self.arg.val_feeder_args
        if not val_feeder_args:
            val_feeder_args = dict(self.arg.test_feeder_args)
            val_feeder_args.pop("num_clips", None)
            val_feeder_args["frame_sample"] = "resize"
            val_feeder_args["p_interval"] = [0.95]
        return val_feeder_args

    @staticmethod
    def _match_mult(name, rules, mult_key):
        for rule in rules:
            if any(pattern in name for pattern in rule["patterns"]):
                if mult_key not in rule:
                    raise ValueError(
                        f"Rule {rule} matched parameter '{name}' but has no '{mult_key}' key "
                        "-- check for a lr_mult/wd_mult typo in this config's param group rules."
                    )
                return rule[mult_key]
        return 1.0

    def _build_param_groups(self):
        named_params = list(self.model.named_parameters()) + list(self.loss.named_parameters())
        lr_rules = self.arg.param_group_lr_mult or []
        wd_rules = self.arg.param_group_wd_mult or []

        buckets = {}
        for name, p in named_params:
            lr_mult = self._match_mult(name, lr_rules, "lr_mult")
            wd_mult = self._match_mult(name, wd_rules, "wd_mult")
            buckets.setdefault((lr_mult, wd_mult), []).append(p)

        return [
            {
                "params": params,
                "lr_mult": lr_mult,
                "weight_decay": self.arg.weight_decay * wd_mult,
            }
            for (lr_mult, wd_mult), params in buckets.items()
        ]

    def load_optimizer(self):
        param_groups = self._build_param_groups()
        if self.arg.optimizer == "SGD":
            self.optimizer = optim.SGD(
                param_groups,
                lr=self.arg.base_lr,
                momentum=0.9,
                nesterov=self.arg.nesterov,
            )
        elif self.arg.optimizer == "Adam":
            self.optimizer = optim.Adam(param_groups, lr=self.arg.base_lr)
        else:
            raise ValueError(f"Unknown optimizer: {self.arg.optimizer}")
        logging.info(f"Using warm-up, epoch: {self.arg.warm_up_epoch}")

    def adjust_learning_rate(self, epoch: int) -> float:
        if epoch < self.arg.warm_up_epoch:
            lr = self.arg.base_lr * (epoch + 1) / self.arg.warm_up_epoch
        else:
            lr = self.arg.base_lr * (self.arg.lr_decay_rate ** np.sum(epoch >= np.array(self.arg.step)))
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr * param_group.get("lr_mult", 1.0)
        return lr

    def record_time(self):
        self.cur_time = time.time()
        return self.cur_time

    def split_time(self):
        elapsed = time.time() - self.cur_time
        self.record_time()
        return elapsed

    def _setup_writers(self):
        arg = self.arg

        if arg.phase in ["train", "resume"]:
            summary_logdir = os.path.join(arg.work_dir, arg.model_saved_name)
            if os.path.isdir(summary_logdir) and arg.phase == "train":
                logging.warning(
                    f"TensorBoard log dir already exists: {summary_logdir}. "
                    "New events will be appended. Delete manually to start fresh."
                )
            self.train_writer = SummaryWriter(os.path.join(summary_logdir, "train"), "train")
            self.val_writer = SummaryWriter(os.path.join(summary_logdir, "val"), "val")

        json_log_path = os.path.join(arg.work_dir, f"{arg.log_file_name}.json")
        self.json_logger = JsonLogger(json_log_path)
        self.json_logger.log(
            {
                "config_name": os.path.basename(getattr(arg, "config", "") or ""),
                "work_dir": arg.work_dir,
                "phase": arg.phase,
                "seed": getattr(arg, "seed", None),
            }
        )

    def _load_weights(self, path: str):
        logging.info(f"Load weights from {path}.")

        if path.endswith(".pkl"):
            with open(path, "rb") as f:
                raw = pickle.load(f)
        else:
            raw = _load_ckpt(path, self.output_device)

        weights = raw["model"] if isinstance(raw, dict) and "model" in raw else raw
        weights = OrderedDict((k.split("module.")[-1], v.to(self.output_device)) for k, v in weights.items())

        for pattern in self.arg.ignore_weights:
            for key in list(weights.keys()):
                if pattern in key:
                    weights.pop(key)
                    logging.info(f"Removed weight: {key}.")

        try:
            self.model.load_state_dict(weights)
        except RuntimeError:
            state = self.model.state_dict()
            missing = set(state.keys()) - set(weights.keys())
            logging.error("Missing weights: " + ", ".join(sorted(missing)))
            state.update(weights)
            self.model.load_state_dict(state)

    def _save_checkpoint(self, epoch: int, path: str = None):
        state_dict = self.model.state_dict()
        weights = OrderedDict((k.split("module.")[-1], v.cpu()) for k, v in state_dict.items())
        target = path or self._last_ckpt
        tmp = target + ".tmp"
        torch.save(
            {
                "model": weights,
                "optimizer": self.optimizer.state_dict(),
                "epoch": epoch + 1,
                "best_acc": float(self.best_acc),
                "best_acc_epoch": int(self.best_acc_epoch),
                "global_step": int(self.global_step),
            },
            tmp,
        )
        os.replace(tmp, target)

    def _load_best_weights(self):
        ckpt = _load_ckpt(self._best_ckpt, self.output_device)
        weights = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
        if isinstance(self.arg.device, list) and len(self.arg.device) > 1:
            weights = OrderedDict(("module." + k, v.to(self.output_device)) for k, v in weights.items())
        self.model.load_state_dict(weights)

    def _resume_from_checkpoint(self):
        if not os.path.exists(self._last_ckpt):
            raise FileNotFoundError(
                f"Resume checkpoint not found: {self._last_ckpt}\n"
                "Make sure --work-dir points to the interrupted run's directory."
            )

        try:
            ckpt = _load_ckpt(self._last_ckpt, self.output_device)
        except RuntimeError as e:
            if os.path.exists(self._best_ckpt):
                logging.warning(f"{e}\nFalling back to best checkpoint: {self._best_ckpt}")
                ckpt = _load_ckpt(self._best_ckpt, self.output_device)
            else:
                raise

        if isinstance(ckpt, dict) and "model" in ckpt:
            weights = OrderedDict((k, v.to(self.output_device)) for k, v in ckpt["model"].items())
            self.model.load_state_dict(weights)
            if ckpt.get("optimizer") is not None:
                self.optimizer.load_state_dict(ckpt["optimizer"])
            else:
                logging.warning("Checkpoint has no optimizer state. Optimizer starts from scratch.")
            self.arg.start_epoch = int(ckpt["epoch"])
            self.best_acc = float(ckpt["best_acc"])
            self.best_acc_epoch = int(ckpt["best_acc_epoch"])
            self.global_step = int(ckpt["global_step"])
            logging.info(
                f"Resumed from epoch {self.arg.start_epoch} "
                f"(best_acc={self.best_acc:.4f} @ ep{self.best_acc_epoch})"
            )
        else:
            logging.warning("Legacy checkpoint: optimizer state not available. Set --start-epoch manually.")
            weights = OrderedDict((k.split("module.")[-1], v.to(self.output_device)) for k, v in ckpt.items())
            self.model.load_state_dict(weights)
            logging.info(f"Loaded model weights from {self._last_ckpt} (epoch={self.arg.start_epoch})")

    @staticmethod
    def _avg_over_clips(tensor: torch.Tensor, B: int, num_clips: int) -> torch.Tensor:
        return tensor.reshape(B, num_clips, -1).mean(dim=1)

    def _eval_step(self, data, label, num_clips: int):
        if num_clips > 1:
            B = data.size(0)
            output = self.model(data.flatten(0, 1))
            if isinstance(output, tuple):
                logits = self._avg_over_clips(output[0], B, num_clips)
                avg_rest = tuple(
                    self._avg_over_clips(o, B, num_clips) if isinstance(o, torch.Tensor) else o
                    for o in output[1:]
                )
                loss = self.loss((logits,) + avg_rest, label)
            else:
                logits = self._avg_over_clips(output, B, num_clips)
                loss = self.loss(logits, label)
        else:
            output = self.model(data)
            loss = self.loss(output, label)
            logits = output[0] if isinstance(output, tuple) else output
        return logits, loss

    def _test_best(self):
        if not os.path.exists(self._best_ckpt):
            raise FileNotFoundError(
                f"Best checkpoint not found: {self._best_ckpt}\n"
                "Make sure --work-dir points to a completed training run."
            )
        self._load_best_weights()
        self.eval(
            epoch=self.best_acc_epoch - 1,
            loader_name=["test"],
            save_score_stem=self._best_ckpt.replace(".pt", ""),
        )

    def _run_complexity_analysis(self):
        C = self.arg.model_args.get("in_channels", 3)
        T = (self.arg.train_feeder_args or {}).get("window_size", 64)
        V = self.arg.model_args.get("num_point", 25)
        M = self.arg.model_args.get("num_person", 2)
        m = self.model.module if hasattr(self.model, "module") else self.model
        ModelComplexityAnalyzer(m, (1, C, T, V, M), self.output_device).analyze()

    def _log_train_summary(self, final_test_only: bool = False):
        elapsed = time.time() - self.__start_time
        h, rem = divmod(int(elapsed), 3600)
        m, s = divmod(rem, 60)
        time_label = "Final-test time" if final_test_only else "Total training time"
        wd_line = f"Weight decay: {self.arg.weight_decay}"
        if self.arg.param_group_wd_mult:
            overrides = ", ".join(
                f"{rule['patterns']}->{self.arg.weight_decay * rule['wd_mult']}"
                for rule in self.arg.param_group_wd_mult
            )
            wd_line += f" (overrides: {overrides})"
        logging.info(
            f"\n\tBest accuracy: {self.best_acc}"
            f"\n\tEpoch number: {self.best_acc_epoch}"
            f"\n\tModel name: {self.arg.work_dir}"
            f"\n\t{wd_line}"
            f"\n\tBase LR: {self.arg.base_lr}"
            f"\n\tBatch Size: {self.arg.batch_size}"
            f"\n\tTest Batch Size: {self.arg.test_batch_size}"
            f"\n\tseed: {self.arg.seed}"
            f"\n\t{time_label}({elapsed:.0f}s): {h:02d}h {m:02d}m {s:02d}s"
        )


if __name__ == "__main__":
    parser = get_parser()

    p = parser.parse_args()
    if p.config is not None:
        with open(p.config, "r") as f:
            default_arg = yaml.safe_load(f)
        key = vars(p).keys()
        for k in default_arg.keys():
            if k not in key:
                print(f"WRONG ARG: {k}")
                assert k in key
        parser.set_defaults(**default_arg)
    arg = parser.parse_args()

    if arg.test_batch_size is None:
        arg.test_batch_size = 2 * arg.batch_size

    log_file_name = f"{time.strftime('%Y-%m-%d_%H-%M-%S')}_{arg.phase}.log"
    log_file(arg.work_dir, log_file_name, level=logging.INFO)
    arg.log_file_name = log_file_name

    try:
        commit_id = (
            subprocess.check_output(["git", "rev-parse", "--short=7", "HEAD"], stderr=subprocess.DEVNULL)
            .decode()
            .strip()
        )
    except Exception:
        commit_id = "unknown"

    logging.info(f"commit: {commit_id}")
    logging.info(f"Configuration Parameters:\n{vars(arg)}\n")

    init_seed(arg.seed)
    processor = Processor(arg)
    processor.start()
