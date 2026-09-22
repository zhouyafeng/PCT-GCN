import logging
import warnings
from typing import Optional, Tuple
import sys
from io import StringIO


import torch
import torch.nn as nn


_CALFLOPS_AVAILABLE = False
_CALFLOPS_ERROR = None

try:
    from calflops import calculate_flops

    _CALFLOPS_AVAILABLE = True
except ImportError as e:
    logging.warning(
        f"CalFlops not available: {e}, FLOPs calculation will be skipped."
        + "Install with: pip install calflops transformers if needed."
    )
    _CALFLOPS_ERROR = str(e)


class ModelComplexityAnalyzer:

    def __init__(self, model: nn.Module, input_shape: tuple, device: torch.device):
        self.model = model
        self.input_shape = input_shape
        self.device = device

    def analyze(self):
        self._log_params()
        self._log_flops_calflops()

    def _log_params(self):
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.model.parameters())
        logging.info(f"Parameters: {trainable / 1e6:.2f}M / {total / 1e6:.2f}M " f"(trainable/total)")

    def _log_flops_calflops(self) -> Optional[Tuple[str, str]]:
        if not _CALFLOPS_AVAILABLE:
            msg = "CalFlops not available. Install: pip install calflops transformers"
            if _CALFLOPS_ERROR:
                msg += f"\nOriginal error: {_CALFLOPS_ERROR}"
            logging.error(f"[complexity] {msg}")
            return None, None

        try:
            dummy_input = torch.zeros(self.input_shape, dtype=torch.float32, device=self.device)
            old_stdout = sys.stdout
            sys.stdout = captured_output = StringIO()

            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    flops_str, macs_str, params_str = calculate_flops(
                        self.model, args=[dummy_input], print_results=True, print_detailed=True, output_as_string=True
                    )
            finally:
                sys.stdout = old_stdout
                calflops_output = captured_output.getvalue()
            if calflops_output:
                logging.info(f"CalFlops Detailed Output:\n{calflops_output}")

            logging.info(
                f"FLOPs (CalFlops): {flops_str} | "
                f"MACs (CalFlops): {macs_str} | "
                f"Params (CalFlops): {params_str} | "
                f"input shape: {self.input_shape}"
            )

        except Exception as e:
            logging.error(f"[complexity] CalFlops calculation failed: {e}")
            return None, None
