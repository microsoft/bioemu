"""
tools for profiling PyTorch models, including timing and memory usage.
Original code from Hannes, there will be future ai4s-timing module.
It will be replaced by that module once it is ready.
"""
import logging
import time
import typing as ty
from contextlib import ExitStack
from functools import partial

import numpy as np
import torch

LOG = logging.getLogger(__name__)


class ProfilingDoneException(Exception):
    """Exception to signal that profiling is done."""

    pass


class CPUTimer:
    def __init__(self):
        self.start_time = time.perf_counter()
        self.end_time = None

    def stop(self):
        self.end_time = time.perf_counter()

    def elapsed(self) -> float:
        if self.end_time is None:
            raise ValueError("Timer has not been stopped")
        return self.end_time - self.start_time


class CudaTimer:
    """Works for a single GPU only."""

    def __init__(self) -> None:
        self.start = torch.cuda.Event(enable_timing=True)  # type: ignore[no-untyped-call]
        self.end = torch.cuda.Event(enable_timing=True)  # type: ignore[no-untyped-call]
        self.start.record()
        self._elapsed = None

    def stop(self) -> None:
        self.end.record()  # type: ignore[no-untyped-call]
        torch.cuda.synchronize()  # type: ignore[no-untyped-call]
        self._elapsed = self.start.elapsed_time(self.end) / 1000  # type: ignore[no-untyped-call]

    def elapsed(self) -> float:
        assert self._elapsed is not None, "Timer has not been stopped"
        return self._elapsed


class GenericTimer:
    def __init__(self):
        self._timer = CPUTimer() if not torch.cuda.is_available() else CudaTimer()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._timer.stop()
        if exc_type is not None:
            LOG.error(f"Exception in timer: {exc_type}, {exc_val}, {exc_tb}")
            return False
        return True

    def elapsed(self) -> float:
        return self._timer.elapsed()


class ModelProfiler:
    """
    Only profile the later steps include in the "profile_batch_idx" slice.

    Example:
    ```python
        with ModelProfiler(
            model,
            ...
        ) as prof:
            prof.set_batch(batch)
            prof.step()
    ```

    """

    def __init__(
        self,
        profile_memory: bool,
        trace: bool,
        device
    ):

        self.batch: ty.Any = None
        self.ground_truth = None
        self.batch_idx = 0
        # self.train = train
        self.stack = ExitStack()
        # self.profile_batch_idx = profile_batch_idx
        self._forward_dts: list[float] = []
        self._loss_dts: list[float] = []
        self._backward_dts: list[float] = []
        self._max_memory: list[int] = []
        self._profile_memory = profile_memory
        self._trace = trace
        self._prof = False
        self._device = device

        # if self.train:
        #     assert loss_function is not None, "Loss function must be provided for training"
        #     self.loss_function = loss_function
        #     self.optimizer = torch.optim.AdamW(
        #         model.parameters(),
        #         lr=3e-4,
        #         eps=1e-6,
        #         weight_decay=0.0,
        #     )

    def __enter__(self):
        self.stack.__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stack.__exit__(exc_type, exc_val, exc_tb)
        self.stack.close()
        if isinstance(exc_val, ProfilingDoneException):
            LOG.info("Profiling done")
            return True
        return False

    def start_profiling(self, name: str):
        LOG.info(f"Starting profiling for {name} at batch {self.batch_idx}")
        prof = None
        if self._profile_memory or self._trace:
            prof = self.stack.enter_context(
                torch.profiler.profile(
                    with_stack=self._profile_memory or self._trace,
                    profile_memory=self._profile_memory,
                    record_shapes=self._profile_memory or self._trace,
                    on_trace_ready=partial(
                        self.trace_handler, file_prefix=name, memory=self._profile_memory
                    ),
                )
            )
        return prof

    def step(self):
        if self.batch is None:
            raise ValueError("Batch not set")
        # if self.train:
        #     self.model.train()
        # else:
        #     self.model.eval()

        # if self.batch_idx == self.profile_batch_idx.start:
        #     self._prof = self.start_profiling("model_inference")

        if self._profile_memory and self._device.type == "cuda":
            # note this record history line is important for cuda memory profiling,
            # otherwise it will just produce a square block
            torch.cuda.memory._record_memory_history(
                max_entries=100000, stacks="python", context="alloc"
            )
            torch.cuda.reset_max_memory_allocated(self._device)

        # with torch.profiler.record_function(f"prof_step_{self.batch_idx}"):
        #     if self.train:
        #         with torch.profiler.record_function(f"prof_backward_{self.batch_idx}"):
        #             with GenericTimer() as loss_timer:
        #                 loss = self.loss_function(model=self.model, batch=self.batch)
        #             self._backward_dts.append(loss_timer.elapsed())
        #             self.optimizer.zero_grad(set_to_none=True)
        #             with GenericTimer() as bwd_timer:
        #                 loss.backward()
        #                 self.optimizer.step()
        #             self._loss_dts.append(bwd_timer.elapsed())
        #     else:
        #         with torch.profiler.record_function(f"prof_forward_{self.batch_idx}"):
        #             with GenericTimer() as fwd_timer:
        #                 _ = self.forward()
        #             self._forward_dts.append(fwd_timer.elapsed())
        #             LOG.info("Time forward: %2.6f", self._forward_dts[-1])

        # memory
        if self._profile_memory and self._device.type == "cuda":
            max_memory = torch.cuda.max_memory_allocated(self._device) / (1024**2)
            self._max_memory.append(max_memory)

        self.batch_idx += 1
        # if self.batch_idx == self.profile_batch_idx.stop:
        # raise ProfilingDoneException("Profiling done")

    @staticmethod
    def trace_handler(prof: torch.profiler.profile, file_prefix: str, memory: bool) -> None:
        if memory:
            LOG.info(f"Memory profiling done, dumping memory snapshot to {file_prefix}.pickle")
            torch.cuda.memory._dump_snapshot(f"{file_prefix}.pickle")  # type: ignore[no-untyped-call]
        LOG.info(f"Profiling done, dumping trace to {file_prefix}.json.gz")
        prof.export_chrome_trace(f"{file_prefix}.json.gz")
