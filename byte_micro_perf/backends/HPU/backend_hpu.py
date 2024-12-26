# Copyright 2024 ByteDance and/or its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import json
import logging
import subprocess

from datetime import timedelta
from typing import Any, Dict, List

import torch
import torch.distributed as dist
import torch.distributed.distributed_c10d as dist_c10d

from backends import module_store
from backends.backend import Backend

from dataclasses import dataclass


@dataclass
class HPUDeviceProperties:
    total_memory: int


logging.basicConfig(level=logging.INFO)
log = logging.getLogger("PerfEngine")
log.setLevel(logging.INFO)


class BackendHPU(Backend):
    def __init__(self, workload_dict, vendor_path):
        log.debug(f"BackendHPU.__init__() called by pid: {os.getpid()}")
        super().__init__(workload_dict, vendor_path)
        self.use_hpu_graphs = os.getenv("USE_HPU_GRAPHS", "0").lower() in ["1", "true"]

    def get_device_name(self):
        log.debug(f"BackendHPU.get_device_name() called by pid: {os.getpid()}")
        # return torch.hpu.get_device_name()
        return "Gaudi2"

    def get_torch_device_name(self):
        log.debug(f"BackendHPU.get_torch_device_name() called by pid: {os.getpid()}")
        import habana_frameworks.torch as htorch

        return "hpu"

    def get_device_properties(self):
        log.debug(f"BackendHPU.get_device_properties() called by pid: {os.getpid()}")

        dramSize = 102106132480
        return HPUDeviceProperties(dramSize)

    def get_device_count(self):
        log.debug(f"BackendHPU.get_device_count() called by pid: {os.getpid()}")
        # return torch.hpu.device_count()
        device_count = (
            int(
                subprocess.check_output(
                    "hl-smi -Q module_id -f csv | wc -l", shell=True, text=True
                )
            )
            - 1
        )
        return device_count

    def set_device(self, device_index: int):
        log.debug(f"BackendHPU.set_device() called by pid: {os.getpid()}")
        import habana_frameworks.torch as htorch

        # HPU only support set_device to 0
        torch.hpu.set_device(0)

    def get_device(self):
        log.debug(f"BackendHPU.get_device() called by pid: {os.getpid()}")
        import habana_frameworks.torch as htorch

        return torch.hpu.current_device()

    def device_synchronize(self):
        log.debug(f"BackendHPU.device_synchronize() called by pid: {os.getpid()}")
        import habana_frameworks.torch as htorch

        torch.hpu.synchronize()

    def empty_cache(self):
        log.debug(f"BackendHPU.empty_cache() called by pid: {os.getpid()}")
        # empty_cache() is non-op on HPU
        return

    def get_dist_module(self):
        log.debug(f"BackendHPU.get_dist_module() called by pid: {os.getpid()}")
        return dist

    def initialize_ccl(self, rank, world_size):
        log.debug(
            f"BackendHPU.initialize_ccl() called by pid: {os.getpid()}, rank = {rank}, world_size = {world_size}"
        )
        import habana_frameworks.torch as htorch

        # check device_count
        device_count = self.get_device_count()
        if world_size > device_count:
            world_size = device_count
        if rank >= world_size:
            return False
        self.set_device(rank)

        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "29500"
        os.environ["RANK"] = str(rank)
        os.environ["PT_HPU_ENABLE_LAZY_COLLECTIVES"] = "True"

        # create default process group
        import habana_frameworks.torch.core.hccl

        dist.init_process_group("hccl", rank=rank, world_size=world_size)

        return True

    def get_op_instance(self):
        from backends import module_store

        default_op_registry = module_store.op_registry.copy()
        if self.op_name in default_op_registry:
            self.op = default_op_registry[self.op_name]
            if self.use_hpu_graphs:
                if self.op_name in [
                    "allgather",
                    "allreduce",
                    "alltoall",
                    "broadcast",
                    "p2p",
                    "reduce_scatter",
                ]:
                    os.environ["PT_HPU_ENABLE_LAZY_COLLECTIVES"] = True
                import habana_frameworks.torch as htorch

                self.op = htorch.hpu.wrap_in_hpu_graph(self.op)
        else:
            raise NotImplementedError

    def _run_operation(self, operation, inputs):
        import habana_frameworks.torch as htorch

        result = operation(*inputs)
        htorch.core.mark_step()
        return result
