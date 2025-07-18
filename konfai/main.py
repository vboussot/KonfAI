import argparse
import os
import sys

import torch.multiprocessing as mp
from torch.cuda import device_count

from konfai import konfai_nb_cores
from konfai.utils.utils import Log, TensorBoard, setup

sys.path.insert(0, os.getcwd())


def main():
    """
    Entry point for launching KonfAI training locally.

    - Parses arguments (if any) via a setup parser.
    - Initializes distributed environment based on available CUDA devices or CPU cores.
    - Launches training via `mp.spawn`.
    - Manages logging and TensorBoard context.

    KeyboardInterrupt is caught to allow clean manual termination.
    """
    parser = argparse.ArgumentParser(description="KonfAI", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    try:
        with setup(parser) as distributed_object:
            with Log(distributed_object.name, 0):
                world_size = device_count()
                if world_size == 0:
                    world_size = int(konfai_nb_cores())
                distributed_object.setup(world_size)
                with TensorBoard(distributed_object.name):
                    mp.spawn(distributed_object, nprocs=world_size)
    except KeyboardInterrupt:
        print("\n[KonfAI] Manual interruption (Ctrl+C)")


def cluster():
    """
    Entry point for launching KonfAI on a cluster using Submitit.

    - Parses cluster-specific arguments: job name, nodes, memory, time limit, etc.
    - Sets up distributed environment based on number of nodes and GPUs.
    - Configures Submitit executor with job specs.
    - Submits the job to SLURM (or another Submitit-compatible backend).

    Environment variables:
        KONFAI_OVERWRITE: Set to force overwrite of previous training runs.
        KONFAI_CLUSTER: Mark this as a cluster job (used downstream).

    Raises:
        KeyboardInterrupt: On manual interruption.
        Exception: Any submission-related error is printed and causes exit.
    """
    parser = argparse.ArgumentParser(description="KonfAI", formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Cluster manager arguments
    cluster_args = parser.add_argument_group("Cluster manager arguments")
    cluster_args.add_argument("--name", type=str, help="Task name", required=True)
    cluster_args.add_argument("--num-nodes", "--num_nodes", default=1, type=int, help="Number of nodes")
    cluster_args.add_argument("--memory", type=int, default=16, help="Amount of memory per node")
    cluster_args.add_argument(
        "--time-limit",
        "--time_limit",
        type=int,
        default=1440,
        help="Job time limit in minute",
    )
    cluster_args.add_argument(
        "--resubmit",
        action="store_true",
        help="Automatically resubmit job just before timout",
    )
    try:
        with setup(parser) as distributed_object:
            args = parser.parse_args()
            config = vars(args)
            os.environ["KONFAI_OVERWRITE"] = "True"
            os.environ["KONFAI_CLUSTER"] = "True"

            n_gpu = len(config["gpu"].split(","))
            distributed_object.setup(n_gpu * int(config["num_nodes"]))
            import submitit

            executor = submitit.AutoExecutor(folder="./Cluster/")
            executor.update_parameters(
                name=config["name"],
                mem_gb=config["memory"],
                gpus_per_node=n_gpu,
                tasks_per_node=n_gpu // distributed_object.size,
                cpus_per_task=config["num_workers"],
                nodes=config["num_nodes"],
                timeout_min=config["time_limit"],
            )
            with TensorBoard(distributed_object.name):
                executor.submit(distributed_object)
    except KeyboardInterrupt:
        print("\n[KonfAI] Manual interruption (Ctrl+C)")
    except Exception as e:
        print(e)
        exit(1)
