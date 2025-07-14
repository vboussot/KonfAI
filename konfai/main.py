import argparse
import os
from torch.cuda import device_count
import torch.multiprocessing as mp
from konfai.utils.utils import setup, TensorBoard, Log
from konfai import KONFAI_NB_CORES

import sys
sys.path.insert(0, os.getcwd())

def main():
    parser = argparse.ArgumentParser(description="KonfAI", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    try:
        with setup(parser) as distributedObject:
            with Log(distributedObject.name, 0):
                world_size = device_count()
                if world_size == 0:
                    world_size = int(KONFAI_NB_CORES())
                distributedObject.setup(world_size)
                with TensorBoard(distributedObject.name):
                    mp.spawn(distributedObject, nprocs=world_size)
    except KeyboardInterrupt:
        print("\n[KonfAI] Manual interruption (Ctrl+C)")

def cluster():
    parser = argparse.ArgumentParser(description="KonfAI", formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Cluster manager arguments
    cluster_args = parser.add_argument_group('Cluster manager arguments')
    cluster_args.add_argument('--name', type=str, help='Task name', required=True)
    cluster_args.add_argument('--num-nodes', '--num_nodes', default=1, type=int, help='Number of nodes')
    cluster_args.add_argument('--memory', type=int, default=16, help='Amount of memory per node')
    cluster_args.add_argument('--time-limit', '--time_limit', type=int, default=1440, help='Job time limit in minute')
    cluster_args.add_argument('--resubmit', action='store_true', help='Automatically resubmit job just before timout')
    try:
        with setup(parser) as distributedObject:
            args = parser.parse_args()
            config = vars(args)
            os.environ["KONFAI_OVERWRITE"] = "True"
            os.environ["KONFAI_CLUSTER"] = "True"

            n_gpu = len(config["gpu"].split(","))
            distributedObject.setup(n_gpu*int(config["num_nodes"]))
            import submitit
            executor = submitit.AutoExecutor(folder="./Cluster/")
            executor.update_parameters(name=config["name"], mem_gb=config["memory"], gpus_per_node=n_gpu, tasks_per_node=n_gpu//distributedObject.size, cpus_per_task=config["num_workers"], nodes=config["num_nodes"], timeout_min=config["time_limit"])
            with TensorBoard(distributedObject.name):
                executor.submit(distributedObject)
    except KeyboardInterrupt:
        print("\n[KonfAI] Manual interruption (Ctrl+C)")
    except Exception as e:
        print(e)
        exit(1)