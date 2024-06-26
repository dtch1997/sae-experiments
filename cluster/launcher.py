#!/usr/bin/env python3
from absl import app, flags
from lxm3 import xm, xm_cluster
from lxm3.contrib import ucl

import os
import dotenv
from sae_experiments.sweep import get_sweep

_LAUNCH_ON_CLUSTER = flags.DEFINE_boolean(
    "launch_on_cluster", False, "Launch on cluster"
)
_GPU = flags.DEFINE_boolean("gpu", False, "If set, use GPU")
_RAM = flags.DEFINE_integer("ram", 16, "RAM in GB")
_HOURS = flags.DEFINE_integer("hours", 16, "Walltime in hours")
_SINGULARITY_CONTAINER = flags.DEFINE_string(
    "container", "sae-experiments-latest.sif", "Path to singularity container"
)
_ENTRYPOINT = flags.DEFINE_string(
    "entrypoint", "sae_experiments.run", "Entrypoint"
)
_SWEEP  = flags.DEFINE_string(
    "sweep", "none", "Sweep name"
)
_HIGH_VRAM = flags.DEFINE_boolean("high_vram", False, "If set, restrict to high VRAM instances")

def get_cluster_name() -> str:
    return os.getenv("CLUSTER")

def main(_):
    # use script name as experiment title
    script_name = _ENTRYPOINT.value.split(".")[-1]
    with xm_cluster.create_experiment(experiment_title=script_name) as experiment:

        # Define the job requirements
        ram = _RAM.value
        if _GPU.value:
            job_requirements = xm_cluster.JobRequirements(gpu=1, ram=ram * xm.GB)
        else:
            job_requirements = xm_cluster.JobRequirements(ram=ram * xm.GB)

        extra_directives = []
        if _HIGH_VRAM.value:
            if get_cluster_name() == "cs":
                extra_directives = ["-l gpu_type=(a100|rtx8000|a100_80|a100_dgx)"]
            elif get_cluster_name() == "myriad":
                extra_directives = ["-ac allow=L"]
            else:
                raise ValueError(f"Unknown cluster name: {get_cluster_name()}")

        # Define the walltime
        walltime = _HOURS.value * xm.Hr

        # Define the executor
        if _LAUNCH_ON_CLUSTER.value:
            # This is a special case for using SGE in UCL where we use generic
            # job requirements and translate to SGE specific requirements.
            # Non-UCL users, use `xm_cluster.GridEngine directly`.
            executor = ucl.UclGridEngine(
                requirements = job_requirements,
                walltime = walltime,
                extra_directives = extra_directives
            )
        else:
            executor = xm_cluster.Local(job_requirements)

        # Define the python package
        entrypoint = _ENTRYPOINT.value
        spec = xm_cluster.PythonPackage(
            # NOTE: 'path' is a relative path to the launcher that contains
            # your python package (i.e. the directory that contains pyproject.toml)
            # Here, .../cluster --> .../
            path="..",
            # Entrypoint is the python module that you would like to
            # In the implementation, this is translated to
            #   python3 -m py_package.main
            entrypoint=xm_cluster.ModuleName(entrypoint),
            # # NOTE: by default, pip install --no-deps is used to install the package.
            # # This means that the dependencies required by the project are not installed.
            # # We override this default behavior by specifying the pip_args.
            # pip_args=[]
        )

        # Wrap the python_package to be executing in a singularity container.
        singularity_container = _SINGULARITY_CONTAINER.value
        if singularity_container is not None:
            spec = xm_cluster.SingularityContainer(
                spec,
                image_path=singularity_container,
            )

        [executable] = experiment.package(
            [xm.Packageable(spec, executor_spec=executor.Spec())]
        )

        args = list(get_sweep(_SWEEP.value))
        env_vars = [dotenv.dotenv_values(".env")] * len(args) 
        experiment.add(
            xm_cluster.ArrayJob(executable=executable, executor=executor, args=args, env_vars=env_vars)
        )


if __name__ == "__main__":
    app.run(main)