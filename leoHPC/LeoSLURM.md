Below is the provided text converted into Markdown. You can copy and paste this into your Markdown editor or save it as a file (for example, `slurm_guide.md`).

---

# Introduction to SLURM Workload Manager

*Created by Antonio De Nicola, last modified on Feb 28, 2025*

[ Introduction to SLURM Workload Manager ] [ Basic Usage of SLURM ] [ SLURM Resources ] [ Submit Jobs ] [ Job Script Examples ] [ Essential SLURM Commands ] [ SLURM Environment Variables ]

CINECA HPC clusters are accessed via a dedicated set of login nodes. These nodes are intended for simple tasks such as customizing the user environment (e.g., installing applications), transferring data, and performing basic pre- and post-processing of simulation data. Access to the compute nodes is managed by the workload manager to ensure fair resource access for all users—production jobs must be submitted using a scheduler.

CINECA uses **Slurm** (Simple Linux Utility for Resource Management) as its workload manager and batch system. Slurm is an open-source, highly scalable job scheduling system with three key functions:

- **Resource Allocation:** Allocates compute nodes to users for a specified duration.
- **Job Management:** Provides a framework for starting, executing, and monitoring work (usually parallel jobs) on the allocated nodes.
- **Queue Management:** Manages resource contention by handling the queue of pending jobs.

There are two main modes of using compute nodes:

- **Batch Mode:** Intended for production runs. You prepare a shell script with all the commands to execute once the requested resources are available. Data, programs, and scripts should reside in the `$WORK` or `$SCRATCH` filesystems.
- **Interactive Mode:** Similar to batch mode in that you specify resources, but once the job starts, you can interactively run applications within the allocated resources (all resources remain available during the entire walltime and are billed accordingly).

> **Note:**  
> *Interactive Mode under SLURM* differs from typical interactive execution in a Linux shell. Interactive execution is allowed on compute nodes only via SLURM. On login nodes, tasks such as data movement, archiving, code development, compilation, basic debugging, and very short test runs (not exceeding 10 minutes of CPU time) are allowed free of charge under the current billing policy.

A comprehensive documentation of SLURM and examples for job submission is available in a separate section of this chapter as well as on the original SchedMD site.

---

# Basic Usage of SLURM

With SLURM, you specify the tasks you want to run, and the system manages executing those tasks and returning the results. If the requested resources are not immediately available, SLURM queues your job until they are.

Typically, you create a batch job—a shell script containing the commands to execute as well as directives that specify job characteristics and resource requirements (e.g., number of processors, CPU time). Once you have created your job script, you can reuse or modify it for subsequent runs.

## Simple SLURM Job Script Example

Below is an example SLURM job script that runs an application with a maximum wall time of one hour, requesting 1 node with 32 cores:

```bash
#!/bin/bash

#SBATCH --nodes=1                    # Request 1 node
#SBATCH --ntasks-per-node=32         # 32 tasks per node
#SBATCH --time=1:00:00               # Time limit: 1 hour
#SBATCH --error=myJob.err            # Standard error file
#SBATCH --output=myJob.out           # Standard output file
#SBATCH --account=<account_no>       # Account name
#SBATCH --partition=<partition_name> # Partition name
#SBATCH --qos=<qos_name>             # Quality of service

./my_application
```

---

# SLURM Resources

A job’s resources are requested via SLURM directives; SLURM matches these requests with available resources per administrator-defined rules.

### Resource Types

- **Server-level resources:** e.g., walltime.
- **Trackable resources (TRES):** e.g., number of CPUs or nodes.
- **Generic resources (GRES):** e.g., GPUs on systems that have them.

### Request Syntax

The SLURM syntax depends on the type of resource. For example:

```bash
#SBATCH --time=10:00:00               # Server-level: Walltime
#SBATCH --ntasks-per-node=1           # Trackable: Tasks per node
#SBATCH --gres=gpu:2                  # Generic: Request 2 GPUs
```

Resources can be requested either via SLURM directives in your job script or by providing options to the `sbatch` or `salloc` command.

---

# SLURM Job Script and Directives

A SLURM job script consists of:

- An optional shell specification.
- SLURM directives, which begin with `#SBATCH`.
- Tasks (programs or commands) to be executed.

For resource requests:

- **Trackable Resources:**  
  Use directives such as:
  
  ```bash
  #SBATCH --nodes=NN               # Number of nodes
  #SBATCH --ntasks-per-node=CC     # Number of tasks per node
  #SBATCH --cpus-per-task=TT       # Number of threads/CPUs per task
  ```

  *Example for a mixed MPI/OpenMP job with 2 MPI processes and 8 threads per node:*

  ```bash
  #SBATCH --nodes=2
  #SBATCH --ntasks-per-node=8
  ```

- **Processing Time:**  
  Request wall clock time using:

  ```bash
  #SBATCH --time=1:00:00            # 1 hour
  ```

- **Memory Allocation:**  
  The default memory allocation depends on the partition. You can specify memory explicitly:

  ```bash
  #SBATCH --mem=10000               # Request 10000 MB per node
  #SBATCH --mem=10GB                # Alternatively, 10GB
  ```

- **MPI Tasks/OpenMP Threads Affinity:**  
  To control process binding, use directives such as:

  ```bash
  #SBATCH --cpu-bind=<cores|threads>
  #SBATCH --cpus-per-task=<number>
  ```
  
  When launching with `srun`, ensure these environment variables are correctly exported:

  ```bash
  export SRUN_CPUS_PER_TASK=$SLURM_CPUS_PER_TASK
  ```

---

# Other SLURM Directives

Additional directives in your job script may include:

```bash
#SBATCH --account=<account_no>     # Project account to be charged (use "saldo -b" for a list)
#SBATCH --job-name=<name>          # Job name
#SBATCH --partition=<destination>  # Partition/queue destination
#SBATCH --qos=<qos_name>           # Quality of service
#SBATCH --output=<out_file>        # Output file; default is "slurm-<PID>"
#SBATCH --error=<err_file>         # Error file
#SBATCH --mail-type=<events>       # Email notification options (NONE, BEGIN, END, FAIL, etc.)
#SBATCH --mail-user=<email>        # Email address for notifications
```

### Contracted Directive Syntax

Some SLURM directives offer a shortened form:

```bash
#SBATCH -N <NN>      # Equivalent to: --nodes=<NN>
#SBATCH -c <TT>      # Equivalent to: --cpus-per-task=<TT>
#SBATCH -t <value>   # Equivalent to: --time=<value>
#SBATCH -A <account_no>  # Equivalent to: --account=<account_no>
#SBATCH -J <name>    # Equivalent to: --job-name=<name>
#SBATCH -p <dest>    # Equivalent to: --partition=<dest>
#SBATCH -q <qos>     # Equivalent to: --qos=<qos>
#SBATCH -o <out>     # Equivalent to: --output=<out>
#SBATCH -e <err>     # Equivalent to: --error=<err>
```

> **Note:**  
> The directives `--mem`, `--mail-type`, `--mail-user`, and `--ntasks-per-node` cannot be contracted. Although the `-n` option exists for the number of tasks, it may be misleading—it represents the total number of tasks, not tasks per node. Use `--ntasks-per-node` for clarity.

---

# Submit Jobs

Once your SLURM job script is ready, you submit the job to the workload manager.

- **Batch Mode Submission:**

  ```bash
  $ sbatch [opts] your_job_script
  ```

  *Options may include: `--nodes=<nodes_no>`, `--ntasks-per-node=<tasks_per_node_no>`, `--account=<account_no>`, `--partition=<name>`, etc.*

- **Interactive Mode Submission:**

  There are two methods:

  1. **Using `salloc`:**

     ```bash
     $ salloc [opts] <command>
     ```

     Example interactive session:
     
     ```bash
     salloc -N 1 --ntasks-per-node=8   # Request 1 node with 8 tasks
     squeue -u $USER                    # Check that allocation is ready
     hostname                           # Runs on the front-end (login node)
     srun hostname                      # Runs on the allocated compute node
     exit                               # Ends the salloc allocation
     ```

  2. **Using `srun` with `--pty`:**

     ```bash
     srun -N 1 --ntasks-per-node=8 --pty /bin/bash
     ```
     
     > **Warning:**  
     > When using `salloc`, your prompt remains on the login node, so it is easy to forget that an interactive job is running. Always use `srun` (or equivalent) for commands that must execute on the compute nodes.

---

# Job Script Examples

## Serial Job Example

A typical serial job script that asks for 10 minutes of wall clock time and runs a serial application (R) in the directory `$CINECA_SCRATCH/test/`:

```bash
#!/bin/bash
#SBATCH --time=00:10:00
#SBATCH --nodes=1 --ntasks-per-node=1 --cpus-per-task=1
#SBATCH --mem=10000
#SBATCH --out=job.out
#SBATCH --account=<account_no>

cd $CINECA_SCRATCH/test/
module load autoload r
R < data > out.txt
```

## MPI Job Example

A typical MPI job script requesting 8 tasks across 2 nodes for a 1-hour job, compiled with the Intel compiler and MPI library:

```bash
#!/bin/bash
#SBATCH --time=01:00:00
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --ntasks-per-socket=2
#SBATCH --cpus-per-task=1
#SBATCH --mem=<mem_per_node>
#SBATCH --partition=<partition_name>
#SBATCH --qos=<qos_name>
#SBATCH --job-name=jobMPI
#SBATCH --err=myJob.err
#SBATCH --out=myJob.out
#SBATCH --account=<account_no>

module load intel intelmpi
srun myprogram < myinput > myoutput
```

## OpenMPI Job Example

For a serial OpenMPI job on a single node without hyperthreading:

```bash
#!/bin/bash
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=<48_or_32>    # Use 48 (MARCONI SKL/GALILEO100) or 32 (MARCONI100)
#SBATCH --partition=<partition_name>
#SBATCH --qos=<qos_name>
#SBATCH --mem=<mem_per_node>
#SBATCH --out=myJob.out
#SBATCH --err=myJob.err
#SBATCH --account=<account_no>

module load intel
export SRUN_CPUS_PER_TASK=$SLURM_CPUS_PER_TASK
export OMP_NUM_THREADS=<48_or_32>
srun myprogram < myinput > myoutput
```

## Hybrid MPI+OpenMPI Job Example

A typical hybrid MPI+OpenMP job requiring 8 MPI tasks across 2 nodes with 4 OpenMP threads per task:

```bash
#!/bin/bash
#SBATCH --time=01:00:00
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --ntasks-per-socket=2
#SBATCH --cpus-per-task=4
#SBATCH --mem=<mem_per_node>
#SBATCH --partition=<partition_name>
#SBATCH --qos=<qos_name>
#SBATCH --job-name=jobMPI
#SBATCH --err=myJob.err
#SBATCH --out=myJob.out
#SBATCH --account=<account_no>

module load intel intelmpi
export SRUN_CPUS_PER_TASK=$SLURM_CPUS_PER_TASK
export OMP_NUM_THREADS=4
export OMP_PLACES=cores
export OMP_PROC_BIND=true
srun myprogram < myinput > myoutput
```

## Hybrid MPI+OpenMPI with Pure MPI Job Example

If you wish to run an MPI code compiled with OpenMP flags as a pure MPI code, explicitly set `OMP_NUM_THREADS` to 1:

```bash
#!/bin/bash
#SBATCH --time=01:00:00
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=1
#SBATCH --partition=<partition_name>
#SBATCH --qos=<qos_name>
#SBATCH --mem=86000
#SBATCH --out=myJob.out
#SBATCH --err=myJob.err
#SBATCH --account=<account_no>

module load intel
export SRUN_CPUS_PER_TASK=$SLURM_CPUS_PER_TASK
export OMP_NUM_THREADS=1
srun myprogram < myinput > myoutput
```

---

# Chaining Multiple Jobs

To chain multiple jobs (for example, to use the output of one job as input for the next), you can use the `-d` or `--dependency` option with `sbatch`. For example, to run `job2.cmd` only after `job1.cmd` finishes successfully:

```bash
$ sbatch job1.cmd
submitted batch job 100
$ sbatch -d afterok:100 job2.cmd
submitted batch job 101
```

Alternatively:

```bash
$ sbatch job1.cmd
submitted batch job 100
$ sbatch --dependency=afterok:100 job2.cmd
submitted batch job 102
```

The available options for dependency include: `afterany`, `afternotok`, `afterok`, etc. (See the `sbatch` man page for full details.)

---

# High Throughput Computing with SLURM

**Array jobs** allow you to submit multiple similar jobs with a single submission. The maximum allowed number of runs in an array job depends on the cluster. Use the `--array` (or `-a`) option with `sbatch`.

### Examples

- Submit 20 serial runs with indices from 0 to 20:

  ```bash
  sbatch --array=0-20 -N1 job.cmd
  ```

- Submit a job array with specific indices (e.g., 1, 2, 5, 8):

  ```bash
  sbatch --array=1,2,5,8 -N1 job.cmd
  ```

- Submit a job array with indices 1 to 7 with a step size of 2 (i.e., 1, 3, 5, 7):

  ```bash
  sbatch --array=1-7:2 -N1 job.cmd
  ```

When you submit a job array, SLURM sets the following environment variables for each element:

- `SLURM_ARRAY_JOB_ID`: Job ID of the array.
- `SLURM_ARRAY_TASK_ID`: The job array index.
- `SLURM_ARRAY_TASK_COUNT`: The number of tasks in the array.
- `SLURM_ARRAY_TASK_MAX`: The highest array index.
- `SLURM_ARRAY_TASK_MIN`: The lowest array index.

*Example:* If you submit:

```bash
sbatch --array=1-3 -N1 job.cmd
```

and receive:

```bash
Submitted batch job 100
```

Then the environment for each task will be similar to:

- **Task 1:**  
  `SLURM_JOB_ID=100`, `SLURM_ARRAY_JOB_ID=100`, `SLURM_ARRAY_TASK_ID=1`, `SLURM_ARRAY_TASK_COUNT=3`, etc.

- **Task 2:**  
  `SLURM_JOB_ID=101`, `SLURM_ARRAY_JOB_ID=100`, `SLURM_ARRAY_TASK_ID=2`, ...

- **Task 3:**  
  `SLURM_JOB_ID=102`, `SLURM_ARRAY_JOB_ID=100`, `SLURM_ARRAY_TASK_ID=3`, ...

Most SLURM commands recognize both the single `SLURM_JOB_ID` and the combination of `SLURM_ARRAY_JOB_ID` and `SLURM_ARRAY_TASK_ID` (e.g., "100_2" to identify task 2).

Two additional options allow you to specify a job's stdin, stdout, and stderr file names:
- `%A` is replaced by `SLURM_ARRAY_JOB_ID`
- `%a` is replaced by `SLURM_ARRAY_TASK_ID`

For example, the default output file format for a job array is `slurm-%A_%a.out`.

---

# Essential SLURM Commands

Below is a summary of commonly used SLURM commands:

| **Command**                | **Description**                                                                          |
|----------------------------|------------------------------------------------------------------------------------------|
| `sbatch` / `srun` / `salloc` | Submit a job                                                                            |
| `squeue`                   | Lists jobs in the queue                                                                  |
| `sinfo`                    | Prints queue information about nodes and partitions                                      |
| `sbatch <batch script>`    | Submits a batch script to the queue                                                      |
| `scancel <jobid>`          | Cancels a job                                                                            |
| `scontrol hold <jobid>`    | Puts a job on hold                                                                       |
| `scontrol release <jobid>` | Releases a held job                                                                      |
| `scontrol update <jobid>`  | Changes attributes of a submitted job                                                    |
| `scontrol requeue <jobid>` | Requeues a running, suspended, or finished SLURM batch job into a pending state            |
| `scontrol show job <jobid>`| Produces a detailed report for the job                                                   |
| `sacct -k` or `--timelimit-min` | Displays data about jobs with a given time limit                                     |
| `sacct -A <account_list>`  | Displays jobs for a specific list of accounts                                            |
| `sstat`                    | Displays information about CPU, Task, Node, Resident Set Size, and Virtual Memory          |
| `sshare`                   | Displays shared resource information (for a user, repo, job, partition, etc.)             |
| `sprio`                    | Displays job scheduling priority based on multiple factors                               |

For more detailed information, consult the respective man pages or system-specific guides.

---

# Check Job Status

To check the status of your jobs:

- List all jobs (default format):

  ```bash
  squeue
  ```

- List jobs in a more readable format with custom options:

  ```bash
  squeue --format=...
  ```

- List only your jobs:

  ```bash
  squeue -u $USER
  ```

- Check a specific job:

  ```bash
  squeue --job <job_id>
  squeue --job <job_id> -l    # Full details
  scontrol show job <job_id>  # Detailed information
  ```

---

# Check Queue Status

The command `sinfo` displays information about nodes and partitions. For example:

```bash
sinfo -o "%20P %10a %10l %15F %10z"
```

This command shows available partitions, their status, timelimit, node state (Allocated/Idle/Other/Total), and specifications (sockets:cores:threads). Other useful options:

- `sinfo -p <partition>`: Long format for a specific partition.
- `sinfo -d`: Information about offline nodes.
- `sinfo --all`: Displays more details.
- `sinfo -i <n>`: Top-like display, updating every `n` seconds.
- `sinfo -l` or `--long`: Detailed info, including reasons for node downtime.
- `sinfo -n <node>`: Information about a specific node.

---

# Delete a Job

To cancel a job, use `scancel`:

```bash
scancel <jobID>
```

- To cancel specific array tasks (e.g., tasks 1 to 3 from job array 100):

  ```bash
  scancel 100_[1-3]
  ```

- To cancel specific tasks (e.g., tasks 4 and 5 from job array 100):

  ```bash
  scancel 100_4 100_5
  ```

- To cancel all elements of job array 100:

  ```bash
  scancel 100
  ```

For more details, refer to the `scancel` man page.

---

# SCONTROL Usage with Job Arrays

The command `scontrol show job <jobid>` includes additional fields for job array support:
- **JobID:** Unique identifier for the job.
- **ArrayJobID:** The JobID of the first element of the job array.
- **ArrayTaskID:** The array index for a particular job record.

To update an individual job array element:

```bash
scontrol update JobID=100_2 name=my_job_name
scontrol suspend 100      # Suspend all tasks in job array 100
scontrol resume 100       # Resume all tasks in job array 100
scontrol suspend 100_3    # Suspend only task 3
scontrol resume 100_3     # Resume only task 3
```

---

# SQUEUE for Job Arrays

When a job array is submitted, only one job record is initially created. Additional records are generated as tasks change state. By default, `squeue` lists all tasks associated with a job array on one line, with the "array_task_id" indicated using a regular expression. Use the `--array` or `-r` option to display each array element on a separate line.

For example:

```bash
squeue -j 100_2,100_3
squeue -s 100_2.0,100_3.0
```

---

# SLURM Environment Variables

SLURM sets several environment variables for each job. Some important ones include:

- **SLURM_JOB_NAME:** The name of the job.
- **SLURM_NNODES** or **SLURM_JOB_NUM_NODES:** Number of nodes allocated for the job.
- **SLURM_JOBID** or **SLURM_JOB_ID:** The job ID.
- **SLURM_JOB_NODELIST:** List of nodes assigned to the job.
- **SLURM_SUBMIT_DIR:** Directory from which the job was submitted.
- **SLURM_SUBMIT_HOST:** Host from which the job was submitted.
- **SLURM_CLUSTERNAME:** Name of the cluster.
- **SLURM_JOB_PARTITION:** Partition used by the job.

These variables facilitate dynamic job scripts. For example, you can use `$SLURM_JOB_NAME` to retrieve the job name and `$SLURM_SUBMIT_DIR` for the submission directory.

> **Warning:**  
> The variable `$SLURM_JOB_NODELIST` displays node names in a contracted form (e.g., node ranges). Square brackets indicate the range of node IDs.

### Job TMPDIR

When a job starts, each compute node gets a temporary area on its local storage:

```bash
TMPDIR=/scratch_local/slurm_job.$SLURM_JOB_ID
```

This directory is exclusive to the job's owner and is accessible via the `$TMPDIR` variable. It is removed when the job ends, so ensure you save any essential data elsewhere (e.g., in `$HOME`, `$WORK`, or `$CINECA_SCRATCH`). For multi-node jobs, if shared access is required, use the shared filesystems instead.

---

This concludes the Markdown conversion of the SLURM introduction and usage section. If you need further modifications or additional sections converted, please let me know!