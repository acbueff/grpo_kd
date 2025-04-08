Below is the provided text converted into Markdown. You can copy and paste this into your Markdown editor or save it as a file (for example, `booster_section.md`).

---

# Booster Section

*Created by Antonio De Nicola, last modified by Diego Molinari on Mar 20, 2025*

---

## Table of Contents

1. **Production Environment**  
   1.1 Job Managing and SLURM Scheduler  
   1.2 Programming Environment  
   1.3 MPI Environment  
   1.4 Scientific Libraries  
   1.5 Hardware Locality (CPU)  
   1.6 Intra Node Connection Environment

---

## Production Environment

Since **LEONARDO** is a general purpose system and is used by several users at the same time, long production jobs must be submitted using a queuing system (scheduler). The scheduler guarantees that the access to the resources is as fair as possible. The production environment on the **LEONARDO Booster** partition is based on the **SLURM scheduler**.

LEONARDO is based on a policy of node sharing among different jobs, i.e. a job can request resources that might span only part of a node (for example, a few cores and one GPU). This means that, at a given time, one physical node can be allocated to multiple jobs by different users. Nevertheless, exclusivity at the level of a single core is guaranteed by low-level mechanisms.

There are two main modes of using compute nodes:

### Batch Mode

- **Purpose:** Intended for production runs.
- **Usage:**  
  Users must prepare a shell script with all the operations to be executed once the requested resources are available. The job will then run on the compute nodes.
- **Data Location:**  
  Store all your data, programs, and scripts in the `$WORK` or `$SCRATCH` filesystems, as these are optimized for compute node access.
- **Requirements:**  
  You must have valid active projects to run batch jobs and be aware of any specific policies regarding project budgets on our systems.

An example submission using `sbatch`:

```bash
#SBATCH -A <account_name>
# [other SBATCH options...]
./your_program
```

### Interactive Mode

- **Purpose:** Similar to batch mode, but allows interactive execution.
- **Usage:**  
  You still specify the resources to allocate, but once the job starts, you can interactively run applications within the allocated limits.
- **Billing:**  
  All allocated resources are available for the entire requested walltime (and consequently billed) during the submission process.

> **Note:**  
> *Interactive Mode under SLURM* has a different meaning compared to a typical interactive shell. Interactive execution of applications is allowed on compute nodes only via SLURM (see the next sections).

On login nodes, it is permitted to perform tasks such as data movement, archiving, code development, compilations, basic debugging, and very short test runs (not exceeding 10 minutes of CPU time, free of charge under the current billing policy).

For a general discussion, see the section **User Environment Customization**.

---

## Job Managing and SLURM Scheduler

A list of partitions defined on the cluster, with access rights and resource definitions, can be displayed using the command:

```bash
$ sinfo -o "%10D %20F %P"
```

This command returns a readable output which shows, for each partition, the total number of nodes and the number of nodes by state in the format "Allocated/Idle/Other/Total".

### SLURM Partitions Overview

Below is a summary table of the main features and limits imposed on the partitions of the **LEONARDO Booster**:

| **SLURM Partition**          | **Job QoS** | **# cores/# GPU per job**                                              | **Max Walltime** | **Max Running Jobs per User**      | **Max Nodes/Cores/GPUs per User**                         | **Priority** | **Notes**                                          |
|------------------------------|-------------|------------------------------------------------------------------------|------------------|------------------------------------|-----------------------------------------------------------|--------------|----------------------------------------------------|
| lrd_all_serial (default)     | normal      | max = 4 physical cores (8 logical cpus), max mem = 30800 MB            | 04:00:00         | 1 node / 4 cores / 30800 MB, 40     | —                                                         | —            | No GPUs; Hyperthreading x2                         |
| boost_usr_prod               | normal      | max = 64 nodes                                                         | 24:00:00         | 40                                 | —                                                         | —            |                                                    |
| boost_qos_dbg                | —           | max = 2 nodes                                                          | 00:30:00         | 2 nodes / 64 cores / 8 GPUs, 80     | —                                                         | —            |                                                    |
| boost_qos_bprod              | —           | min = 65 nodes, max = 256 nodes                                          | 24:00:00         | 256 nodes, 60                      | runs on 1536 nodes; min is 65 FULL nodes                  | —            |                                                    |
| boost_qos_lprod              | —           | max = 3 nodes                                                          | 4-00:00:00       | 3 nodes / 12 GPUs, 40              | —                                                         | —            |                                                    |

> **Note for EUROFusion users:**  
> For details regarding dedicated queues, please refer to the dedicated document.

---

## Programming Environment

**LEONARDO Booster** compute nodes host four A100 GPUs per node (CUDA compute capability 8.0). The most recent versions of the **NVIDIA CUDA Toolkit** and the **NVIDIA nvhpc compilers** (formerly PGI, supporting CUDA Fortran) are available in the module environment.

### Compilers

To check the complete list of available compilers on LEONARDO, use:

```bash
$ modmap -c compilers
```

#### Available CUDA-Aware Compilers

- **GNU Compiler Collection (GCC)**
- **NVIDIA nvhpc (ex PGI)**
- **CUDA**

*Note:* Intel compilers are available but do not support CUDA; details are provided in the LEONARDO Data Centric partition section.

#### GNU Compiler Collection (GCC)

- **Availability:**  
  The GNU compilers are always available. GCC version 8.5.0 is available without loading any module. Newer versions are available in the module environment.
- **Compiler Names:**
  - `g77` &mdash; Fortran77 compiler
  - `gfortran` &mdash; Fortran95 compiler
  - `gcc` &mdash; C compiler
  - `g++` &mdash; C++ compiler

Documentation is available via the `man` command after loading the GNU module:

```bash
$ man gfortran
$ man gcc
```

#### NVIDIA nvhpc (ex PGI + NVIDIA CUDA)

As of August 5, 2020, the "PGI Compilers and Tools" technology is now part of the **NVIDIA HPC SDK**, available as a free download.

- **Commands:**
  - `nvc` &mdash; Compile C source files (C11; supports GPU programming with OpenACC, and multicore CPU programming with OpenACC/OpenMP)
  - `nvc++` &mdash; Compile C++ source files (C++17; supports GPU programming with C++17 parallel algorithms and OpenACC, and multicore CPU programming with OpenACC/OpenMP)
  - `nvfortran` &mdash; Compile FORTRAN source files (supports ISO Fortran 2003 and many features of ISO Fortran 2008; supports CUDA Fortran and OpenACC)
  - `nvcc` &mdash; CUDA C/C++ compiler driver

For legacy reasons, the suite also provides PGI compiler commands:

- `pgcc` &mdash; Compile C source files  
- `pgc++` &mdash; Compile C++ source files  
- `pgf77` &mdash; Compile FORTRAN 77 source files  
- `pgf90` &mdash; Compile FORTRAN 90 source files  
- `pgf95` &mdash; Compile FORTRAN 95 source files  

To enable CUDA C++ or CUDA Fortran and link with the CUDA runtime libraries, use the `-cuda` option (note: `-Mcuda` is deprecated). The `-gpu` option can be used to tailor compilation for target accelerator regions.

**Parallelization Options:**

- **OpenACC:**  
  Enabled via the `-acc` flag.
- **OpenMP:**  
  Enabled via the `-mp` flag.  
  GPU offload via OpenMP is enabled by the `-mp=gpu` option.

#### CUDA

**Compute Unified Device Architecture (CUDA)** is NVIDIA’s platform for parallel computing on GPUs. CUDA enables developers to accelerate computing applications by offloading compute‑intensive portions to thousands of GPU cores. Developers can use languages such as C, C++, Fortran, Python, and MATLAB with CUDA extensions.

CUDA compilers are available both inside the nvhpc module and as stand-alone modules.

---

## Debugger and Profilers

If your code fails at runtime, you might need to analyze core files or use debuggers and profilers.

### Compiler Flags for Debugging

The following flags are generally available for all compilers and are recommended for debugging:

- `-O0` &mdash; Lower optimization level  
- `-g` &mdash; Produce debugging information

Additional compiler-specific flags:

#### PORTLAND Group (PGI) Compilers

- `-C` &mdash; Enable array bounds checking  
- `-Ktrap=ovf,divz,inv` &mdash; Control behavior on exceptions (e.g., FP overflow, divide-by-zero, invalid operands)

#### GNU Fortran Compilers

- `-Wall` &mdash; Enable all warnings  
- `-fbounds-check` &mdash; Enable array subscript checking

### Debuggers Available

- **GNU Debugger (gdb):** Serial debugger available with GNU toolchain.
- **Valgrind:** A suite of tools for memory management, threading bugs, and profiling.

### Profilers

Profiling helps identify performance bottlenecks. Some available profilers include:

#### gprof

- **Usage Example:**

  ```bash
  $ gfortran -pg -O3 -o myexec myprog.f90
  $ ./myexec
  $ ls -ltr
  # Look for the generated gmon.out file
  $ gprof myexec gmon.out
  ```

- **Line-Level Profiling:**  
  Use the `-g` flag along with `-pg` for detailed annotated source output.

  ```bash
  $ gfortran -pg -g -O3 -o myexec myprog.f90
  $ ./myexec
  $ gprof -annotated-source myexec gmon.out
  ```

*Note:* When profiling MPI programs, set the environment variable `GMON_OUT_PREFIX` so that each task produces a separate `gmon.out` file.

#### Nvidia Nsight System (GPU Profiler)

- A system‑wide performance analysis tool designed for both CPU and GPU profiling.
- **Usage Example in an MPI job:**

  ```bash
  $ mpirun <options> nsys profile -o ${PWD}/output_%q{OMPI_COMM_WORLD_RANK} -f true --stats=true --cuda-memory-usage=true <your_code> <input> <output>
  ```

- **Workaround for Temporary Files:**  
  (For NVHPC versions prior to 24.3)  
  ```bash
  $ export TMPDIR=/dev/shm  # or use $SCRATCH as needed
  $ ln -s $TMPDIR /tmp/nvidia
  $ mpirun ... nsys profile ...
  ```
  *It is recommended to request exclusive use of the compute node with:*
  
  ```bash
  #SBATCH --exclusive
  ```

> **Important Update:**  
> Since NVHPC/24.3, `nsys profile` writes temporary output to `$TMPDIR` rather than `/tmp`. If you need more than 10 GB for temporary profiler files, export `TMPDIR` to a location with more capacity (e.g., `/dev/shm` or `$SCRATCH`).

---

## MPI Environment

**OpenMPI** is the most common MPI implementation on LEONARDO Booster. It is installed in the GNU environment and configured to support CUDA. (For details on Intel-OneAPI-MPI without CUDA support, refer to the Data Centric partition section.)

### Compiling with OpenMPI

To compile MPI applications using OpenMPI:

1. **Load the OpenMPI module:**  
   Use `modmap -m openmpi` to see available versions.
2. **Select the MPI compiler wrapper** for Fortran, C, or C++ codes.

**Available Compiler Wrappers:**

- For **C++:**
  - `g++`
  - `mpic++`
  - `mpiCC`
  - `mpicxx`
- For **C:**
  - `gcc`
  - `mpicc`
- For **FORTRAN:**
  - `gfortran`
  - `mpif77`
  - `mpif90`
  - `mpifort`

**Example (Compiling C Code):**

```bash
$ module load openmpi/<version>
$ mpicc -o myexec myprog.c
```

You can view the backend compiler options using the `-show` flag:

```bash
$ mpicc -show
```

Alternatively, consult the manual:

```bash
$ man mpicc
```

### Running MPI Applications

There are two primary ways to launch MPI jobs:

#### Using the `mpirun` Launcher

- **Directly:**
  ```bash
  $ mpirun ./mpi_exec
  ```
- **Via salloc:**
  ```bash
  $ salloc -N 2
  $ mpirun ./mpi_exec
  ```
- **Via sbatch:**  
  Create a batch script (e.g., `my_batch_script.sh`):

  ```bash
  #!/bin/sh
  mpirun ./mpi_exec
  ```
  Then submit:
  ```bash
  $ sbatch -N 2 my_batch_script.sh
  ```

#### Using the `srun` Launcher

- **Directly:**
  ```bash
  $ srun -N 2 ./mpi_exec
  ```
- **Via salloc:**
  ```bash
  $ salloc -N 2
  $ srun ./mpi_exec
  ```
- **Via sbatch:**  
  Create a batch script:

  ```bash
  #!/bin/sh
  srun -N 2 ./mpi_exec
  ```
  Then submit:
  ```bash
  $ sbatch -N 2 my_batch_script.sh
  ```

---

## Scientific Libraries

The libraries listed here are GPU-accelerated and support CUDA (for non‑CUDA-aware libraries, refer to the LEONARDO Data Centric section).

- **NVIDIA Math Libraries:**  
  Available by loading the `nvhpc` module (use `modmap -m nvhpc` to view versions).
- **Non‑NVIDIA Math Libraries (with CUDA support):**  
  For example, load the module:
  
  ```bash
  $ module load magma/<version>
  ```

### Library Categories

- **Linear Algebra:**  
  - BLAS: NVIDIA cuBLAS, MAGMA  
  - LAPACK: NVIDIA cuSOLVER, MAGMA  
  - SCALAPACK: SLATE  
  - Eigenvalue Solvers: NVIDIA cuSOLVER, MAGMA (single‑node), SLATE, ELPA, slepC (multi‑node)  
  - Sparse Matrices: NVIDIA cuSPARSE, PETSc (multi‑node), SuperLU‑dist (multi‑node); Hypre (multi‑node)
- **Fast Fourier Transform:**  
  - NVIDIA cuFFT/cuFFTW (single‑node)

---

## Hardware Locality (CPU)

Each compute node in the **Booster** partition is equipped with one Intel Xeon Platinum 8358 Processor (3.40 GHz, Turbo enabled), featuring:

- **Cores:** 32 cores, each with 1.25 MiB of L2 cache and 80 KiB of L1 cache.
- **L3 Cache:** 48 MiB shared across all cores.
- **RAM:** 503 GiB available, divided into 2 NUMA nodes.

A detailed description of the node topology is available in the next section.

---

## Intra Node Connection Environment

Each compute node in the **Booster** partition is equipped with:

- 4 NVIDIA A100 GPUs
- 2 dual‑port HDR100 NICs  
  (providing 100 Gbps per GPU and 400 Gbps per node)

### GPU and NIC Interconnection

- **GPU Topology:**  
  The GPUs are interconnected in an all‑to‑all topology, each linked via 4 bonded sets of NVLinks (NV4). All GPUs are closer to the first NUMA node, resulting in a GPU-to-core affinity of cores 0–15 for all 4 GPUs.

To visualize the node’s topology, run the following command:

```bash
[<username>@lrdnXXXX ~]$ nvidia-smi topo -m
```

Example output:

```
        GPU0    GPU1    GPU2    GPU3    NIC0    NIC1    NIC2    NIC3    CPU Affinity    NUMA Affinity
GPU0    X       NV4     NV4     NV4     PXB     NODE    NODE    NODE    0-15            0
GPU1    NV4     X       NV4     NV4     NODE    PXB     NODE    NODE    0-15            0
GPU2    NV4     NV4     X       NV4     NODE    NODE    PXB     NODE    0-15            0
GPU3    NV4     NV4     NV4     X       NODE    NODE    NODE    PXB     0-15            0
NIC0    PXB     NODE    NODE    NODE    X       NODE    NODE    NODE        
NIC1    NODE    PXB     NODE    NODE    NODE    X       NODE    NODE        
NIC2    NODE    NODE    PXB     NODE    NODE    NODE    X       NODE        
NIC3    NODE    NODE    NODE    PXB     NODE    NODE    NODE    X         
```

**Legend:**

- **X:** Self  
- **SYS:** Connection traversing PCIe and the SMP interconnect between NUMA nodes (e.g., QPI/UPI)  
- **NODE:** Connection traversing PCIe and the interconnect between PCIe Host Bridges within a NUMA node  
- **PHB:** Connection traversing PCIe as well as a PCIe Host Bridge (typically the CPU)  
- **PXB:** Connection traversing multiple PCIe bridges (without traversing the PCIe Host Bridge)  
- **PIX:** Connection traversing at most a single PCIe bridge  
- **NV#:** Connection traversing a bonded set of # NVLinks

**NIC Legend:**

- **NIC0:** mlx5_0  
- **NIC1:** mlx5_1  
- **NIC2:** mlx5_2  
- **NIC3:** mlx5_3

---

This concludes the Markdown conversion of the provided Booster Section text. If you require further adjustments or additional conversions, please let me know!