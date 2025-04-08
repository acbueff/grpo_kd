Below is the provided DCGP Section content converted into Markdown. You can copy and paste this text into your Markdown editor or save it as a file (for example, `dcgp_section.md`):

---

# DCGP Section

*Created by Antonio De Nicola, last modified by Daniele Di Bari on Mar 11, 2025*

---

## Table of Contents

1. **Production Environment**  
   1.1 SLURM Partitions  
   1.2 Programming Environment  
   1.3 MPI Environment  
   1.4 Scientific Libraries  
   1.5 Hardware Locality

---

## Production Environment

Since **LEONARDO** is a general purpose system and is used by several users at the same time, long production jobs must be submitted using a queuing system (scheduler). The scheduler guarantees that access to the resources is as fair as possible. The production environment on the **LEONARDO Data Centric General Purpose (DCGP)** partition is based on the **SLURM scheduler**.

**Node Sharing Policy:**  
LEONARDO employs a policy of node sharing among different jobs. A job may request only part of a node (for example, a few cores). This means that, at a given time, one physical node can be allocated to multiple jobs from different users. Nevertheless, exclusivity at the single-core level is ensured by low-level mechanisms.

There are two main modes of using compute nodes:

### Batch Mode

- **Purpose:** Intended for production runs.  
- **Method:** Users must prepare a shell script containing all the operations to execute once the requested resources are available. The job then runs on the compute nodes.  
- **Data Storage:** Store all your data, programs, and scripts in the `$WORK` or `$SCRATCH` filesystems, which are best suited for compute node access.  
- **Requirements:** You must have valid, active projects to run batch jobs and be aware of specific policies regarding project budgets.

*Example:* An `sbatch` submission script.

### Interactive Mode

- **Purpose:** Similar to batch mode, but allows interactive execution.  
- **Method:** Resources are allocated and the job is managed like any other submitted job. However, once the job is running, the user can interactively execute applications within the allocated limits. All allocated resources are available for the entire requested walltime (and are consequently billed).  

> **Note:**  
> *Interactive Mode under SLURM* differs from the traditional interactive execution in a Linux shell. Interactive application execution is allowed on compute nodes only via SLURM (see the sections below).

On login nodes, tasks such as data movement, archiving, code development, compilations, basic debugging, and very short test runs are permitted, provided these tasks do not exceed 10 minutes of CPU time and are exempt from billing under current policy.

For a general discussion, see the section **Scheduler and Job Submission**.

---

## SLURM Partitions

A list of partitions defined on the cluster (including access rights and resource definitions) can be displayed with the command:

```bash
$ sinfo -o "%10D %20F %P"
```

This command returns output showing, for each partition, the total number of nodes and the number of nodes by state in the format “Allocated/Idle/Other/Total”.

### Overview of SLURM Partitions for LEONARDO Data Centric

The following table summarizes the main features and limits for the LEONARDO Data Centric partitions:

| **SLURM Partition**       | **Job QoS** | **# cores/ # GPU per job**                                      | **Max Walltime** | **Max Nodes/Cores/Mem per User**       | **Max Nodes per Account** | **Priority** | **Notes** |
|---------------------------|-------------|------------------------------------------------------------------|------------------|----------------------------------------|---------------------------|--------------|-----------|
| lrd_all_serial (default)  | normal      | max = 4 physical cores (8 logical cpus), max mem = 30800 MB        | 04:00:00         | 1 node / 4 cores / 30800 MB             | 40                        | —            | No GPUs; Hyperthreading x2 |
| dcgp_usr_prod             | normal      | max = 16 nodes                                                  | 24:00:00         | —                                      | 512                       | 40           |           |
| dcgp_qos_dbg              | —           | max = 2 nodes; 2 nodes / 224 cores per user                        | 00:30:00         | —                                      | 512                       | 80           |           |
| dcgp_qos_bprod            | —           | min = 17 nodes, max = 128 nodes                                  | 24:00:00         | 128 nodes per user                     | 512                       | 60           | GrpTRES = 1536 node; min is 17 FULL nodes |
| dcgp_qos_lprod            | —           | max = 3 nodes; 3 nodes / 336 cores per user                        | 4-00:00:00       | —                                      | 512                       | 40           |           |

> **Note:** A maximum of 512 nodes per account is imposed on the **dcgp_usr_prod** partition. This means that, for each account, all jobs associated with it cannot run on more than 512 nodes simultaneously.

---

## Programming Environment

**LEONARDO Data Centric** compute nodes do not have GPUs; hence, GPU-enabled applications are meant for the Booster partition. The programming environment in DCGP includes a set of compilers and debugging/profiling tools suitable for CPU programming.

### Compilers

To list all available compilers on LEONARDO, run:

```bash
$ modmap -c compilers
```

The recommended compilers for the DCGP partition are the **Intel compilers** because the architecture is based on Intel processors. This may yield improved performance and stability compared to other compilers.

*Note:*  
CUDA-aware compilers (e.g., GNU, NVIDIA nvhpc, CUDA) are not recommended for DCGP since they target GPU-enabled applications and are documented in the LEONARDO Booster section.

#### Intel OneAPI Compilers

Initialize the environment with:

```bash
$ module load intel-oneapi-compilers/<VERSION>
```

The suite includes:

- **C/C++ compilers:**  
  - Classic: `icc` / `icpc`  
  - Nextgen: `icx` / `icpx`  
    - *ICX* is the Intel nextgen compiler based on Clang/LLVM and proprietary optimizations. It enables OpenMP TARGET offload (not applicable on DCGP).  
    - Use `icx` for C and `icpx` for C++; note that `icx` does not automatically determine C/C++ by file extension.
- **Fortran compilers:**  
  - Classic: `ifort`  
  - Beta: `ifx`  
    - *ifx* supports OpenMP TARGET directives for offloading to Intel GPUs (irrelevant on DCGP).

> **Notes:**  
> - ICX is a new compiler and may require porting for existing applications using ICC.  
> - Although ifx is available, Intel recommends using ifort for production Fortran applications until ifx matures.
> 
> Please refer to the official Intel Porting Guides for more details.

After loading the module, documentation is accessible via:

```bash
$ man ifort
$ man icc
```

---

## Debugger and Profilers

If your code crashes at runtime, you may need to analyze core files or run your code using a debugger.

### Compiler Flags for Debugging

Enable runtime checks during compilation with:

- `-O0` &mdash; Lower optimization level  
- `-g` &mdash; Produce debugging information

Additional flags are available per compiler:

#### PORTLAND Group (PGI) Compilers

- `-C` &mdash; Enable array bounds checking  
- `-Ktrap=ovf,divz,inv` &mdash; Configure behavior on floating-point exceptions (overflow, division by zero, invalid operands)

#### GNU Fortran Compilers

- `-Wall` &mdash; Enable warnings  
- `-fbounds-check` &mdash; Enable bounds checking

### Available Debuggers

- **GNU Debugger (gdb):** For serial debugging.
- **Valgrind:** For memory management, threading issues, and profiling. Valgrind includes various tools such as memory error detectors and call-graph profilers.

### Profilers

Profiling is used to analyze program performance. Commonly used profilers include:

#### gprof

- **Usage Example:**

  ```bash
  $ gfortran -pg -O3 -o myexec myprog.f90
  $ ./myexec
  $ ls -ltr
  # Locate the generated gmon.out file
  $ gprof myexec gmon.out
  ```

- **Line-Level Profiling:**  
  Use the `-g` flag with `-pg`:
  
  ```bash
  $ gfortran -pg -g -O3 -o myexec myprog.f90
  $ ./myexec
  $ gprof -annotated-source myexec gmon.out
  ```

*Note:* For MPI programs, set the environment variable `GMON_OUT_PREFIX` so that each task writes its own output file:

```bash
$ export GMON_OUT_PREFIX=<name>
```

---

## MPI Environment

For the **LEONARDO Data Centric** partition, the recommended MPI implementation is **Intel-OneAPI-MPI**, which does not support CUDA.

### Compiling with Intel-OneAPI-MPI

1. **Load the module:**  
   ```bash
   $ module load intel-oneapi-mpi/<version>
   ```
2. **Compiler Wrappers:**  
   - For **C/C++:**
     - `icpc` (classic), `icpx` (oneAPI)  
     - To compile using oneAPI:  
       ```bash
       $ mpiicpc -cxx=icpx
       ```
   - For **C:**
     - `icc` (classic), `icx` (oneAPI)  
       ```bash
       $ mpiicc -cc=iccx
       ```
   - For **Fortran:**
     - `ifort` (classic), `ifx` (oneAPI)  
       ```bash
       $ mpiifort -fc=ifx
       ```

*Example (Compiling Fortran Code):*

```bash
$ module load intel-oneapi-compilers/<VERSION>
$ module load intel-oneapi-mpi/<version>
$ mpiifort -o myexec myprog.f90
```

View compiler backend options using:

```bash
$ man mpiifort
```

### Running MPI Applications

There are two ways to launch MPI jobs:

#### Using the `mpirun` Launcher

- **Direct Launch:**

  ```bash
  $ mpirun ./mpi_exec
  ```
  
- **Via salloc:**

  ```bash
  $ salloc -N 2
  $ mpirun ./mpi_exec
  ```

- **Via sbatch:**  

  Create a batch script (`my_batch_script.sh`):

  ```bash
  #!/bin/sh
  mpirun ./mpi_exec
  ```
  
  Then submit:
  
  ```bash
  $ sbatch -N 2 my_batch_script.sh
  ```

#### Using the `srun` Launcher

- **Direct Launch:**

  ```bash
  $ srun -N 2 ./mpi_exec
  ```
  
- **Via salloc:**

  ```bash
  $ salloc -N 2
  $ srun ./mpi_exec
  ```

- **Via sbatch:**  

  Batch script example (`my_batch_script.sh`):

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

The libraries listed here do **not** support CUDA. (For GPU-accelerated libraries, see the LEONARDO Booster section.)

### Linear Algebra

- **BLAS:**  
  - OpenBLAS  
  - Intel-OneAPI-MKL
- **LAPACK:**  
  - OpenBLAS  
  - Intel-OneAPI-MKL
- **SCALAPACK:**  
  - netlib-scalapack  
  - Intel-OneAPI-MKL
- **Sparse Matrices:**  
  - PETSc (multi-node)  
  - SuperLU-dist (multi-node)  
  > *Note:* PETSc and SuperLU-dist are GPU‑accelerated in the Booster partition; here they are provided for non‑accelerated applications.
  
### Fast Fourier Transform

- **FFTW:** Available for both single‑node and multi‑node applications.

---

## Hardware Locality

Each compute node in the **DCGP** partition is equipped with:

- **Sockets:** 2 sockets, each with a multi‑core processor.
- **Cores:** 112 cores in total (56 cores per socket).
- **RAM:** 503 GiB available, divided into 8 NUMA nodes (4 per socket).

### Processor Details

The multi‑core processors are **Intel Xeon Platinum 8480+** (3.80 GHz, Turbo enabled) with:

- **Cores per Processor:** 56 cores  
- **Cache:**  
  - Each core: 2 MiB of L2 cache and 80 KiB of L1 cache  
  - Shared L3 cache: 105 MiB across all cores

---

This concludes the Markdown conversion of the DCGP Section text. If you need further adjustments or additional sections converted, please let me know!