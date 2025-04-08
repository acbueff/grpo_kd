Below is a Markdown conversion of the provided content. You can copy this text into your Markdown editor or file (e.g., `LEONARDO_User_Guide.md`) as needed:

---

# LEONARDO User Guide

*Created by Antonio De Nicola, last modified by Daniele Di Bari on Mar 13, 2025*

---

## Table of Contents

1. **System Architecture**  
   1.1 Hardware Details  
   1.2 Nominal Peak Performance  
2. **Access to the System**  
3. **Accounting**  
4. **Budget Linearization Policy**  
5. **Disks and Filesystems**  
6. **Software Environment**  
   6.1 Module Environment  
   6.2 Spack Environment  
7. **Graphic Session**  
8. **Network**  
   8.1 Booster Cells  
   8.2 DCGP Cells  
   8.3 Advanced Information  
9. **Documents**

> **Note:**  
> Sections *Production Environment* and *Programming Environment* are specific for the two partitions:
> - **Booster**
> - **DCGP**

---

## System Architecture

Leonardo HPC system is the new pre-exascale Tier‑0 EuroHPC Joint Undertaking supercomputer hosted by CINECA and currently built in the Bologna Technopole, Italy. It is supplied by **EVIDEN ATOS**, and it is based on two new specifically‑designed compute blades, which are available through two distinct SLURM partitions on the cluster:

- **X2135 GPU Blade:**  
  Based on NVIDIA Ampere A100‑64 accelerators – **LEONARDO Booster** partition  
- **X2140 CPU‑only Blade:**  
  Based on Intel Sapphire Rapids processors – **LEONARDO Data Centric General Purpose (DCGP)** partition

The overall system architecture also uses **NVIDIA Mellanox InfiniBand High Data Rate (HDR)** connectivity, with smart in‑network computing acceleration engines that enable extremely low latency and high data throughput to provide the highest AI and HPC application performance and scalability.

The system also includes a Capacity Tier and a Fast Tier storage, based on **DDN Exascaler**.

The Operating System is **RedHat Enterprise Linux 8.6**.

- **Early availability:** March, 2023 (Booster)  
- **Start of pre‑production:**  
  - June, 2023 (Booster)  
  - January 2024 (DCGP)  
- **Start of production:**  
  - August 2023 (Booster)  
  - February 2024 (DCGP)

---

## Hardware Details

### Booster

- **Model:**  
  Atos BullSequana X2135 "Da Vinci" single‑node GPU blade
- **Racks:**  
  116  
- **Nodes:**  
  3456
- **Processors:**  
  Single socket 32 cores Intel Ice Lake CPU  
  *(1 x Intel Xeon Platinum 8358, 2.60GHz, TDP 250W)*
- **Accelerators:**  
  4 x NVIDIA Ampere GPUs/node, 64GB HBM2e, NVLink 3.0 (200GB/s)
- **Cores:**  
  32 cores/node
- **RAM:**  
  512 GB (8 x 64GB) DDR4 3200 MHz
- **Peak Performance:**  
  About **309 Pflop/s**

### DCGP

- **Model:**  
  Atos BullSequana X2140 three‑node CPU blade
- **Racks:**  
  22
- **Nodes:**  
  1536
- **Processors:**  
  Dual socket 56 cores Intel Sapphire Rapids CPU  
  *(2 x Intel Xeon Platinum 8480p, 2.00GHz, TDP 350W)*
- **Accelerators:**  
  *Not applicable*
- **Cores:**  
  112 cores/node
- **RAM:**  
  512 GB (16 x 32GB) DDR5 4800 MHz
- **Peak Performance:**  
  **7.2 Pflops/s**

### Internal Network

- **Technology:**  
  DragonFly+ 200 Gbps (NVIDIA Mellanox InfiniBand HDR)
- **Connectivity per Node:**  
  - Booster: 2 x dual port HDR100 per node  
  - DCGP: Single port HDR100 per node

### Storage (raw capacity)

- **Capacity Tier:**  
  106 PB based on DDN ES7990X and Hard Drive Disks
- **Fast Tier:**  
  5.4 PB based on DDN ES400NVX2 and Solid State Drives

---

## Nominal Peak Performance

### Node Performance

| **Component**                      | **Theoretical Peak Performance**  |
|------------------------------------|-----------------------------------|
| CPU (nominal/peak freq.)           | 1680 Gflops                       |
| GPU                                | 75000 Gflops                      |
| **Total**                          | 76680 Gflops                      |
| Memory Bandwidth (nominal/peak)    | 24.4 GB/s                         |

---

## Access to the System

All the login nodes (4 Ice Lake, no‑GPU) share an identical environment and can be reached with the SSH (Secure Shell) protocol using the **collective** hostname:

```bash
$ login.leonardo.cineca.it
```

This will establish a connection to one of the available login nodes. To connect to LEONARDO, you can also explicitly indicate the login nodes:

```bash
$ login01-ext.leonardo.cineca.it
$ login02-ext.leonardo.cineca.it
$ login05-ext.leonardo.cineca.it
$ login07-ext.leonardo.cineca.it
```

**Mandatory Access:**  
Access to LEONARDO requires two‑factor authentication (2FA). Please refer to the relevant section of the User Guide to activate and connect via 2FA. For information about data transfer from other computers, follow the instructions in the dedicated section on Data Storage or consult the document on Data Management.

---

## Accounting

The accounting (consumed budget) is active from the start of the production phase. For accounting information, please consult our dedicated section.

The `account_name` (or project) is important for batch executions. You must specify an `account_name` in the scheduler using the flag `-A`:

```bash
#SBATCH -A <account_name>
```

You can list all the `account_name` associated with your username using the `saldo` command:

```bash
$ saldo -b          # Reports projects defined on LEONARDO Booster
$ saldo --dcgp -b   # Reports projects defined on LEONARDO DCGP
```

**Note:**  
Accounting is measured in consumed core hours; however, it is also influenced by factors such as requested memory, local storage, and number of GPUs. Refer to the dedicated section for details.

---

## Budget Linearization Policy

On LEONARDO, as on other HPC clusters in CINECA, a linearization policy for project budgets has been implemented. The goal is to improve response time by aligning the usage of CPU hours with the actual project size (total amount of core‑hours assigned). Please refer to the dedicated page for further details.

---

## Disks and Filesystems

The storage organization conforms to the CINECA infrastructure (see the section on Data Storage and Filesystems). In addition to the home directory (`$HOME`), each user is assigned:

- **Scratch Area:** `$SCRATCH` (or `$CINECA_SCRATCH`)  
  A large disk for run‑time data and file storage.  
- **Public Area:** `$PUBLIC`  
  A user‑specific area useful for sharing installations (also the default location for SPACK sub‑directories).
- **Work Area:** `$WORK`  
  A dedicated area for each active project, accessible by all project collaborators.
- **Fast Area:** `$FAST`  
  A subset of the scratch filesystem on fast NVMe SSD flash drives, reserved for each active project.

An extension of the default `$WORK` quota (1 TB) may be granted if justified, while `$FAST` is limited to 1 TB per project.

### Filesystem Details

| **Filesystem**      | **Capacity**      | **Quota**                | **Properties**                                                                                                          |
|---------------------|-------------------|--------------------------|-------------------------------------------------------------------------------------------------------------------------|
| `$HOME`             | 0.46 PiB          | 50 GB per user           | Permanent, backed up (suspended), user‑specific                                                                       |
| `$CINECA_SCRATCH`   | 40 PiB            | No quota                 | HDD storage, temporary, user‑specific, no backup, automatic cleaning for data older than 40 days                        |
| `$PUBLIC`           | 0.46 PiB          | 50 GB per user           | Permanent, user‑specific, no backup                                                                                    |
| `$WORK`             | 30 PB             | 1 TB per project         | Permanent, project‑specific, no backup (extensions can be considered if needed, contact: [superc@cineca.it](mailto:superc@cineca.it)) |
| `$FAST`             | 3.5 PB            | 1 TB per project         | Permanent, project‑specific, no backup                                                                                 |

> **Important:**  
> The automatic cleaning of the scratch area is now active!

### Temporary Node-Local Area

A temporary area is available on login and compute nodes (generated at job start on compute nodes and removed at job end), accessible via the `$TMPDIR` environment variable:

- **Login Nodes:**  
  Located on local SSD disks (14 TB capacity), mounted as `/scratch_local` (`TMPDIR=/scratch_local`). This shared area has no quota but is subject to cleaning procedures if improperly used.
- **Serial Node:**  
  On `lrd_all_serial` (14 TB capacity), managed via the `slurm job_container/tmpfs` plugin.  
  - Provides a job‑specific, private temporary file system space (with `/tmp` and `/dev/shm`), visible via `df -h`.
  - Maximum allocation of 1 TB for serial jobs; default `/tmp` size is 10 GB if not explicitly requested (use sbatch directive or `--gres=tmpfs:XX` to request more).
- **DCGP Nodes:**  
  Local SSD disks (3 TB capacity).  
  - Managed via a plugin that mounts private `/tmp` and `/dev/shm` areas at job start (visible via `df -h /tmp`), and unmounts them at job end.
  - Maximum allocation of 3 TB; default `/tmp` size is 10 GB unless explicitly requested (via sbatch directive or `--gres=tmpfs:XX`).
  - **Note:** For DCGP jobs, the requested `gres/tmpfs` resource contributes to the consumed budget, affecting the number of equivalent core hours.
- **Diskless Booster Nodes:**  
  Provide local RAM storage of fixed size (10 GB), with no possibility of increase and with `gres/tmpfs` disabled.

For more information, see the dedicated section on Data Storage and Filesystems.

Since all filesystems are based on Lustre, the usual Unix command `quota` does not work. Instead, use the local command `cindata` to query disk usage and quota:

```bash
$ cindata   # Use 'cindata -h' for help
```

Or use the tool `cinQuota` available in the module **cintools**:

```bash
$ cinQuota
```

For further details on monitoring disk occupancy, refer to the dedicated section.

---

## Software Environment

### Module Environment

The software modules are organized into different profiles and functional categories (compilers, libraries, tools, applications, etc.). There are two main profile types:

- **Programming Type:**  
  Includes "base" and "advanced" profiles for compilation, debugging, and profiling activities.
- **Domain Type:**  
  Includes profiles such as chem‑phys, lifesc, etc., for production activities.

The **Base** profile is the default and is automatically loaded after login. It contains basic modules for programming activities (IBM, GNU, PGI, CUDA compilers, math libraries, profiling and debugging tools, etc.).

If you need to use a module from another profile (for example, an application module), you must load the corresponding profile first:

```bash
$ module load profile/<profile_name>
$ module load <module_name>
```

Almost all software on LEONARDO was installed using the Spack manager, which automatically loads dependencies, so the `autoload` command is not required.

To list all currently loaded profiles, use:

```bash
$ module list
```

To see if a specific module is available and which profile to load for it, use the `modmap` command:

```bash
$ modmap -m <module_name>
```

> **Note:**  
> On LEONARDO, some modules are compiled to support GPUs while others are for CPU‑only use. Check the module names for compiler information (e.g., `gromacs/2022.3--intel-oneapi-mpi--2021.10.0--oneapi–2023.2.0`).  
> - Modules compiled with **gcc**, **nvhpc**, **cuda** should only be used on the **Booster** partition.  
> - Modules compiled with **intel oneapi** are suitable for running on the **DCGP** partition.  
>  
> Please refer to the specific sections for **Booster Programming Environment** and **DCGP Programming Environment** for more details.

### Spack Environment

If you do not find a desired software application, you can install it yourself using the “spack” environment. Load the corresponding module to use Spack:

```bash
$ module load spack
```

*Please note that we are still optimizing the LEONARDO software stack, and more installations may be added or replaced. Always check available modules with:*

```bash
$ module av
```

On LEONARDO (unlike other CINECA clusters), the default area for Spack directories (`/cache`, `/install`, `/modules`, `/user_cache`) is the `$PUBLIC` area (see Disks and Filesystems).

---

## Graphic Session

*It will be available soon.*

---

## Network

This section describes the network architecture of Leonardo, a state‑of‑the‑art interconnect system designed for high‑performance computing (HPC). It delivers low latency and high bandwidth by leveraging **NVIDIA Mellanox InfiniBand High Data Rate (HDR)** technology with a Dragonfly+ topology.

**Key Features:**

- **Cell Structure:**  
  All nodes are divided into cells.
- **Connectivity:**  
  Cells are interconnected in an all‑to‑all topology using 18 independent connections between any two cells (one per L2 switch).
- **Within-Cell Topology:**  
  A non‑blocking two‑layer fat tree topology is used within each cell.
- **System Composition:**
  - 19 cells for the Booster partition.
  - 2 cells for the DCGP partition.
  - 1 hybrid cell composed of both accelerated and conventional compute nodes (36 Booster + 288 DCGP).
  - 1 cell dedicated to management, storage, and login systems.
- **Adaptive Routing:**  
  Enabled to alleviate network congestion.

---

This concludes the Markdown conversion of the provided LEONARDO User Guide content. If you need additional formatting adjustments or further sections converted, please let me know!