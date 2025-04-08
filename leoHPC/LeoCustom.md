Below is the provided text converted into Markdown. You can copy and paste this into your Markdown editor or save it as a file (for example, `user_environment_customization.md`).

---

# 5: User Environment Customization

*Created by Antonio De Nicola, last modified on Jan 23, 2025*

[ The Software Catalog ] [ The module command ] [ Customization of your account by installing fresh softwares ]

---

## The Software Catalog

CINECA offers a variety of third-party applications and community codes installed on its HPC systems. Most third-party software is installed using the software modules mechanism (see *The module command* section).

- **Catalog Organization:**  
  Information on available packages and their detailed descriptions is organized in a catalog divided by discipline on our website. To access the catalog, select "software" and "Application Software for Science."  
- **Cluster Access:**  
  The catalog is also accessible directly on HPC clusters via the commands `module` or `modmap` (see next section).

For specific questions about currently available software, please contact the specialist support.

---

## The Module Command

All software installed on the CINECA clusters is available as modules. By default, a set of basic modules is preloaded in your environment at login. To manage modules in the production environment, use the `module` command with various options. Below is a short description of common module command usages:

- **List available modules:**

  ```bash
  module avail
  ```

- **Load a module:**

  ```bash
  module load <appl>
  ```

- **Load a module with all its dependencies:**

  ```bash
  module load autoload <appl>
  ```

- **Show specific information and basic help on an application:**

  ```bash
  module help <appl>
  ```

- **List modules currently loaded in the session:**

  ```bash
  module list
  ```

- **Unload all loaded modules:**

  ```bash
  module purge
  ```

- **Unload a specific module:**

  ```bash
  module unload <appl>
  ```

---

## Customization of Your Account by Installing Fresh Software

To help users customize their production environment by installing new software, we offer a powerful tool named **Spack**. Spack is a multi-platform package manager that facilitates the easy installation of multiple versions and configurations of software. It is currently available on MARCONI, GALILEO100, and LEONARDO.

Below is a step-by-step guide to installing new software or a new release of existing software using Spack. For a comprehensive guide, please refer to the [official Spack documentation](https://spack.readthedocs.io).

### Loading the Spack Module on the Cluster

A module is provided to load a pre-configured Spack instance:

1. **Display available Spack modules:**

   ```bash
   $ modmap -m spack
   ```

2. **Load a specific Spack version:**

   ```bash
   $ module load spack/<version>
   ```

- **Prod vs. Preprod:**  
  When you load the "prod" module, you use a Spack instance configured for compilation in the CINECA environment. On a CINECA cluster, both "preprod" and "prod" Spack modules are available. Use the pre-production module only if no production module is available.

When you load a Spack module, the `setup-env.sh` file is sourced. Then:
- `$SPACK_ROOT` is initialized to `/cineca/prod/opt/tools/spack/<version>/none`
- The `spack` command is added to your `PATH`
- Command-line integration tools are also provided

The directory `/spack-<version>` is automatically created in a default space and contains subdirectories used by Spack during package installation. For example, on LEONARDO, the default area is `$PUBLIC`, while on MARCONI and GALILEO100 it is `$WORK/$USER`. You might see paths such as:

- **Sources cache:** `$PUBLIC/spack-<version>/cache`
- **Software installation root:** `$PUBLIC/spack-<version>/install`
- **Modulefiles location:** `$PUBLIC/spack-<version>/modules`
- **User scope:** `$PUBLIC/spack-<version>/user_cache`

*Note:* For MARCONI and GALILEO100 users, `$WORK` will be removed at the end of the project. To define different paths for the cache, installation, modules, and user scope directories, refer to the Spack manual.

---

### Listing the Software Packages Available to be Installed

To check if a software package is available via Spack, use:

```bash
$ spack list <package_name>
```

Or filter with `grep`:

```bash
$ spack list | grep <package_name>
```

---

### Providers

In the Spack environment, "virtual packages" (e.g., `mpi`) are defined and provided by multiple specific packages (e.g., `intel-oneapi-mpi`, `openmpi`, etc.).

- **List available virtual packages:**

  ```bash
  $ spack providers
  ```

- **List packages that provide a specific virtual package:**

  ```bash
  $ spack providers <virtual_package_name>
  ```

  *Example:*

  ```bash
  $ spack providers mpi
  ```

---

### Variants and Dependencies

If the package you want to install is available via Spack, you can view its build “variants” and required “dependencies” by executing:

```bash
$ spack info <package_name>
```

---

### Listing Installed Software Packages

CINECA staff has already installed, through the Spack module, a suite of compilers, libraries, tools, and applications. To list the packages already installed:

```bash
$ spack find
```

To check if a specific package (or virtual package) is already installed:

```bash
$ spack find <package_name>
$ spack find <virtual_package_name>
```

For a detailed list showing variants (`-v`), dependencies (`-d`), installation path (`-p`), and the unique hash (`-l`):

```bash
$ spack find -ldvp <package>
```

You can also list the packages installed with a specific variant:

```bash
$ spack find -l +<variant>
```

*Example:*

```bash
$ spack find -l +cuda
```

Or list packages that depend on a specific package:

```bash
$ spack find -l ^<package_name>
```

*Example:*

```bash
$ spack find -l ^openmpi
$ spack find -l ^mpi
```

Or list packages installed with a specific compiler:

```bash
$ spack find %<compiler>
```

To list all compilers already installed and ready for use:

```bash
$ spack compilers
```

---

### Installing a New Package

#### The Spec Command

Before installing a package, check its default spec (i.e., the combination of version, compiler, variants, and dependencies) with:

```bash
$ spack spec <package_name>
```

To display a detailed spec with its unique hash, use:

```bash
$ spack spec -l <package_name>
```

*Example:*

```bash
$ spack spec -l openmpi %intel
$ spack spec -l openmpi %gcc
```

The combination of all installation parameters is your **spec**. If you don’t select any parameters, Spack uses a default spec. To view installation status (installed or not) along with the hash, use:

```bash
$ spack spec -Il <package_name>
```

> The installation status is described by:
>
> - **“-”**: not installed  
> - **“+/^”**: installed / installed from another user
>
> If a dependency is not installed (indicated by “-”), Spack will install it automatically as an implicit dependency.

On CINECA clusters, it is recommended to always execute the `spack spec` command before installing a package to ensure its dependencies are satisfied by the CINECA installations (indicated by the `^` symbol).

*Example Comparison:*

To install **parmetis** with openmpi not provided by Cineca (new hash):

```bash
$ spack spec -Il parmetis %gcc@10.2.0 ^openmpi
```

To install **parmetis** using the Cineca openmpi installation (with a specific hash, e.g., `ov3ei7j`):

```bash
$ spack spec -Il parmetis %gcc@10.2.0 ^/ov3ei7j
```

#### Installing the Package

You can install a package with the default spec or a custom spec that specifies a version, compiler, variants, and dependencies:

- **Default Installation:**

  ```bash
  $ spack install <package_name>
  ```

  or

  ```bash
  $ spack install /<package_hash>
  ```

- **Custom Installation Syntax:**

  ```bash
  $ spack install <package_name>@<version> +/<variant> <variant>=<value> %<compiler>@<version> ^<dependency_name>
  ```

*Tip:* Run `spack info <package_name>` to view available versions, variants, and dependencies.

Ensure you run `spack spec` before installation and use the CINECA installations for dependencies if available.

#### Installing a New Compiler

If you install a new compiler, add it to your `compilers.yaml` by:

1. Installing the compiler package:

   ```bash
   $ spack install <compiler_package>
   ```

2. Loading the compiler package:

   ```bash
   $ spack load <compiler_package>
   ```

3. Adding it to Spack's list of compilers:

   ```bash
   $ spack compiler add
   $ spack compilers
   ```

---

## Module Command and Spack Management

After installing a package with Spack, you can load it using:

```bash
$ spack load <package_name>
```

Alternatively, you can create a modulefile and load the software package as a module. To create a modulefile for your installed software:

```bash
$ spack module tcl refresh --upstream-modules <package_name>
```

Then, you can list and load the modulefile:

```bash
$ module load spack
$ module av <package_module>
$ module load <package_module>
```

For more detailed information, please refer to the [Spack documentation](https://spack.readthedocs.io).

---

This concludes the Markdown conversion of the "User Environment Customization" section. If you need further modifications or additional sections converted, please let me know!