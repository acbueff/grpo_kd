Below is the provided text converted into Markdown. You can copy and paste the text into your Markdown editor or save it as a file (e.g., `data_storage_filesystems.md`).

---

# 4: Data Storage and FileSystems

*Created by Antonio De Nicola, last modified by Alessandro Marani on Mar 07, 2025*

[ Data Storage architecture ] [ Backup Policies and Data Availability ] [ Monitoring the occupancy ] [ File permissions ] [ Endianness ]

---

## Data Storage Architecture

All HPC systems share the same logical disk structure and file systems definition.

The available storage areas can have multiple definitions/purposes:

- **Time-related:**
  - **Temporary:** Data are accessible for a defined time window before being deleted.
  - **Permanent:** Data are accessible up to six months after the end of the project.

- **Scope-related:**
  - **User specific:** Each username has a different data area.
  - **Project specific:** Data accessible by all users linked to the same project.

- **Locality:**
  - **Local:** Specific for each system.
  - **Shared:** The same area can be accessed by all HPC systems.

The available data areas are defined on all HPC clusters through predefined environment variables. You can access these areas simply using the following names:

```bash
cd $HOME
cd $SCRATCH
cd $WORK
cd $DRES
cd $FAST     # (Leonardo Only)
cd $PUBLIC   # (Leonardo Only)
```

> **Suggestion:**  
> It is strongly encouraged to use these environment variables instead of full paths when referring to your scripts and data.

---

## Overview of Available Data Areas

| Variable   | Nature & Scope                           | Capacity     | Frequency/Retention              | Notes                                                                                     |
|------------|------------------------------------------|--------------|----------------------------------|-------------------------------------------------------------------------------------------|
| **$HOME**  | Local, Permanent, User-specific, Backed  | 50 GB        | Daily backups                    | Data are critical, not so large, and should be preserved.                                 |
| **$WORK**  | Local, Permanent, Project-specific       | 1 TB         | —                                | Large data shared with project collaborators.                                             |
| **$FAST**  | Local, Permanent, Project-specific       | 1 TB         | —                                | *(Leonardo Only)* Faster I/O compared to other areas.                                     |
| **$SCRATCH** | Local, Temporary, User-specific        | Up to 20 TB | Temporary (files deleted after 40 days) | On Marconi the variable is named `$CINECA_SCRATCH`. Data shared with other users possible. |
| **$TMPDIR**| Local, Temporary, User-specific          | —            | Directory removed at job completion |                                                                                           |
| **$PUBLIC**| Permanent, User-specific, Shared         | 50 GB        | —                                | *(Leonardo Only)* Data are fully accessible (read, write, execute) to all group members.    |
| **$DRES**  | Permanent, Shared, Defined by project      | —            | —                                | Data intended to be maintained even beyond the project, usable on different CINECA hosts.   |

> **Note:**  
> All the filesystems are based on Lustre.

---

## Ethical Use of the SCRATCH Area

Users are encouraged to respect the intended use of the various areas. Although the SCRATCH area is not subject to restrictions (quotas) in order to facilitate the production of large amounts of data, it **must not** be used as a long-term storage solution. Users are warned **against** using commands (e.g., `touch`) or similar methods to extend the retention time of files beyond the 40‑day limit. Improper procedures will be monitored, and users may be subjected to restrictions up to a ban.

---

## Description of Data Areas

- **$HOME**
- **$WORK**
- **$FAST**
- **$SCRATCH**
- **$TMPDIR**
- **$PUBLIC**
- **$DRES**

---

## Backup Policies and Data Availability

- **Daily Backups:**  
  The `$HOME` filesystem is backed up daily. The daily backup procedure preserves a maximum of three different copies of the same file. Older versions of files are kept for 1 month. The last version of deleted files is kept for 2 months before being permanently removed from the backup archive.

- **Backup Agreements:**  
  Different backup policies may apply based on agreements. For further details, contact HPC support at [superc@cineca.it](mailto:superc@cineca.it).

- **Data Availability:**  
  Data, whether backed up or not, are available for the entire duration of the project. After a project expires, users retain full access to their data for an additional six months. Beyond this period, data availability is not guaranteed.

> **Important:**  
> Users are responsible for backing up their important data.

A scheme of data availability is illustrated in the figure below.  
*(Figure not provided here)*

---

## Monitoring the Occupancy

The occupancy status of all user-accessible areas, along with storage quota limits, can be monitored using simple commands available on all HPC clusters. Two commands are available:

- **cindata**
- **cinQuota** (available only for Galileo 100 and Leonardo)

Use the `-h` flag with either command to display help. Below is an example output for a DRES user:

```bash
$ cindata
$ cinQuota
```

---

## File Permissions

`$WORK` and `$DRES` are environment variables automatically set in the user environment.

- **$WORK:**  
  Points to a directory (fileset) specific to one of your user projects:  
  `/gpfs/work/<account_name>`

- **$DRES:**  
  Points to space where all DRES are defined:  
  `/gss/gss_work/`  
  To use a specific DRES type, refer to the following path:  
  `$DRES/<dres_name>`

The owner of the root directory is the "Principal Investigator" (PI) or the designated "owner" of the DRES. The group is typically the project name or the DRES name. Default permissions are:

- **Owner:** `rwx`
- **Group:** `rwx`
- **Others:** `-`

This configuration allows all project collaborators (sharing the same project group) to read and write into the project/DRES fileset, while other users cannot.

> **Recommendation:**  
> Create a personal subdirectory under `$WORK` and `$DRES`. By default, files in a personal subdirectory are private, but you can share the directory with other collaborators by modifying permissions, for example:
>
> ```bash
> chmod 777 mydir   # Open access to all users
> chmod 755 mydir   # More restrictive, but readable/executable by others
> ```

Since the `$WORK/$DRES` fileset is restricted to project collaborators, data sharing is active only among those users.

---

## Pointing $WORK to a Different Project: The chprj Command

You can modify the project associated with the `$WORK` variable using the **chprj** (change project) command.

- To list all your accounts (active or completed) and the default project:

  ```bash
  chprj -l
  ```

- To set `$WORK` to point to a different project (using `<account_no>`):

  ```bash
  chprj -d <account_no>
  ```

For more details, consult the help page:

```bash
chprj -h
chprj --help
```

> **Note on LEONARDO:**  
> The chprj command also applies to the `$FAST` variable.

For a comprehensive discussion on managing your data, please refer to the specific documentation.

---

## Endianness

Endianness refers to the attribute of a system that indicates whether integers are represented from left to right or right to left. Currently, all clusters in Cineca are **"little-endian."**

---

This concludes the Markdown conversion of the "Data Storage and FileSystems" section. If you require further modifications or additional sections to be converted, please let me know!