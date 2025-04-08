Below is the converted text into Markdown. You can copy and paste this into your Markdown editor or save it as a file (e.g., `Access_and_Security_Guide.md`).

---

# Cineca HPC Access and Security Guide

*Created by Antonio De Nicola, last modified on Feb 03, 2025*

[ First access ] [ Access to the systems ] [ Access via Secure Shell (SSH) ] [ Remote Connection Manager (RCM) ] [ Managing Password ] [ Access to download or upload data ] [ Policy for password definition ]

---

## Prerequisites

Access to any section of the Cineca HPC requires the activation of two-factor authentication (2FA) for each **USER ACCOUNT**. This extra security step verifies user identity by requiring a second, independent factor, ensuring enhanced system security. Even if the correct account password is provided, 2FA prevents unauthorized access.

This access modality operates seamlessly with standard protocols such as the SSH client. Before connecting to the cluster, users must request an SSH certificate from our Identity Provider (IP) via the **smallstep client**. During the request, a web page will automatically open in your browser, prompting you to authenticate with our IP by entering a one-time password (OTP). Following successful authentication, the server issues a time-limited certificate valid for 12 hours. This certificate allows you to connect to Cineca systems via your SSH client.

---

## 2.1: How to Install the Smallstep Client and Configure 2FA/OTP

### First Access

For first-time access and activation of 2FA, follow these steps:

1. **Configure OTP and Password:**  
   - Activate the 2FA setup from the link you received by email after being enabled on our systems.  
   - Configure OTP (One-Time Password) authentication for your account.

2. **Install and Configure the Smallstep Client:**  
   - Install and configure the smallstep client on your local PC.  

   > **Note:**  
   > For services like Adacloud that authenticate via the web, the smallstep installation and configuration are unnecessary. The website will prompt for both password and OTP during login.

---

## Access to the Systems

Once you have activated 2FA, configured the smallstep client, and obtained the temporary certificate (via the smallstep client using the `step` command), you can access Cineca HPC sections in several ways:

### Access via Secure Shell (SSH)

SSH is commonly used for remote access, allowing you to execute commands, run programs, and transfer files securely.  
- **On Linux and macOS:** The SSH client is typically pre-installed.  
- **On Windows:** You may need to install an SSH client; popular options include PowerShell, OpenSSH, PuTTY, or Tectia.  

> **Important:**  
> Connecting using the 2FA procedure does not require you to provide a password.

After 12 hours, you must generate a new certificate using the smallstep client.

**Access Commands:**

```bash
ssh <username>@login.marconi.cineca.it
ssh <username>@login.g100.cineca.it
ssh <username>@login.leonardo.cineca.it
```

You can add the `-X` option to enable X11 display forwarding.

Your login shell will be either **bash** or **tcsh**. To change your default login shell, contact HPC support at **superc@cineca.it**.

> **Note:**  
> Login is prevented on systems where you do not have budget accounts.  
> If you encounter the error `"WARNING: REMOTE HOST IDENTIFICATION HAS CHANGED!"`, please check the FAQ section.  
> We have identified a potential issue with OpenSSH 8.6; check with `ssh -V` and consult our FAQ page for solutions.

---

## Remote Connection Manager (RCM)

The remote visualization service at Cineca is provided via the **Remote Connection Manager (RCM)** application. With RCM, you can graphically inspect your data without moving them to your local workstation.

> **Note:**  
> Similar to SSH, using RCM does not require a password.

---

## Managing Password

Using the new Identity Provider website service, you can now manage your authentication credentials directly. This includes:
- Resetting your password.
- Reconfiguring OTP on your smartphone.
- Generating new recovery authentication codes.

For detailed procedures on these tasks, please refer to the dedicated documentation.

---

## Access to Download or Upload Data

You can use several protocols/utilities to access our systems for data transfer:

- **SCP (Secure Copy)** or **SFTP (SSH File Transfer Protocol)**
- **RSYNC:**  
  For details on efficient use of `rsync`, see the dedicated page.
- **GridFTP:**  
  A protocol that enables very efficient data transfers among HPC platforms. A detailed description is provided in the specific document **Globus Online**.

Additional details are available on the **Data Transfer** dedicated page.

---

## Typical Issues

After the initial 2FA setup, you can manage any authentication-related issues via the Identity Provider Website. From there, you can:
- Reset your password.
- Reconfigure OTP on your smartphone.
- Generate new Recovery Authentication codes.

---

## Policy for Password Definition

If you change your password on the portal `sso.hpc.cineca.it`, it will be automatically updated on all clusters (propagation may take up to one hour).

**New Password Policies:**

- **Length & Composition:**  
  The new password must be at least 10 characters long and contain at least one capital letter, one number, and one special character (e.g., `!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~`).

- **Validity:**  
  The password is valid for 3 months. A reminder will be sent 10 days before expiration when you log in.

- **History:**  
  The new password must differ from your previous 5 passwords.

- **Notification:**  
  Any password change will be notified to the user by email.

---

This concludes the Markdown conversion of the provided text. If you need further modifications or additional sections converted, please let me know!