Dear Fredrik Heintz,


thank you for your application EHPC-DEV-2025D02-101. On behalf of CINECA, I'm happy to inform you that the amount of resources that will be allocated to your project on our clusters are:

144.000 local core hours on Leonardo Booster (CINECA, Italy).

Or:

4.500 node hours

for a period of 12 months. Your project will start on 24/03/2025 and will end on 24/03/2026.

Storage

Note that as regards the storage resources, the default value of disk space on the work file system is 1 TB. If you need more disk space, please ask to CINECA User Support (superc@cineca.it) to increase your quota up to the amount granted in the allocation awarded by EuroHPC once the project is started. If you need also some space on the archive storage, please remember that it is not active by default and it must be requested sending an email to the CINECA User Support team.


Data Transfer
Please remember that the data transfer procedure is the full and complete responsibility of the users – even using GridFTP (the tool we recommend), moving huge amounts of data (i.e. > 20 TB/year) could be challenging. The transfer rate depends on many factors (e.g., the bandwidth of the endpoints, the number of users that are using the network,…) and CINECA cannot be considered responsible for unpredictable issues in moving the data. We strongly suggest starting to move the data as soon as they are produced, especially if the amount of data to move is significant.


Important remarks
Please remember that the installation and the optimization of the codes to be used in the project must be done by the applicants. Some community codes are already installed as modules on Leonardo, but CINECA cannot guarantee that they are the most suitable ones for your simulations. If you need assistance in installing your own code, you can ask our User Support team (superc@cineca.it). In any case, you cannot delegate this task only to the support team since it is not included in the standard EuroHPC agreement. Installed modules will be available following the procedures described in Leonardo’s User Guide.

Pay attention especially to the section named: “Budget linearization policy”_since on Leonardo for each month your monthly budget will be calculated using the formula: total budget for your project/number of months of the project. As long as you do not finish your monthly quota, you have a higher priority on our queue. If you exceed your monthly quota, your priority decreases. Clearly at the end of the month the remaining budget will be still available, but with lower priority.
Please consider also the accounting procedure at CINECA. Your budget is calculated in local core hours. Please consider the following example to clarify our policy:

If you are asking for 1 node for 1 hour (with or without using the GPUs), you are consuming 32 local core hours, since each node is equipped with 32 cores;

If you are requesting ½ of the total available memory of 1 node (~240 GB) and 1 core for 1 hour, you are still consuming the resources of ½ a node, i.e. 16 core hours.

For the same reasons, if you are requesting 10 GB of RAM and 16 cores (of 32) for 1 hour, you are consuming the resources of ½ a node, i.e. 16 core hours.

Consistent with the above, each GPU is equivalent to ¼ a node.

Remember that you are responsible for the correct utilization of your budget. You can check your monthly quota and your total and monthly consumption using the "saldo -b" command (type "saldo --help" on the Leonardo shell to obtain more information). If, at the end of the allocation, a large amount of unused resources is found, this information will be notified to the EuroHPC peer review office. The requests of budget extensions must be sent directly to EuroHPC (access@eurohpc-ju.europa.eu).
Furthermore, please consider that although CINECA does its best to prevent possible technical issues, on a large and complex HPC supercomputer such as Leonardo, operational problems with the system may occur.

Registration
If you are already registered in our UserDB database (https://userdb.hpc.cineca.it/) your account will be ready in few days and you will have only to add your collaborators to the project. If you are not yet registered in the UserDB, in order to have access on Leonardo, it is necessary to complete the registration procedure on the CINECA UserDB portal: https://userdb.hpc.cineca.it/ . Detailed instructions are reported in our User Guide: Get Start
After the registration, please contact eurohpc-tech@cineca.it in order to add the budget of your EuroHPC project to your CINECA account.
Please remember that in order to obtain a valid username on Leonardo, it is necessary to upload a digital copy of your valid national ID card (or passport) on the userDB portal.
Please remember also that your account is strictly reserved to you. No other person can use your username and password. For this reason, we kindly ask you to complete your registration personally.

IMPORTANT:

we are happy to announce that we have increased the security on our cluster introducing a two-factor authentication (2FA) method to access Leonardo.

The 2FA adds a further level of security to the authentication for access to services based on the Identity Provider.

A one-time configuration on the Identity Provider, and the installation of the smallstep client and of a One-Time Password (OTP) authenticator are required.

Shortly, before connecting to the cluster you have to request the ssh certificate to our Identity Provider (IP) via the smallstep client.

A web page will be automatically opened on the browser and you will be asked to authenticate to our IP by inserting an OTP.

Once the authentication has taken place, the server will issue a time-limited certificate valid for 12 hours through which you can connect to Cineca systems via SSH client (just ssh to the cluster login). After 12 hours a new certificate needs to be generated.

Please, refer to the on-line documentation to configure the OTP and install/configure the smallstep client on your local PC: Access to the system


Your project on Leonardo will start at your official start date or as soon as you complete the registration. In both cases the project will end 12 months after the original start date.
Therefore, please keep in mind that, the later you register, the harder it will be to consume the whole budget granted to your project.
In order to facilitate the registration please follow the instructions attached to this document.


Userguide
You can find the user guide for Leonardo here: Leonardo User Guide


Get in touch
In order to be automatically informed on any news about our HPC infrastructure (including maintenance operations, shutdowns, hardware and software issues), registering on our mailing list service (http://www.hpc.cineca.it/content/stay-tuned ) is mandatory.

Please do not hesitate to contact us in case you need any assistance.



Best Regards,

EuroHPC Support Team @CINECA