# ContractCheck



**ACCEPTED BY IEEE TRANSACTIONS ON SOFTWARE ENGINEERING**

---

## Project Description

This project aims to provide fine-grained vulnerabilities detection on smart contracts, including Integer Overflow/Underflow, Denial of Service, Authorization through tx.origin, and Time Manipulation (Block values as a proxy for time).

## Quick Start

Step 1: Pull the Docker Image
To start, you will need to pull the latest version of the contractcheck image from the Docker registry. This can be done with the following command:
bash

```
docker pull hitew/contractcheck:2.0
```

Step 2: Run the Docker Container
After successfully pulling the image, you can run a Docker container using the following command:

```
docker run -v /path/to/:/app/input.sol contractcheck:2.0
```

Replace /path/to/ with the path to the directory on your host system that contains the smart contract files (.sol) you wish to check.
