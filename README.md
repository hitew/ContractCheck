# ContractCheck

## Authors

**Submitted to IEEE TRANSACTIONS ON SOFTWARE ENGINEERING**

---

## Project Description

This project aims to provide fine-grained vulnerabilities detection on smart contract, including Integer Overflow/Underflow, Denial of Service, Authorization through tx.origin and Time Manipulation(Block values as a proxy for time).

## Docker Usage Guide

### 1. Docker Installation

Firstly, ensure that Docker is installed on your system. If not installed, follow the steps below:

- **Linux:** Install Docker using the package manager suitable for your distribution.
- **Windows and macOS:** Docker Desktop can be downloaded and installed from the Docker official website.

### 2. Pulling the Image

Pull the image from Docker Hub using the following command:

```bash
docker pull hitew/contractcheck:1.0
