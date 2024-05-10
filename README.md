# ContractCheck

## Authors

**Submitted to IEEE TRANSACTIONS ON SOFTWARE ENGINEERING**

---

## Project Description

This project aims to provide fine-grained vulnerabilities detection on smart contracts, including Integer Overflow/Underflow, Denial of Service, Authorization through tx.origin, and Time Manipulation (Block values as a proxy for time).

## Requirements

```shell
pip install -r requirements.txt
```
## Running Project
This folder contains the code for the ContrackCheck method. There are a few important subfolders and files as follows.

- AST_Parsed - Uses ANTLR to parse abstract syntax trees.
- AFU_Extracted - Extracts AFU code slices.
- Embedded - Embeds code slices.
- Method - Performs vulnerability detection using different methods.

## References
Checking Smart Contracts With Structural Code Embedding
```shell
@ARTICLE{8979435,
  author={Gao, Zhipeng and Jiang, Lingxiao and Xia, Xin and Lo, David and Grundy, John},
  journal={IEEE Transactions on Software Engineering}, 
  title={Checking Smart Contracts With Structural Code Embedding}, 
  year={2021},
  volume={47},
  number={12},
  pages={2874-2891},
  keywords={Computer bugs;Smart contracts;Cloning;Blockchains;Security;Smart contract;code embedding;clone detection;bug detection;ethereum;blockchain},
  doi={10.1109/TSE.2020.2971482}}
```
SySeVR: A Framework for Using Deep Learning to Detect Software Vulnerabilities
```shell
@ARTICLE{9321538,
  author={Li, Zhen and Zou, Deqing and Xu, Shouhuai and Jin, Hai and Zhu, Yawei and Chen, Zhaoxuan},
  journal={IEEE Transactions on Dependable and Secure Computing}, 
  title={SySeVR: A Framework for Using Deep Learning to Detect Software Vulnerabilities}, 
  year={2022},
  volume={19},
  number={4},
  pages={2244-2258},
  keywords={Deep learning;Syntactics;Software;Semantics;Proposals;Image processing;Big Data;Vulnerability detection;security;deep learning;program analysis;program representation},
  doi={10.1109/TDSC.2021.3051525}}
```
