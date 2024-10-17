# Addressing Shortcomings in Fair Graph Learning Datasets: Towards a New Benchmark
The official repository for the paper "[Addressing Shortcomings in Fair Graph Learning Datasets: Towards a New Benchmark](https://arxiv.org/abs/2403.06017)" *（KDD'24 ADS）*.

## Installation

### If you want to recreate the original environment used for the paper:

run the installation script (*a clearer version will come soon*)

```shell
conda env create -f environment.yml
```

### Otherwise, for a new env:

we use:  `Python=3.7.9`, `dgl-cu102==0.4.3`,  `torch==1.6.0`

## Datasets

In this paper, we develop and introduce a collection of synthetic, semi-synthetic, and real-world datasets. You can find these datasets in the `dataset` folder.

### Synthetic dataset

Based on the analysis framework in this paper, you can adjust the bias level in synthetic data by setting parameters in `synthetic_config.yaml`. Also, you can save or load the existing synthetic datasets by the code in `load_data.py`.

### Semi-synthetic dataset

Through the functions `add_edges` and `remove_edges` in `utils.py`, we obtain three new semi-synthetic datasets named `germanA`, `creditA`, and `bailA`. Following the analysis framework, You can modify other datasets to achieve the desired bias level.

### Real-world dataset

Our real-world datasets both originate the social data from [Twitter](https://developer.twitter.com/en).

Because the size is limited, download them from [Google Drive](https://drive.google.com/drive/folders/1MRjSz7Uxs9U95mqhQZJEB9oUdWChZgr_?usp=sharing).

We provide some statistics of our datasets in the table below:

| Dataset             | Syn-1  | Syn-2  |      New German      |      New Bail      |         New Credit         |       Sport        |      Occupation      |
| :------------------ | :----: | :----: | :------------------: | :----------------: | :------------------------: | :----------------: | :------------------: |
| \# of nodes         | 5,000  | 5,000  |        1,000         |       18,876       |           30,000           |       3,508        |        6,951         |
| \# of edges         | 34,363 | 44,949 |        20,242        |      31,5870       |         1,121,858          |      136,427       |        44,166        |
| \# of features      |   48   |   48   |          27          |         18         |             13             |        768         |         768          |
| Sensitive attribute |  0/1   |  0/1   | Gender (Male/Female) | Race (Black/White) |     Age ($<$25/$>$25)      | Race (White/Black) | Gender (Male/Female) |
| Label               |  0/1   |  0/1   |   Good/bad Credit    |    Bail/no bail    | Payment default/no default |      NBA/MLB       |        Psy/CS        |
| Average degree      | 13.75  | 17.98  |        41.48         |       34.47        |           75.79            |       78.78        |        13.71         |

More details on our datasets can be found in the paper.

## Running the experiments

To reproduce the experiments, the main scripts running the experiments are in the `script` folder. For example, you can train GCN among all datasets by typing:

```shell
bash ./script/gcn.sh
```

Certainly, You can change the parameter search space or modify some commands to implement multi-threaded training.

## Citation

Please cite our paper if you found our datasets or code helpful.

```latex
@inproceedings{qian2024addressing,
  title={Addressing Shortcomings in Fair Graph Learning Datasets: Towards a New Benchmark},
  author={Qian, Xiaowei and Guo, Zhimeng and Li, Jialiang and Mao, Haitao and Li, Bingheng and Wang, Suhang and Ma, Yao},
  booktitle={Proceedings of the 30th ACM SIGKDD Conference on Knowledge Discovery and Data Mining},
  pages={5602--5612},
  year={2024}
}
```

