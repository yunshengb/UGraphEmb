# UGraphEmb

This repository is the implementation of "Unsupervised Inductive Graph-Level Representation Learning via
Graph-Graph Proximity".

Yunsheng Bai, Hao Ding, Yang Qiao, Agustin Marinovic, Ken Gu, Ting Chen, Yizhou Sun, and Wei Wang. 2019. Unsupervised Inductive Graph-Level Representation Learning via Graph-Graph Proximity. IJCAI (2019).

## Data Files

Download the data files from https://drive.google.com/drive/folders/1XWkQmrqQxwsZ9KvCZwcOyDgAVPfbDBM3?usp=sharing

Extract under `/data` and `/save`.

## Dependencies

Install the following the tools and packages:

* `python3`: Assume `python3` by default (use `pip3` to install packages).
* `numpy`
* `pandas`
* `scipy`
* `scikit-learn`
* `tensorflow` (1.8.0 recommended)
* `networkx==1.10` (NOT `2.1`)
* `beautifulsoup4`
* `lxml`
* `matplotlib`
* `seaborn`
* `colour`
* `pytz`
* `requests`
* `klepto`

Reference commands:
```sudo pip3 install numpy pandas scipy scikit-learn tensorflow networkx==1.10 beautifulsoup4 lxml matplotlib seaborn colour pytz requests klepto```
    
## Tips for PyCharm Users

* If you see red lines under `import`, mark `src` and `model/Siamese` as `Source Root`,
so that PyCharm can find those files.
* Mark `src/Siamese/logs` as `Excluded`, so that PyCharm won't spend time inspecting those logs.

## Run

Example commands to run our model:

```
cd model/Siamese
python3.5 run.py 
```
