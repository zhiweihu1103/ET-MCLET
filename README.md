# Multi-view Contrastive Learning for Entity Typing over Knowledge Graph
## Dependencies
* conda create -n mclet python=3.7 -y
* PyTorch 1.8.1
* dgl 0.6.1
* scipy 1.7.3

## Running the code
### Dataset
* Download the datasets from [Here](https://drive.google.com/drive/folders/1YrE1HnAbTVjovY9NzzUnPvPXKqy_xnnD?usp=sharing).
* Create the root directory ./data and put the dataset in.

### Training model
```python
sh FB15kET.sh
sh YAGO43kET.sh
```

* **Note:** Before running, you need to create the ./logs folder first.

## Citation
If you find this code useful, please consider citing the following paper.
```
@article{
  author={Zhiwei Hu and Víctor Gutiérrez-Basulto and Zhiliang Xiang and and Ru Li and Jeff Z. Pan},
  title={Multi-view Contrastive Learning for Entity Typing over Knowledge Graph},
  publisher="The Conference on Empirical Methods in Natural Language Processing",
  year={2023}
}
```
## Acknowledgement
We refer to the code of [CET](https://github.com/CCIIPLab/CET), [TET](https://github.com/zhiweihu1103/ET-TET), and [MiNer](https://github.com/jinzhuoran/MiNer). Thanks for their contributions.
