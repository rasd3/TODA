<div align='center'>

# Smi-Supervised Domain Adaptation Using Target-Oriented Domain Augmentation for 3D Object Detection

IEEE Transaction on Inteeligent Vehicles


[Yecheol Kim](https://rasd3.github.io)<sup>1*</sup>&nbsp;&nbsp;
Junho Lee<sup>1*</sup>&nbsp;&nbsp;
Changsoo Park<sup>2</sup> &nbsp;&nbsp;
Hyung won Kim<sup>2</sup>&nbsp;&nbsp;
Inho Lim<sup>2</sup>&nbsp;&nbsp;
Christopher Chang<sup>2</sup>&nbsp;&nbsp;
Jun Won Choi<sup>3</sup>

<div>
<sup>1</sup> Hanyang University
<sup>2</sup> Kakao Mobility Corp
<sup>3</sup> Seoul National Univeristy
</div>
<br/>

[![arXiv](https://img.shields.io/badge/arXiv-2406.11313-darkred)](https://arxiv.org/abs/2406.11313)

</div>



We prospose novel two-stage SSDA framework for 3D object detection TODA. TODA achieves SOTA on Waymo to nuScenes domain adaptation benchmarks, attains performances on par with the *Oracle* performance utilizing merely **5% of labeled data** in the target domain. 

<div align='center'>
<img src="./fig/comp.png" alt="Compressed Image" width="600">
</div>


## Main Results
### Waymo to nuScenes
We utilizes 100% of Waymo annotations along with partial nuScenes annotations. For nuScenes, we uniformly downsample the training samples to 0.1%, 1%, 5%, and 10% (resulting in 28, 282, 1,407, 2,813, and frames respectively), while the remaining samples are left unlabeled.
| Methods | 0.1%| 0.5%| 1% | 5% | 10% | 
| ------- | -- |  -- | --| -- | --- | 
| Labeled Target | fail | 36.0 / 37.7 | 37.2 / 38.1 | 61.0 / 53.2 | 65.6 / 58.2 | 
| SSDA3D | 62.0 / 57.4 |70.3 / 65.1 |73.4 / 67.1 | 76.2 / 68.8 | 78.8 / 70.9 | 
| Ours  | 69.7 / 63.6 |73.7 / 67.3 | 75.6 / 68.5 | 79.0 / 71.1 | 78.8 / 70.9 | 
| Oracle  | 78.4 / 69.9 |78.4 / 69.9 |78.4 / 69.9 | 78.4 / 69.9 | 78.4 / 69.9 |

## Citation
If you find this work or code useful, please cite 

```
@article{kim2024semi,
  title={Semi-Supervised Domain Adaptation Using Target-Oriented Domain Augmentation for 3D Object Detection},
  author={Kim, Yecheol and Lee, Junho and Park, Changsoo and won Kim, Hyoung and Lim, Inho and Chang, Christopher and Choi, Jun Won},
  journal={IEEE Transactions on Intelligent Vehicles},
  year={2024},
  publisher={IEEE}
}
```
e
