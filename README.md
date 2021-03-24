# Quantum Robust Classifiers
This repository includes the code in our paper "[Quantum noise protects quantum classifiers against adversaries](https://arxiv.org/abs/2003.09416)". 

---

## Requirements
```
pennylane==0.11.0
pennylane-qiskit==0.11.0
qiskit==0.20.1
```

## Evaluation
* Quantum Machine Learning
  ```shell
  python bench-QC.py    # train a quantum classifier without any denfending technique
  python robust-QC.py --noise    # train a robust quantum classifier
  ```

## Attacking method
Please refer to the study "[Quantum adversarial machine learning](https://journals.aps.org/prresearch/pdf/10.1103/PhysRevResearch.2.033212)" to realize different attacking methods. Note that the attached codes can be easily integrated with the employed attacking methods to investigate the robustness of quantum classifiers.  
---

## Citation
If you find our code useful for your research, please consider citing it:
```
@article{du2020quantum,
  title={Quantum noise protects quantum classifiers against adversaries},
  author={Du, Yuxuan and Hsieh, Min-Hsiu and Liu, Tongliang and Tao, Dacheng and Liu, Nana},
  journal={arXiv preprint arXiv:2003.09416},
  year={2020}
}
```
