# FedMAD
Federated Learning with Multi-head Attention Distillation 

suncheol heo ([suncheolheo@yuhs.ac])

## Federated distillation with multi communication rounds
```python
python main.py --alpha 1.0 --seed 1 --C 1
```
## Oneshot federated distillation
```python
python main.py --alpha 1.0 --seed 1 --C 1 --oneshot
```
## Oneshot federated distillation with quantization and noise
```python
python main.py --alpha 1.0 --seed 1 --C 1 --oneshot --noisescale 1.0 --quantify 100
python main.py --alpha 0.1 --seed 1 --C 1 --oneshot --noisescale 1.0 --quantify 100
python main.py --alpha 1.0 --seed 1 --C 1 --dataset cifar100 --oneshot --noisescale 1.0 --quantify 100
python main.py --alpha 0.1 --seed 1 --C 1 --dataset cifar100 --oneshot --noisescale 1.0 --quantify 100
```