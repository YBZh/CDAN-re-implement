# CDAN-re-implement
The re-implement of NIPS 2018 paper: Conditional Adversarial Domain Adaptation

The results on the Office31 dataset (Resnet-50):

Method | A->W | D->W | W->D | A->D | D->A | W->A | ACC
-|-|-|-|-|-|-|-
CADA-RM (long) | 93.0±0.2 | 98.4±0.2 | 100.0±.0 |89.2±0.3 | 70.2±0.4 | 69.4±0.4 | 86.7
CADA-M (focal loss, long)  | 93.1±0.1 | 98.6±0.1 | 100.0±.0 | 93.4±0.2 | 71.0±0.3 | 70.3±0.3 | 87.7
CADA-M (no focal loss, long) | 91.7±0.2 | 98.3±0.1 | 100.0±.0 | 92.5±0.2 | 70.0±0.2 | 67.8±0.2 | 86.8 
CADA-M (no focal loss, our)  | 93.3±0.2 | 98.0±0.1 | 100.0±.0 | 90.3±0.3 | 71.7±0.1 | 74.9±0.4 | 88.0
CADA-M (focal loss, our)     | 92.7 | 97.7 | 100 | 90.0|70.7|73.8|87.5
CADA-RM (no focal loss, our) |93.0

For the CADA-M (no focal loss, our), each result is obtained by averaging three random experiments.

For the CADA-M (focal loss, our) and CADA-RM (no focal loss, our), each experiment is only run once.

With our re-implemented focal loss, there is a bit drop on performance. So I don't present the focal loss in the code.
