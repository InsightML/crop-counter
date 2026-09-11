| Model | Config | MAE | RMSE | Bias | Precision | Recall | F1 |
|---|---|---|---|---|---|---|---|
| best epoch (val-loss) | NMS off @τ=0.35 | 0.238 | 0.547 | -0.097 | 0.7104 | 0.6005 | 0.6509 |
| best epoch (val-loss) | NMS 0.5 @τ=0.35 | 0.213 | 0.523 | -0.138 | 0.7674 | 0.5995 | 0.6731 |
| last epoch | NMS off @τ=0.35 | 0.251 | 0.544 | -0.069 | 0.7410 | 0.6595 | 0.6979 |
| last epoch | NMS 0.5 @τ=0.35 | 0.231 | 0.520 | -0.098 | 0.7799 | 0.6580 | 0.7138 |

Per-row tau, not one global tau: best epoch (val-loss) τ=0.35 · last epoch τ=0.35 | IoU 0.5 | top-k 100

Best epoch = lowest validation loss (the rule `train.py` writes `best.pt` on), not best AP. Headline checkpoint = **last epoch**: the val-loss selector stopped early while AP / AP50 / AP75 / F1 all kept improving.
