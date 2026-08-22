# Real Shapes: Shape-Aware Init vs Canonical Init

| Shape | Pred Class | 500p plain | 500p shape | Δ500 | 100p plain | 100p shape | Δ100 |
|-------|-----------|-----------|-----------|------|-----------|-----------|------|
| bunny | ico_g1_CC2_264v | 0.9155 | 0.8838 | -0.0317 | 0.7412 | 0.7360 | -0.0052 |
| torus | ico_g1_CC_66v | 0.9796 | 0.9937 | +0.0141 | 0.8993 | 0.9813 | +0.0821 |
| stretched_bar | ico_g0_CCDS_240v | 0.9532 | 0.9866 | +0.0334 | 0.5260 | 0.6550 | +0.1290 |
| flat_plate | ico_g0_CCDS_240v | 0.9719 | 0.9896 | +0.0177 | 0.4757 | 0.9034 | +0.4277 |

- **bunny**: pred_shape=['1.50', '1.19', '1.17'], gt_extents=['1.60', '1.60', '1.25']
- **torus**: pred_shape=['1.59', '0.73', '1.54'], gt_extents=['1.60', '0.46', '1.60']
- **stretched_bar**: pred_shape=['1.68', '0.78', '0.69'], gt_extents=['1.60', '0.53', '0.27']
- **flat_plate**: pred_shape=['1.40', '0.43', '1.58'], gt_extents=['1.60', '0.24', '1.60']
