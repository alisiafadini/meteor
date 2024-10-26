
# Benchmark Dataset for Meteor Scripts #
Produced: 22 OCT 2024
---------------------

## Magic Numbers
- Full commit hash: 0b6acdb4f62070adb8c792512502485e8730a772
- k_parameter: 0.05
- map_sampling: 3.0
- lambda_values_to_scan: np.linspace(0, 0.1, 100)
- tv_weights_to_scan: [0.01, 0.05, 0.1]

## Folder Contents

1. 8a6g.pdb
  - Deposited fluorescent OFF state PDB for Cl-rsEGFP2.

2. 8a6g_nochrom.pdb
  - Deposited fluorescent OFF state PDB for Cl-rsEGFP2 without chromophore atoms. 
    This is used for calculating model structure factors.

3. scaled-test-data.mtz
  - F_off, SIGF_off: Dark unscaled structure factors and uncertainties.
  - F_on, SIGF_on: Light unscaled structure factors and uncertainties.
  - FC_nochrom, PHIC_nochrom: Amplitudes and phases calculated from 8a6g_nochrom.pdb.
  - F_off_scaled, SIGF_off_scaled: Output of scaling F_off to FC_nochrom.
  - F_on_scaled, SIGF_on_scaled: Output of scaling F_on to FC_nochrom.
  - F_k, SIGF_k, PHI_k: k-weighted difference map output.
  - F_TV, SIGF_TV, PHI_TV: Single-pass TV denoise difference map output.
  - F_itTV, SIGF_itTV, PHI_itTV: Iterative TV difference map output.

## Commands

1. F_k, SIGF_k, PHI_k
```bash
python compute_difference_map.py   --native_mtz scaled-test-data.mtz F_off SIGF_off PHIC_nochrom   --derivative_mtz scaled-test-data.mtz F_on SIGF_on PHIC_nochrom   --calc_native_mtz scaled-test-data.mtz FC_nochrom PHIC_nochrom   --k_weight_with_fixed_parameter 0.05
```

2. F_TV, SIGF_TV, PHI_TV
```bash
python compute_tvdenoised_difference_map.py   --native_mtz scaled-test-data.mtz F_off SIGF_off PHIC_nochrom   --derivative_mtz scaled-test-data.mtz F_on SIGF_on PHIC_nochrom   --calc_native_mtz scaled-test-data.mtz FC_nochrom PHIC_nochrom   --k_weight_with_fixed_parameter 0.05
```

3. F_itTV, SIGF_itTV, PHI_itTV
```bash
python compute_iterativetv_difference_map.py   --native_mtz scaled-test-data.mtz F_off SIGF_off PHIC_nochrom   --derivative_mtz scaled-test-data.mtz F_on SIGF_on PHIC_nochrom   --calc_native_mtz scaled-test-data.mtz FC_nochrom PHIC_nochrom   --k_weight_with_fixed_parameter 0.05
```
