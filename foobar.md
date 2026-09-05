| Benchmark | pallas (us) | mosaic (us) | xla (us) |
|---|---|---|---|
| alphafold_alphafold_384res_64chan_forward | 30.0 | **29.0** | 31.2 |
| alphafold_alphafold_384res_128chan_forward | 59.5 | **57.5** | 59.6 |
| alphafold_alphafold_768res_128chan_forward | 223.0 | **221.0** | 223.0 |
| alphafold_alphafold_384res_128chan_axis0_forward | **66.6** | 69.1 | 143.0 |
| alphafold_alphafold_768res_128chan_axis0_forward | **239.0** | 419.0 | 495.0 |
| alphafold_alphafold_384res_64chan_forward_and_vjp | **85.6** | 86.5 | 187.0 |
| alphafold_alphafold_384res_128chan_forward_and_vjp | 175.0 | **174.0** | 367.0 |
| alphafold_alphafold_768res_128chan_forward_and_vjp | **614.0** | 615.0 | 1352.0 |
| alphafold_alphafold_384res_128chan_axis0_forward_and_vjp | **186.0** | 191.0 | 409.0 |
| alphafold_alphafold_768res_128chan_axis0_forward_and_vjp | **668.0** | 847.0 | 1471.0 |
