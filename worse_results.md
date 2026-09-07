| Benchmark | pallas (us) | mosaic (us) | xla (us) |
|---|---|---|---|
| alphafold_alphafold_384res_64chan_forward | **28.9** | 29.3 | 31.2 |
| alphafold_alphafold_384res_128chan_forward | 59.6 | **58.0** | 59.7 |
| alphafold_alphafold_768res_128chan_forward | 223.0 | **221.0** | 223.0 |
| alphafold_alphafold_384res_128chan_axis0_forward | 66.2 | **63.6** | 143.0 |
| alphafold_alphafold_768res_128chan_axis0_forward | 240.0 | **227.0** | 495.0 |
| alphafold_alphafold_384res_64chan_forward_and_vjp | **84.6** | 99.8 | 187.0 |
| alphafold_alphafold_384res_128chan_forward_and_vjp | 174.0 | **167.0** | 369.0 |
| alphafold_alphafold_768res_128chan_forward_and_vjp | 615.0 | **605.0** | 1347.0 |
| alphafold_alphafold_384res_128chan_axis0_forward_and_vjp | **186.0** | 353.0 | 409.0 |
| alphafold_alphafold_768res_128chan_axis0_forward_and_vjp | **668.0** | 1492.0 | 1471.0 |
