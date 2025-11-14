## Supported Ops and Hardware

We currently support the following GPU kernels:

*   `tokamax.dot_product_attention`
    ([FlashAttention](https://arxiv.org/abs/2205.14135)).
*   `tokamax.gated_linear_unit`
    ([Gated linear units](https://arxiv.org/abs/2002.05202) (SwiGLU etc)).
*   `tokamax.layer_norm`
    ([Layer normalization](https://arxiv.org/abs/1607.06450) and
    [Root Mean Squared normalization](https://arxiv.org/abs/1910.07467)).

And the following for both GPU and TPU:

*   `tokamax.ragged_dot`
    ([Mixture of Experts](https://arxiv.org/abs/2211.15841)).