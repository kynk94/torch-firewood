#pragma once
#include <torch/extension.h>

torch::Tensor bias_act(torch::Tensor x,
                       torch::Tensor b,
                       torch::Tensor xref,
                       torch::Tensor yref,
                       torch::Tensor dy,
                       int grad,
                       int dim,
                       int act,
                       float alpha,
                       float gain,
                       float clamp);
