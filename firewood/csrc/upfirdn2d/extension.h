#pragma once
#include <torch/extension.h>

torch::Tensor upfirdn2d(torch::Tensor x,
                        torch::Tensor f,
                        int upx,
                        int upy,
                        int downx,
                        int downy,
                        int padx0,
                        int padx1,
                        int pady0,
                        int pady1,
                        bool flip,
                        float gain);
