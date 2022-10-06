import mindspore
import numpy as np
from mindspore import context
from my_functionals.SoftmaxCrossEntropyLoss import SoftmaxCrossEntropyLoss

context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
context.set_context(device_id=4)
# input_x = mindspore.Tensor(np.array([[[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]],
#                            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]],
#                            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]]]), mindspore.float32)

# # adaptive_avg_pool_2d = mindspore.ops.AdaptiveAvgPool2D(1)
# # output = adaptive_avg_pool_2d(input_x)
# # print(output)


# pool = mindspore.nn.AvgPool2d(kernel_size=2, stride=1)
# output = pool(input_x)
# print(output)


input1 = mindspore.Tensor(np.random.randn(2,  19, 720, 720).astype("float32"))
input2 = mindspore.Tensor(np.random.randn(2, 1, 720, 720).astype("float32"))
mask = mindspore.Tensor(np.random.randint(0, 19, (2,  720, 720)).astype("int64"))
edge = mindspore.Tensor(np.random.randint(0, 19, (2,  1, 720, 720)).astype("float32"))

loss = SoftmaxCrossEntropyLoss()
out = loss(input1, mask)[0]
print(out)

