import torch
import torchvision


def fuse(conv, bn):
    fused = torch.nn.Conv2d(
        conv.in_channels,
        conv.out_channels,
        kernel_size=conv.kernel_size,
        stride=conv.stride,
        padding=conv.padding,
        bias=True,
    )

    # setting weights
    w_conv = conv.weight.clone().view(conv.out_channels, -1)
    w_bn = torch.diag(bn.weight.div(torch.sqrt(bn.eps + bn.running_var)))
    fused.weight.copy_(torch.mm(w_bn, w_conv).view(fused.weight.size()))

    # setting bias
    if conv.bias is not None:
        b_conv = conv.bias
    else:
        b_conv = torch.zeros(conv.weight.size(0))
    b_bn = bn.bias - bn.weight.mul(bn.running_mean).div(torch.sqrt(bn.running_var + bn.eps))
    fused.bias.copy_(b_conv + b_bn)

    return fused


# Testing
# we need to turn off gradient calculation because we didn't write it
torch.set_grad_enabled(False)
x = torch.randn(16, 3, 256, 256)
resnet18 = torchvision.models.resnet18(pretrained=True)
# removing all learning variables, etc
resnet18.eval()
model = torch.nn.Sequential(resnet18.conv1, resnet18.bn1)
f1 = model.forward(x)
fused = fuse(model[0], model[1])
f2 = fused.forward(x)
d = (f1 - f2).mean().item()
print("error:", d)
