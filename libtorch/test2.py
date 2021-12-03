import torch


class Mymodel(torch.nn.Module):
    def __init__(self):
        super(Mymodel, self).__init__()
        self.conv = torch.nn.Conv2d(3, 2, 2)

    def forward(self, x):
        out = self.conv(x)
        return x


tmp_in = torch.rand(1, 3, 5, 5)
m = Mymodel()
script_model = torch.jit.trace(m, tmp_in)
script_model.save('src/model.pt')

# use traced model
# out = script_model(some_inputs)