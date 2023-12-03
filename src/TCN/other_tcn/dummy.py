import torch.nn as nn
 
# dummy prediction model (Y_t= Y_{t-1})
class Dummy(nn.Module):
 
    def __init__(self):
        super(Dummy, self).__init__()
 
    def forward(self, x):# x(x_test):[n_test,4,features]
        return x[:, 0, -1].unsqueeze(1)# 返回值形状：[n_test,1]