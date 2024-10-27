import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiTaskMLP(nn.Module):
    def __init__(self, batch_size,input_size, hidden_size):
        super(MultiTaskMLP, self).__init__()

        self.layer1 = SingleMLP(batch_size,input_size, hidden_size)
        self.layer2 = SingleMLP(batch_size,input_size, hidden_size)
        self.layer3 = SingleMLP(batch_size,input_size, hidden_size)

        self.output_logit_map1 = torch.nn.Linear(hidden_size, 1)
        torch.nn.init.normal_(self.output_logit_map1.weight, 0, 0.01)
        torch.nn.init.normal_(self.output_logit_map1.bias, 0, 0.01)

        self.output_logit_map2 = torch.nn.Linear(hidden_size, 1)
        torch.nn.init.normal_(self.output_logit_map2.weight, 0, 0.01)
        torch.nn.init.normal_(self.output_logit_map2.bias, 0, 0.01)

        self.output_logit_map3 = torch.nn.Linear(hidden_size, 1)
        torch.nn.init.normal_(self.output_logit_map3.weight, 0, 0.01)
        torch.nn.init.normal_(self.output_logit_map3.bias, 0, 0.01)


    def forward(self, x):
        eye_movement_output = self.layer1(x)
        cognitive_engagement_output = self.layer2(x)
        emotion_recognition_output = self.layer3(x)

        eye_movement_output = self.output_logit_map1(eye_movement_output)
        cognitive_engagement_output = self.output_logit_map2(cognitive_engagement_output)
        emotion_recognition_output = self.output_logit_map3(emotion_recognition_output)

        return eye_movement_output, cognitive_engagement_output, emotion_recognition_output
     #   return save
class SingleMLP(nn.Module):
    def __init__(self, batch_size,input_size, hidden_size):
        super(SingleMLP, self).__init__()

        self.output_logit_map = torch.nn.Linear(hidden_size, 1)
        torch.nn.init.normal_(self.output_logit_map.weight, 0, 0.01)
        torch.nn.init.normal_(self.output_logit_map.bias, 0, 0.01)

        self.first_map = torch.nn.Linear(input_size, hidden_size)
        torch.nn.init.normal_(self.first_map.weight, 0, 0.01)
        torch.nn.init.normal_(self.first_map.bias, 0, 0.01)
        self.layer2 = LinearLayer(hidden_size, 2048, hidden_size, batch_size, [True, True])

    def forward(self, x):
        """Naive full forward pass."""
        x = torch.sigmoid(x)

        #x = self.layer3(x)

        x = self.first_map(x)
        temp = x
        x = self.layer2(x)

        save = (x + temp) / 2
        # x = self.output_logit_map(save)

        return save


class dense_baens(nn.Module):
    def __init__(self, N, B, D1, D2):
        super(dense_baens, self).__init__()

        self.batch_size = N
        self.B = B
        self.D1 = D1
        self.D2 = D2
        self.U = nn.Parameter(torch.normal(0, 0.01, (self.batch_size, D1, D2)), requires_grad=True)
        self.bias = nn.Parameter(torch.normal(0, 0.01, (self.batch_size, B, D2)), requires_grad=True)

    def forward(self, x):
        # 增加 x 的一个维度
        x = x.unsqueeze(1)  # x 的形状现在是 (batch_size, 1, D1)
        act = torch.bmm(x, self.U)
        act += self.bias
        # 移除之前增加的虚拟维度，如果需要
        act = act.squeeze(1)
        return act


class BELayer(torch.nn.Module):

    def __init__(self, branch,batch_size, input_size, hidden_size):
        super(BELayer, self).__init__()
        self.branch = branch

        self.V_map = dense_baens(batch_size, branch, input_size, hidden_size)
        self.layernorm1 = torch.nn.LayerNorm(input_size, eps=1e-05, elementwise_affine=True)
        self.layernorm2 = torch.nn.LayerNorm(hidden_size, eps=1e-05, elementwise_affine=True)

    def forward(self, x):
        x = self.layernorm1(x)

        x = self.V_map(x)

        x = self.layernorm2(x)
        x = torch.nn.functional.gelu(x)

        return x


class LinearLayer(torch.nn.Module):

    def __init__(self, hidden_dim, ffn_dim, out_dim, n_heads, ea=[True, True]):
        super(LinearLayer, self).__init__()

        self.U_map = torch.nn.Linear(hidden_dim, ffn_dim, bias=True)
        torch.nn.init.normal_(self.U_map.weight, 0, 0.01)
        torch.nn.init.normal_(self.U_map.bias, 0, 0.01)

        self.V_map = torch.nn.Linear(ffn_dim, out_dim, bias=True)
        torch.nn.init.normal_(self.V_map.weight, 0, 0.01)
        torch.nn.init.normal_(self.V_map.bias, 0, 0.01)

        self.layernorm1 = torch.nn.LayerNorm(hidden_dim, eps=1e-05, elementwise_affine=ea[0])
        self.layernorm2 = torch.nn.LayerNorm(out_dim, eps=1e-05, elementwise_affine=ea[1])
        self.ln1 = 1

    def forward(self, x):
        x = self._ffn(x)

        return x

    def _ffn(self, x):
        if self.ln1:
            x = self.layernorm1(x)

        skip = x

        x = self.U_map(x)
        x = torch.nn.functional.gelu(x)
        x = self.V_map(x)

        x = self.layernorm2(x)
        x = torch.nn.functional.gelu(x)

        x = (skip + x) / 2

        return x
