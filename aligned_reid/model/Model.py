import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

from .resnet import resnet50


class GradReverse(torch.autograd.Function):
    def forward(self, x):
        return x.view_as(x)

    def backward(self, grad_output):
        return torch.neg(grad_output)

def grad_reverse(x):
    return GradReverse()(x)

class domain_classifier(nn.Module):
    def __init__(self, channel, cam_classes):
        super(domain_classifier, self).__init__()
        self.fc1 = nn.Linear(channel, 512) 
        self.fc2 = nn.Linear(512, cam_classes)
        self.drop = nn.Dropout2d(0.5)

    def forward(self, x):
        x = grad_reverse(x)
        x = F.leaky_relu(self.drop(self.fc1(x)))
        x = self.fc2(x)
        return torch.sigmoid(x)

class Model(nn.Module):
  def __init__(self, local_conv_out_channels=128, num_classes=None, cam_classes=None):
    super(Model, self).__init__()
    self.base = resnet50(pretrained=True)
    planes = 2048
    self.local_conv = nn.Conv2d(planes, local_conv_out_channels, 1)
    self.local_bn = nn.BatchNorm2d(local_conv_out_channels)
    self.local_relu = nn.ReLU(inplace=True)
    #self.view_classifier = domain_classifier(2048)
    self.bn = nn.BatchNorm1d(planes)

    if num_classes is not None:
      self.fc = nn.Linear(planes, num_classes)
      init.normal_(self.fc.weight, std=0.001)
      init.constant_(self.fc.bias, 0)
      
    if cam_classes is not None:
      self.view_classifier = domain_classifier(2048, cam_classes)

  def forward(self, x):
    """
    Returns:
      global_feat: shape [N, C]
      local_feat: shape [N, H, c]
    """
    # shape [N, C, H, W]
    feat = self.base(x)
    global_feat = F.avg_pool2d(feat, feat.size()[2:])
    # shape [N, C]
    global_feat = global_feat.view(global_feat.size(0), -1)

    local_feat = self.bn(global_feat)
    # shape [N, C, H, 1]
    # local_feat = torch.mean(feat, -1, keepdim=True)
    # local_feat = self.local_relu(self.local_bn(self.local_conv(local_feat)))
    # # shape [N, H, c]
    # local_feat = local_feat.squeeze(-1).permute(0, 2, 1)

    if hasattr(self, 'fc'):
      logits = self.fc(local_feat)
      if hasattr(self, 'view_classifier'):
        view_logits = self.view_classifier(local_feat)
        return feat, global_feat, local_feat, logits, view_logits        
      return feat, global_feat, local_feat, logits

    return feat, global_feat, local_feat
