import torch.nn.functional as F

def supervised_loss(pred, label):
    return F.binary_cross_entropy(pred, label)

def pseudo_label_loss(pred, pseudo):
    return F.mse_loss(pred, pseudo)

def artifact_consistency_loss(pred1, pred2):
    return F.mse_loss(pred1, pred2)

def adversarial_loss(pred, discriminator):
    pred_detach = pred.detach()
    d_out = discriminator(pred_detach)
    return F.binary_cross_entropy(d_out, torch.ones_like(d_out))
