import os
import torch


def save_checkpoint(state, save_dir, is_best=False, filename="checkpoint.pth"):
    os.makedirs(save_dir, exist_ok=True)
    checkpoint_path = os.path.join(save_dir, filename)
    torch.save(state, checkpoint_path)

    if is_best:
        best_path = os.path.join(save_dir, "best_model.pth")
        torch.save(state, best_path)


def load_checkpoint(checkpoint_path, student_model, teacher_vit, teacher_cnn, optimizer=None):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    student_model.load_state_dict(checkpoint['student_model'])
    teacher_vit.load_state_dict(checkpoint['teacher_vit'])
    teacher_cnn.load_state_dict(checkpoint['teacher_cnn'])

    if optimizer and 'optimizer' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])

    print(f"✅ Loaded checkpoint from {checkpoint_path} (epoch {checkpoint['epoch']})")
    return checkpoint.get("epoch", 0), checkpoint.get("best_score", None)
