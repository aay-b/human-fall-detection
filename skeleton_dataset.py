import os
import numpy as np
import torch
from torch.utils.data import Dataset

def load_skeleton_file(file_path, num_joints=25):
    """
    Parse a .skeleton file into numpy array of shape (frames, num_joints*3).
    Only takes the first body if multiple are present.
    """
    frames = []
    with open(file_path, "r") as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):
        # number of bodies in this frame
        try:
            num_bodies = int(lines[i].strip())
        except:
            i += 1
            continue
        i += 1

        frame_joints = []
        for b in range(num_bodies):
            i += 1  # skip body info line
            try:
                num_joints_in_body = int(lines[i].strip())
            except:
                break
            i += 1

            joints = []
            for j in range(num_joints_in_body):
                parts = lines[i].strip().split()
                if len(parts) < 3:
                    i += 1
                    continue
                x, y, z = map(float, parts[:3])
                joints.extend([x, y, z])
                i += 1
            if joints:
                frame_joints.append(joints)

        if frame_joints:
            frames.append(frame_joints[0])  # use first body

    return np.array(frames, dtype=np.float32)



class SkeletonDataset(Dataset):
    def __init__(self, root_dir, sequence_length=45):
        self.sequence_length = sequence_length
        self.samples = []

        for label_name in os.listdir(root_dir):
            label_path = os.path.join(root_dir, label_name)
            if not os.path.isdir(label_path):
                continue

            # Map label_name → index based on CLASS_NAMES
            if label_name.lower() == "walk":
                label_id = 0
            elif label_name.lower() == "fall":
                label_id = 1
            else:
                continue

            for file_name in os.listdir(label_path):
                if file_name.endswith(".skeleton"):
                    self.samples.append((os.path.join(label_path, file_name), label_id))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        file_path, label = self.samples[idx]
        data = load_skeleton_file(file_path)  # shape (frames, features)

        # Pad or trim to fixed sequence_length
        if data.shape[0] < self.sequence_length:
            pad_len = self.sequence_length - data.shape[0]
            pad = np.zeros((pad_len, data.shape[1]))
            data = np.vstack((data, pad))
        else:
            data = data[:self.sequence_length, :]

        sample = torch.tensor(data, dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.long)
        return sample, label
