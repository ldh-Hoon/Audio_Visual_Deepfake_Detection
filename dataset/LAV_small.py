import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import random
import json
from pydub import AudioSegment
import librosa

from var import *

SIZE = 112

class LAVDF_small(Dataset):
    def __init__(self, root_dir,
                 split='train',
                 num_samples=100,
                 target_frames=100,
                 transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.data = []
        self.split = split
        if self.split == 'val':
            self.split = 'dev'
        self.target_frames = target_frames
        metadata_file = os.path.join(root_dir, "metadata.json")
        self.load_metadata(metadata_file, num_samples)

    def load_metadata(self, metadata_file, num_samples):
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)

        video_files = {
            'train': [],
            'dev': [],
            'test': []
        }

        for entry in metadata:
            video_file = entry["file"]
            label = entry["label"]

            split = entry["split"]
            if split in video_files:
                video_files[split].append((label, os.path.join(self.root_dir, video_file)))

        for split in video_files:
            if self.split == split:
                total_count = len(video_files[split])
                print(f"Total {split} videos: {total_count}")

        for split in video_files:
            if self.split == split:
                real_samples = [item for item in video_files[split] if item[0] == 1]
                fake_samples = [item for item in video_files[split] if item[0] == 0]

                target_samples_per_class = num_samples // 2
                sampled_real = random.sample(real_samples, min(target_samples_per_class, len(real_samples)))
                sampled_fake = random.sample(fake_samples, min(target_samples_per_class, len(fake_samples)))

                self.data.extend(sampled_real)
                self.data.extend(sampled_fake)

        self.print_class_distribution()

    def print_class_distribution(self):
        real_count = sum(1 for label, _ in self.data if label == 1)
        fake_count = sum(1 for label, _ in self.data if label == 0)

        total_count = len(self.data)
        print(f"Count for {self.split} set (Total: {total_count}):")
        print(f"Real: {real_count} ({real_count / total_count:.2%}), Fake: {fake_count} ({fake_count / total_count:.2%})")

    def __len__(self):
        return len(self.data)

    def collate_fn(self, batch):
        videos, audio_mfccs, labels = zip(*batch)

        max_video_length = max(video.size(0) for video in videos)
        max_audio_length = max(audio_mfcc.size(0) for audio_mfcc in audio_mfccs)

        # video zero padding
        padded_videos = []
        for video in videos:
            padding_size = max_video_length - video.size(0)
            if padding_size > 0:
                padding = torch.zeros(padding_size, video.size(1), video.size(2), video.size(3))  # (padding_size, C, H, W)
                padded_video = torch.cat((video, padding), dim=0)
            else:
                padded_video = video
            padded_videos.append(padded_video)

        # MFCC zero padding
        padded_audio_mfccs = []
        for audio_mfcc in audio_mfccs:
            padding_size = max_audio_length - audio_mfcc.size(0)
            if padding_size > 0:
                padding = torch.zeros(padding_size, audio_mfcc.size(1))  # (padding_size, N_MFCC)
                padded_audio_mfcc = torch.cat((audio_mfcc, padding), dim=0)
            else:
                padded_audio_mfcc = audio_mfcc
            padded_audio_mfccs.append(padded_audio_mfcc)

        return torch.stack(padded_videos), torch.stack(padded_audio_mfccs), torch.tensor(labels)

    def __getitem__(self, idx):
        label, video_path = self.data[idx]

        video, audio = self.load_video_and_audio(video_path)

        if self.transform:
            video = self.transform(video)

        audio_mfcc = self.mfcc_preprocessing(audio)

        return video, audio_mfcc, label

    def load_video_and_audio(self, video_path):
        cap = cv2.VideoCapture(video_path)
        frames = []
        W = SIZE
        H = SIZE

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.resize(frame, (H, W))
            frames.append(frame)

        cap.release()

        if len(frames) == 0:
            raise ValueError(f"No frames found in video: {video_path}")

        frames = np.array(frames).transpose((0, 3, 1, 2))  # (T, H, W, C) -> (T, C, H, W)

        frames = frames.astype(np.float32) / 255.0
        num_frames = frames.shape[0]
        self.target_frames = num_frames

        audio = self.load_audio(video_path)

        return torch.tensor(frames, dtype=torch.float32), audio

    def load_audio(self, video_path):
        audio_segment = AudioSegment.from_file(video_path)

        video_capture = cv2.VideoCapture(video_path)
        fps = video_capture.get(cv2.CAP_PROP_FPS)
        video_capture.release()

        audio_array = np.array(audio_segment.get_array_of_samples(), dtype=np.float32)

        target_length = 16000 * (self.target_frames / fps)
        target_length = int(target_length)

        if len(audio_array) < target_length:
            padding = np.zeros(target_length - len(audio_array), dtype=np.float32)
            audio_array = np.concatenate((audio_array, padding))
        elif len(audio_array) > target_length:
            audio_array = audio_array[:target_length]

        return audio_array

    def mfcc_preprocessing(self, audio_data, sr=16000, n_mfcc=N_MFCC):
        mfccs = librosa.feature.mfcc(y=audio_data, sr=sr, n_mfcc=n_mfcc)

        target_length = self.target_frames
        num_frames = mfccs.shape[1]

        if num_frames < target_length:
            padding = np.zeros((n_mfcc, target_length - num_frames), dtype=np.float32)
            mfccs = np.concatenate((mfccs, padding), axis=1)  # (n_mfcc, target_length)
        elif num_frames > target_length:
            mfccs = mfccs[:, :target_length]

        return torch.tensor(mfccs.T, dtype=torch.float32)
    


import os
import cv2
import random
import json
import numpy as np

import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.io import read_video

from pydub import AudioSegment
import librosa

class LAVDF_small_torchvision(Dataset):
    def __init__(self, root_dir,
                 split='train',
                 num_samples=100,
                 target_frames=100,
                 transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.data = []
        self.split = split
        if self.split == 'val':
            self.split = 'dev'
        self.target_frames = target_frames
        metadata_file = os.path.join(root_dir, "metadata.json")
        self.load_metadata(metadata_file, num_samples)

    def load_metadata(self, metadata_file, num_samples):
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)

        video_files = {
            'train': [],
            'dev': [],
            'test': []
        }

        for entry in metadata:
            video_file = entry["file"]
            label = entry["label"]

            split = entry["split"]
            if split in video_files:
                video_files[split].append((label, os.path.join(self.root_dir, video_file)))

        for split in video_files:
            if self.split == split:
                total_count = len(video_files[split])
                print(f"Total {split} videos: {total_count}")

        for split in video_files:
            if self.split == split:
                real_samples = [item for item in video_files[split] if item[0] == 1]
                fake_samples = [item for item in video_files[split] if item[0] == 0]

                target_samples_per_class = num_samples // 2
                sampled_real = random.sample(real_samples, min(target_samples_per_class, len(real_samples)))
                sampled_fake = random.sample(fake_samples, min(target_samples_per_class, len(fake_samples)))

                self.data.extend(sampled_real)
                self.data.extend(sampled_fake)

        self.print_class_distribution()

    def print_class_distribution(self):
        real_count = sum(1 for label, _ in self.data if label == 1)
        fake_count = sum(1 for label, _ in self.data if label == 0)

        total_count = len(self.data)
        print(f"Count for {self.split} set (Total: {total_count}):")
        print(f"Real: {real_count} ({real_count / total_count:.2%}), Fake: {fake_count} ({fake_count / total_count:.2%})")

    def __len__(self):
        return len(self.data)

    def collate_fn(self, batch):
        videos, audio_mfccs, labels = zip(*batch)

        max_video_length = max(video.size(0) for video in videos)
        max_audio_length = max(audio_mfcc.size(0) for audio_mfcc in audio_mfccs)

        # 비디오 제로 패딩
        padded_videos = []
        for video in videos:
            padding_size = max_video_length - video.size(0)
            if padding_size > 0:
                padding = torch.zeros(padding_size, video.size(1), video.size(2), video.size(3))  # (padding_size, C, H, W)
                padded_video = torch.cat((video, padding), dim=0)
            else:
                padded_video = video
            padded_videos.append(padded_video)

        # 오디오 MFCC 제로 패딩
        padded_audio_mfccs = []
        for audio_mfcc in audio_mfccs:
            padding_size = max_audio_length - audio_mfcc.size(0)
            if padding_size > 0:
                padding = torch.zeros(padding_size, audio_mfcc.size(1))  # (padding_size, N_MFCC)
                padded_audio_mfcc = torch.cat((audio_mfcc, padding), dim=0)
            else:
                padded_audio_mfcc = audio_mfcc
            padded_audio_mfccs.append(padded_audio_mfcc)

        return torch.stack(padded_videos), torch.stack(padded_audio_mfccs), torch.tensor(labels)

    def __getitem__(self, idx):
        label, video_path = self.data[idx]

        video, audio = self.load_video_and_audio(video_path)

        if self.transform:
            video = self.transform(video)

        audio_mfcc = self.mfcc_preprocessing(audio)

        return video, audio_mfcc, label

    def load_video_and_audio(self, video_path):
        video, audio, info = read_video(video_path, pts_unit='sec')

        if video.size(0) == 0:
            raise ValueError(f"No frames found in video: {video_path}")

        transform = transforms.Compose([
            transforms.Resize((SIZE, SIZE)),
            #transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]) 
        ])

        frames = video.permute(0, 3, 1, 2)
        frames = frames.to(torch.float32) / 255.0
        frames = transform(frames)

        num_frames = frames.shape[0]
        self.target_frames = num_frames

        audio = self.load_audio(video_path)

        return frames, audio

    def load_audio(self, video_path):
        audio_segment = AudioSegment.from_file(video_path)

        video_capture = cv2.VideoCapture(video_path)
        fps = video_capture.get(cv2.CAP_PROP_FPS)
        video_capture.release()

        audio_array = np.array(audio_segment.get_array_of_samples(), dtype=np.float32)

        target_length = 16000 * (self.target_frames / fps)
        target_length = int(target_length)

        if len(audio_array) < target_length:
            padding = np.zeros(target_length - len(audio_array), dtype=np.float32)
            audio_array = np.concatenate((audio_array, padding))
        elif len(audio_array) > target_length:
            audio_array = audio_array[:target_length]

        return audio_array

    def mfcc_preprocessing(self, audio_data, sr=16000, n_mfcc=N_MFCC):
        mfccs = librosa.feature.mfcc(y=audio_data, sr=sr, n_mfcc=n_mfcc)

        target_length = self.target_frames
        num_frames = mfccs.shape[1]

        if num_frames < target_length:
            padding = np.zeros((n_mfcc, target_length - num_frames), dtype=np.float32)
            mfccs = np.concatenate((mfccs, padding), axis=1)  # (n_mfcc, target_length)
        elif num_frames > target_length:
            mfccs = mfccs[:, :target_length]

        return torch.tensor(mfccs.T, dtype=torch.float32)
