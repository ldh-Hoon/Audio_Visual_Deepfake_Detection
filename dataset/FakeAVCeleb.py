import os
import random
import torch
import numpy as np
import cv2
import pandas as pd
from torch.utils.data import Dataset
from pydub import AudioSegment
import librosa

from var import *

N_MFCC = 60
SIZE = 112

class FakeAVCeleb(Dataset):
    def __init__(self, root_dir, num_samples=100, split="train", target_frames=100, transform=None, collate_fn=None):
        self.root_dir = root_dir
        self.transform = transform
        self.data = []
        self.target_frames = target_frames

        self.load_metadata(os.path.join(root_dir, "meta_data.csv"))


        # 비율 맞추기
        self.balance_data()

        if num_samples > 0:
            self.data = random.sample(self.data, min(num_samples, len(self.data)))

        self.train_data, self.val_data, self.test_data = self.split_data()

        if split == "train":
            print(f"{split}:{len(self.train_data)}")
            self.data = self.train_data
        elif split == "val":
            print(f"{split}:{len(self.val_data)}")
            self.data = self.val_data
        elif split == "test":
            print(f"{split}:{len(self.test_data)}")
            self.data = self.test_data
        else:
            raise ValueError("split must be one of ['train', 'val', 'test']")

    def split_data(self):
        train_size = int(0.7 * len(self.data))  # 70%
        val_size = int(0.15 * len(self.data))    # 15%
        test_size = len(self.data) - train_size - val_size  # 나머지 15%

        train_data = self.data[:train_size]
        val_data = self.data[train_size:train_size + val_size]
        test_data = self.data[train_size + val_size:]

        return train_data, val_data, test_data

    def load_metadata(self, metadata_file):
        metadata = pd.read_csv(metadata_file)

        for _, row in metadata.iterrows():
            label = 1 if row['method'] == 'real' else 0
            video_path = os.path.join(self.root_dir, row['type'], row['race'], row['gender'], row['source'], row['path'])  # 전체 경로 만들기
            self.data.append((label, video_path))


    def balance_data(self):
        real_samples = [item for item in self.data if item[0] == 1]
        fake_samples = [item for item in self.data if item[0] == 0]

        min_samples = min(len(real_samples), len(fake_samples))

        sampled_real = random.sample(real_samples, min_samples)
        sampled_fake = random.sample(fake_samples, min_samples)

        self.data = sampled_real + sampled_fake
        random.shuffle(self.data)  # 데이터 섞기

        print(f"Balanced samples: {len(self.data)} (Real: {len(sampled_real)}, Fake: {len(sampled_fake)})")

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
        W, H = SIZE, SIZE

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.resize(frame, (W, H))
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

    def mfcc_preprocessing(self, audio_data, sr=16000, n_mfcc=60):
        mfccs = librosa.feature.mfcc(y=audio_data, sr=sr, n_mfcc=n_mfcc)

        target_length = self.target_frames
        num_frames = mfccs.shape[1]

        if num_frames < target_length:
            padding = np.zeros((n_mfcc, target_length - num_frames), dtype=np.float32)
            mfccs = np.concatenate((mfccs, padding), axis=1)  # (n_mfcc, target_length)
        elif num_frames > target_length:
            mfccs = mfccs[:, :target_length]

        return torch.tensor(mfccs.T, dtype=torch.float32)

