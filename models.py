from tqdm.notebook import tqdm
import torch
import torch.nn as nn
import os
import pickle
from moviepy.editor import VideoFileClip
import numpy as np

torch.manual_seed(42)


class AudioVideoExtractor(nn.Module):

    def __init__(
        self,
        clip_model,
        clip_processor,
        device="cuda",
    ):
        super(AudioVideoExtractor, self).__init__()
        self.text_model = clip_model.text_model
        self.text_projection = clip_model.text_projection
        self.vision_model = clip_model.vision_model
        self.vision_projection = clip_model.visual_projection
        self.text_processor = clip_processor.tokenizer
        self.vision_processor = clip_processor.image_processor
        self.device = device

    def extract_frames(self, video_name, transcript):
        clip = VideoFileClip(video_name)
        print("Processing frames")
        frames = []
        for image in clip.iter_frames():
            encode_image = self.vision_processor(image, return_tensors="pt").to(
                self.device
            )
            vec = self.vision_projection(
                self.vision_model(**encode_image).pooler_output
            ).cpu()
            del image
            frames.append(vec)
        print("Processing Text")
        time_frames = np.linspace(0, (len(frames) // 3), len(frames))
        chunks = []
        print("creating chunks")
        for segment in tqdm(transcript["segments"]):
            frame_data = []
            for time_span, frame in zip(time_frames, frames):
                if time_span >= segment["start"] and time_span <= segment["end"]:
                    frame_data.append(frame)
            if frame_data and segment["text"].strip():
                audio_data = self.text_projection(
                    self.text_model(
                        **self.text_processor(segment["text"], return_tensors="pt").to(
                            self.device
                        )
                    ).pooler_output
                ).cpu()
                chunks.append([frame_data, audio_data])

        return chunks


class EncodeVideoSegment(nn.Module):

    def __init__(self, nhead=8, max_len=128, d_model=512, dropout=0.2, num_blocks=8):
        super(EncodeVideoSegment, self).__init__()
        self.position_embeddings = nn.Embedding(max_len + 2, d_model)
        self.max_len = max_len

        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=d_model, nhead=nhead, dropout=dropout, batch_first=True
            ),
            num_layers=num_blocks,
        )
       
        self.cls_token = self.max_len
        self.pad_token = self.max_len + 1

    def forward(self, x, attention_mask):
        assert (
            x.shape[-2] <= self.max_len
        ), f"Please ensure number of frames is less than or equal to '{self.max_len}'"
        with torch.no_grad():
            cls_tokens = self.position_embeddings(
                torch.ones((x.shape[0], 1), dtype=torch.long, device=x.device)
                * self.max_len
            )
            x = torch.cat([cls_tokens, x], dim=1)

        positions = torch.arange(0, x.shape[-2], device=x.device).unsqueeze(0)
        positional_encodings = self.position_embeddings(positions)
        x += positional_encodings

        output = self.encoder(x, src_key_padding_mask=attention_mask.to(torch.bool))
        return output[:, 0]


class MultiLabelVideoClassifier(nn.Module):
    def __init__(
        self,
        input_shape,
        nhead=8,
        d_model=512,
        num_blocks=16,
        max_seq_len=1024,
        dropout=0.2,
    ):
        super(MultiLabelVideoClassifier, self).__init__()
        self.position_embeddings = nn.Embedding(max_seq_len + 2, d_model)
        self.cls_token = max_seq_len
        self.pad_token = max_seq_len + 1

        self.max_seq_len = max_seq_len
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dropout=dropout,
                batch_first=True,
            ),
            num_layers=num_blocks,
        )
        self.project = nn.Linear(input_shape, d_model)
        self.classification_layer = nn.Linear(d_model, 5 * 3)
        self.softmax = nn.Sigmoid()

    def forward(self, x, attention_mask):
        with torch.no_grad():
            cls_token = self.position_embeddings(
                torch.ones((x.shape[0], 1), dtype=torch.long, device=x.device)
                * self.max_seq_len
            )
            x = torch.cat([cls_token, x], dim=1)
        x = self.project(x)
        seq_length = x.size(1)
        positions = torch.arange(0, seq_length, device=x.device).unsqueeze(0)
        positional_encodings = self.position_embeddings(positions)
        x += positional_encodings
        x = self.encoder(x, src_key_padding_mask=attention_mask.to(torch.bool))
        return self.classification_layer(x[:, 0])
