import torch
from sklearn.metrics import f1_score
import pandas as pd


def pad(batch, classifier=None, dim: int = 512, max_len: int = None, device="cpu"):
    if max_len is None:
        max_len = max([len(x) for x in batch])
    attention_mask = []
    with torch.no_grad():
        for i in range(len(batch)):
            ones = [0] * len(batch[i])
            pad_length = max_len - len(batch[i])
            if pad_length == 0:
                attention_mask.append(ones)
                continue
            if classifier:
                padding = classifier.position_embeddings(
                    torch.ones((pad_length), dtype=torch.long, device=device)
                    * classifier.pad_token
                )
            else:
                padding = torch.zeros(pad_length, dim)
            zeros = [1] * pad_length
            batch[i] = torch.cat([batch[i].to(device), padding])
            attention_mask.append(ones + zeros)
        attention_mask = [[0] + x for x in attention_mask]
        return torch.stack([x.to(device) for x in batch]), torch.tensor(
            attention_mask
        ).to(device)


class Mapper:
    def __init__(self):
        self.original_labels = {
            0: "tax implications",
            1: "fraud",
            2: "service quality",
            3: "transactions",
            4: "customer safety",
            5: "customer service",
            6: "claimed benifit",
            7: "service charge",
            8: "service types",
            9: "others",
        }
        self.counts = {
            "others": 93,
            "service types": 286,
            "claimed benifit": 148,
            "customer service": 111,
            "transactions": 104,
        }
        self.map_to_labels = {
            0: 9,
            1: 9,
            2: 8,
            3: 3,
            4: 5,
            5: 5,
            6: 6,
            7: 8,
            8: 8,
            9: 9,
        }
        self.reverse_maps = {}
        for key, value in self.map_to_labels.items():
            if value in self.reverse_maps:
                self.reverse_maps[value].append(key)
            else:
                self.reverse_maps[value] = [key]

        self.multi_hot_labels = {
            x: ind for ind, x in enumerate(sorted(set(self.map_to_labels.values())))
        }

        self.multi_hot_labels_reverse_map = {
            value: {
                "New_label": key,
                "Label Name": self.original_labels[key],
                "Old label": self.reverse_maps[key],
                "count": self.counts[self.original_labels[key]],
            }
            for key, value in self.multi_hot_labels.items()
        }


from sklearn.metrics import f1_score


class Evaluate:
    def __init__(
        self,
        main_model,
        evs_model,
        device,
        loss_function,
        multi_model=False,
        batch_size=12,
    ):
        self.main_model = main_model.eval()
        self.evs_model = evs_model.eval()
        self.device = device
        self.multi_model = multi_model
        self.loss = loss_function
        self.batch_size = batch_size

    def eval(self, dataset):
        batch_size = self.batch_size
        real = []
        pred = []
        video_ids = []
        losses = []
        with torch.no_grad():
            for i in range(0, len(dataset), batch_size):
                batch = [x[0] for x in dataset[i : i + batch_size]]
                labels = torch.stack([x[1] for x in dataset[i : i + batch_size]])
                video_ids.append([x[-1] for x in dataset[i : i + batch_size]])
                audio_data = [torch.stack([y[1] for y in x]).squeeze(1) for x in batch]
                video_data = [[y[0] for y in x] for x in batch]
                video_data = [
                    [torch.stack(y).squeeze(1) if type(y) is list else y for y in x]
                    for x in video_data
                ]
                video_data = [
                    self.evs_model(*pad(x, self.evs_model, device=self.device))
                    for x in video_data
                ]
                video_data, v_mask = pad(
                    video_data, self.main_model, device=self.device
                )
                audio_data, a_mask = pad(
                    audio_data, self.main_model, device=self.device
                )
                if self.multi_model:
                    final_data = audio_data + video_data
                else:
                    final_data = audio_data
                label_output = self.main_model(final_data, attention_mask=a_mask)
                losses.append(
                    self.loss(label_output.view(-1, 3), labels.to(self.device).view(-1))
                )
                pred.append(torch.argmax(label_output.view(-1, 5, 3), dim=-1))
                real.append(labels)
        real, pred = torch.cat(real).cpu(), torch.cat(pred).cpu()
        return (
            f1_score(real >= 1, pred >= 1, average="micro"),
            f1_score(real >= 2, pred >= 2, average="micro"),
            torch.mean(torch.stack(losses)).item(),
            pd.DataFrame(
                {
                    "Video ID": [y for x in video_ids for y in x],
                    "Aspects real": (real >= 1).to(torch.long).tolist(),
                    "Aspects pred": (pred >= 1).to(torch.long).tolist(),
                    "Complaint real": (real >= 2).to(torch.long).tolist(),
                    "Complaint pred": (pred >= 2).to(torch.long).tolist(),
                }
            ),
        )
