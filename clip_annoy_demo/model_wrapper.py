import json
import zipfile

import clip
import cv2
import matplotlib.pyplot as plt
import pandas as pd
import torch
from annoy import AnnoyIndex


class Singleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


def load_json(fname: str):
    with zipfile.ZipFile(fname, 'r') as zfile:
        with zfile.open(zfile.namelist()[0], 'r') as f:
            return json.load(f)


device = "cuda" if torch.cuda.is_available() else "cpu"


class ModelWrapper(metaclass=Singleton):
    def __init__(self, clip_path, annoy_path, images_path):
        self.clip_model, _ = clip.load(clip_path, device)
        self.index = AnnoyIndex(512, "angular")
        self.index.load(annoy_path, prefault=True)
        self.images = pd.read_pickle(images_path)

    def get_vector_text(self, x):
        text_inputs = torch.cat([clip.tokenize(x)]).to(device)
        with torch.no_grad():
            text_features = self.clip_model.encode_text(text_inputs)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        return text_features[0].cpu().numpy()

    def __call__(self, text):
        idx, dist = self.index.get_nns_by_vector(
            self.get_vector_text(text), 9, search_k=-1, include_distances=True
        )
        images_to_display = self.images.loc[idx].assign(dist=dist).copy()
        fig = plt.figure(figsize=(18, 18), facecolor="w")
        for i, (img, meta) in enumerate(images_to_display.sort_values("dist").groupby("image")):
            # TODO fix file path
            image = cv2.imread(img.replace("/home/jupyter/mnt", "/home/asciishell"), cv2.IMREAD_COLOR)
            assert image is not None
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            for _, row in meta.iterrows():
                image = cv2.rectangle(image, (row["xmin"], row["ymin"]), (row["xmax"], row["ymax"]), (255, 0, 0), 3)
            plt.subplot(3, 3, i + 1)
            plt.imshow(image)
            dists = "_".join(["{:.4f}".format(x) for x in meta["dist"]])
            confs = "_".join(["{:.4f}".format(x) for x in meta["roi_scores"]])
            plt.title(f"Cos sim {dists}, RoI {confs}")
        plt.suptitle(f'Query: "{text}"')
        plt.tight_layout()
        return fig
