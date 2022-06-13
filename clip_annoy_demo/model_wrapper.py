import sys

sys.path.append("../notebooks")

import json
import zipfile

import PIL
import clip
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from assh_utils import nms


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


class Detector(metaclass=Singleton):
    def __init__(
            self,
            clip_model,
            clip_preprocess,
            images,
            index,
            device,
            cos_thr=0.2,
            min_size=(32, 32),
            min_conf=0.05,
            iou_thr=0.45,
            max_crops=10,
    ):
        self.clip_model = clip_model
        self.clip_preprocess = clip_preprocess
        self.images = images
        self.index = index
        self.device = device
        self.cos_thr = cos_thr
        self.min_size = min_size
        self.min_conf = min_conf
        self.iou_thr = iou_thr
        self.max_crops = max_crops

    def get_vector_text(self, text):
        text_inputs = torch.cat([clip.tokenize(text)]).to(self.device)
        with torch.no_grad():
            text_features = self.clip_model.encode_text(text_inputs)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        return text_features[0].cpu().numpy()

    def get_vector_image(self, x):
        img = PIL.Image.open(x)
        img.thumbnail((640, 640), PIL.Image.ANTIALIAS)
        return self._get_vector_image(img)

    def _get_vector_image(self, img):
        img.show()
        image_input = self.clip_preprocess(img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            image_features = self.clip_model.encode_image(image_input)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        return image_features[0].cpu().numpy()

    @staticmethod
    def _cos_sim_matrix(a, b):
        return np.dot(a, b) / np.linalg.norm(b)

    def get_ns_by_vector(self, vec, n):
        vec = vec.astype(np.float16)
        dist = self._cos_sim_matrix(self.index, vec)
        idx = np.argsort(dist)
        top_idx = idx[: -n - 1: -1]
        return top_idx, dist[top_idx], dist

    def get_ns_by_text(self, text, n):
        return self.get_ns_by_vector(self.get_vector_text(text), n)

    def get_prediction_df(self, text, n):
        idx, dist, all_dist = self.get_ns_by_text(text, n)
        images_to_display = self.images.loc[idx].assign(dist=dist).copy()

        img_same = self.images[(self.images["image"].isin(images_to_display["image"]))].copy()
        img_same["dist"] = all_dist[img_same.index]
        img_same = img_same[
            (
                    (img_same["dist"] > self.cos_thr)
                    & (img_same["xmax"] - img_same["xmin"] >= self.min_size[0])
                    & (img_same["ymax"] - img_same["ymin"] >= self.min_size[1])
                    & (img_same["confidence"] > self.min_conf)
            )
            | img_same.index.isin(images_to_display.index)
            ]

        img_nms = nms(
            img_same[["xmin", "ymin", "xmax", "ymax"]].values,
            img_same["confidence"].values,
            self.iou_thr,
            max_dets=self.max_crops,
        )
        img_nms = img_same.iloc[img_nms]
        img_same["flag_orig"] = img_same.index.isin(images_to_display.index)
        img_same["flag_nms"] = img_same.index.isin(img_nms.index)
        return img_same

    def __call__(self, text):
        query_result = self.get_prediction_df(text, 50)
        len_images = query_result["image"].nunique()
        fig = plt.figure(figsize=(18, (len_images // 3 + 1) * 6), facecolor="w")
        for i, (img, meta) in enumerate(query_result.sort_values("dist").groupby("image")):
            image = cv2.imread(img.replace("/home/jupyter/mnt", "/home/asciishell"), cv2.IMREAD_COLOR)
            assert image is not None
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            for _, row in meta.iterrows():
                flag_orig = row['flag_orig']
                flag_nms = row['flag_nms']
                if not flag_nms and not flag_orig:
                    continue
                image = cv2.rectangle(
                    image,
                    (row["xmin"], row["ymin"]),
                    (row["xmax"], row["ymax"]),
                    (255 * flag_orig, 255 * (not flag_orig), 255 * flag_nms),
                    2 + flag_orig * 2 + flag_nms,
                )
            plt.subplot(len_images // 3 + 1, 3, i + 1)
            plt.imshow(image)
            dists = "_".join(["{:.4f}".format(x) for x in meta[meta['flag_orig']]["dist"].head(5)])
            confs = "_".join(["{:.4f}".format(x) for x in meta[meta['flag_orig']]["confidence"].head(5)])
            plt.title(f"Cos sim {dists}, Confidence {confs}")
        plt.suptitle(f'Query: "{text}"')
        plt.tight_layout()
        return fig
