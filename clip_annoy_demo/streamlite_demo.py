import argparse
import sys

import clip
import numpy as np
import pandas as pd
import streamlit as st

from model_wrapper import Detector, device


def parse_args(args):
    parser = argparse.ArgumentParser('Model config files')
    parser.add_argument('--clip', help='CLIP model path', required=True)
    parser.add_argument('--embeddings', help='Embeddings index path', required=True)
    parser.add_argument('--images', help='Images path', required=True)
    return parser.parse_args(args)


args = parse_args(sys.argv[1:])


@st.experimental_singleton
def create_model():
    images = pd.read_pickle(args.images)
    clip_model, clip_preprocess = clip.load(args.clip, device)
    index = np.load(args.embeddings)

    detector = Detector(
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
    )
    return detector


def main():
    if "button_id" not in st.session_state:
        st.session_state["button_id"] = ""
    if "color_to_label" not in st.session_state:
        st.session_state["color_to_label"] = {}
    pages = {
        "Query demo": query_demo,
        # "Keypoints demo": keypoints_demo,
    }
    page = st.sidebar.selectbox("Page:", options=list(pages.keys()))
    pages[page]()


def query_demo():
    model = create_model()
    x = st.sidebar.text_input("Query text", key="name")
    st.write(model(x))


if __name__ == "__main__":
    main()
