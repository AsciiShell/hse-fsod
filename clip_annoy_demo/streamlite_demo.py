import argparse
import sys

import streamlit as st
from streamlit_drawable_canvas import st_canvas

from model_wrapper import ModelWrapper


def parse_args(args):
    parser = argparse.ArgumentParser('Model config files')
    parser.add_argument('--clip', help='CLIP model path', required=True)
    parser.add_argument('--annoy', help='Annoy index path', required=True)
    parser.add_argument('--images', help='Images path', required=True)
    return parser.parse_args(args)


args = parse_args(sys.argv[1:])


@st.experimental_singleton
def create_model():
    model = ModelWrapper(args.clip, args.annoy, args.images)
    return model


def main():
    if "button_id" not in st.session_state:
        st.session_state["button_id"] = ""
    if "color_to_label" not in st.session_state:
        st.session_state["color_to_label"] = {}
    pages = {
        "Annoy demo": annoy_demo,
        "Keypoints demo": keypoints_demo,
    }
    page = st.sidebar.selectbox("Page:", options=list(pages.keys()))
    pages[page]()


def annoy_demo():
    model = create_model()
    x = st.sidebar.text_input("Query text", key="name")
    st.write(model(x))


def keypoints_demo():
    drawing_mode = 'point'
    stroke_width = 3
    point_display_radius = 3
    stroke_color = "#000"
    bg_color = "#eee"
    # bg_image = st.sidebar.file_uploader("Background image:", type=["png", "jpg"])
    realtime_update = True
    height = 500

    # Create a canvas component
    canvas_results = []
    for i in range(3):
        canvas_result = st_canvas(
            fill_color="rgba(255, 165, 0, 0.3)",  # Fixed fill color with some opacity
            stroke_width=stroke_width,
            stroke_color=stroke_color,
            background_color=bg_color,
            background_image=None,
            update_streamlit=realtime_update,
            height=height,
            drawing_mode=drawing_mode,
            point_display_radius=point_display_radius if drawing_mode == 'point' else 0,
            display_toolbar=True,
            key=f"full_app_{i}",
        )

        canvas_results.append(canvas_result)


if __name__ == "__main__":
    main()
