import argparse
import sys

import streamlit as st

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


model = create_model()

x = st.text_input("Query text", key="name")
st.write(model(x))
