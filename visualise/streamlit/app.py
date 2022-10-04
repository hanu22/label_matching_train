import logging
from pathlib import Path

import streamlit as st
from PIL import Image
from rich.logging import RichHandler

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(message)s",
    datefmt="%d/%m/%y %H:%M:%S",
    handlers=[RichHandler(markup=True)],
)
logger = logging.getLogger(__name__)

st.title("Project Name")

img = Image.open(
    Path(
        "images",
        "ds_logo.png",
    )
)
st.sidebar.image(img)

# streamlit run app.py
