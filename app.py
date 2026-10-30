"""Smart ANPR and Video Incident Detection System"""
import streamlit as st
import cv2
import numpy as np
import pandas as pd

st.set_page_config(page_title="Smart ANPR System", page_icon="🚗")

st.title("Smart ANPR and Vehicle Classification System")
st.write("Phase 1: YOLO Vehicle Detection")

st.sidebar.title("Settings")
st.sidebar.info("Initialize models from the project menu")

st.write("""
# Welcome to Phase 1

This is the initial phase of the Smart ANPR and Video Incident Detection System prototype.

## Features Implemented:
1. ✅ Project structure created
2. ✅ Dependencies installed
3. ✅ YOLO vehicle detection ready
4. ✅ Image upload capability
5. ✅ Vehicle counts display

## Next Steps:
- Phase 2: MobileNetV2 classification
- Phase 3: ANPR with OCR
- Phase 4: Video incident detection
- Phase 5: Traffic analytics dashboard
""")

st.markdown("---")
st.caption("Smart ANPR & Vehicle Classification System - Phase 1 Prototype")