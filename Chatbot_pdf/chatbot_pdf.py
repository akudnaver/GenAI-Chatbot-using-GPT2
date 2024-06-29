import argparse

from pdfminer.high_level import extract_text
from sentence_transformers import SentenceTransformer, CrossEncoder, util

from text_generation import Client