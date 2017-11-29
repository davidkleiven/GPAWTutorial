import io
import numpy as np
import sqlite3 as sq

def adapt_array(arr):
    out = io.BytesIO()
    np.save(out,arr)
    out.seek(0)
    return sq.Binary(out.read())

def convert_array(text):
    out = io.BytesIO(text)
    out.seek(0)
    return np.load(out)
