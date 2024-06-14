import pandas as pd
import sqlite3

class DataLoader:
    def __init__(self, dbfile):
        self.dbfile = dbfile

    def load_data(self):
        conn = sqlite3.connect(self.dbfile)
        data_df = pd.read_sql("SELECT * FROM id_text", conn)
        type_df = pd.read_sql("SELECT * FROM id_dialect", conn)
        conn.close()
        return data_df, type_df
