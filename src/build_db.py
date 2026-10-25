"""Load the raw TTC delay files into a SQLite database (data/ttc.db).

Only light work happens here: stack the yearly files together and store them in
a table called delays_raw. The actual cleaning is done in SQL (sql/01_clean.sql).

    python src/build_db.py
"""
import glob
import os
import sqlite3

import pandas as pd

RAW_DIR = os.path.join("data", "raw")
DB_PATH = os.path.join("data", "ttc.db")

KEEP = ["Date", "Time", "Day", "Station", "Code", "Min Delay", "Min Gap", "Bound", "Line", "Vehicle"]


def load_delay_files():
    frames = []
    for path in sorted(glob.glob(os.path.join(RAW_DIR, "delays_*"))):
        if path.endswith(".xlsx"):
            # 2019-2021 have one sheet per month, newer files have a single sheet
            sheets = pd.read_excel(path, sheet_name=None)
            df = pd.concat(sheets.values(), ignore_index=True)
        else:
            df = pd.read_csv(path)

        df.columns = [str(c).strip() for c in df.columns]
        df = df[[c for c in KEEP if c in df.columns]].copy()
        df["source_file"] = os.path.basename(path)
        print(f"{os.path.basename(path)}: {len(df):,} rows")
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def load_codes():
    # the newer csv doesn't cover every code that shows up in the data, so also pull
    # in the subway and SRT lists from the older workbook (they sit side by side)
    new = pd.read_csv(os.path.join(RAW_DIR, "delay_codes.csv")).drop(columns=["_id"], errors="ignore")
    new.columns = [c.lower() for c in new.columns]

    old = pd.read_excel(os.path.join(RAW_DIR, "delay_codes_old.xlsx"), header=None, skiprows=2)
    sub = old[[2, 3]].set_axis(["code", "description"], axis=1)
    srt = old[[6, 7]].set_axis(["code", "description"], axis=1)

    codes = pd.concat([new, sub, srt], ignore_index=True).dropna()
    codes["code"] = codes["code"].astype(str).str.strip().str.upper()
    codes["description"] = codes["description"].astype(str).str.strip().str.upper()
    # the old workbook has mangled dashes/quotes (an en dash comes through as "Â" plus
    # two control characters). Turn the dashes back into "-" and drop the rest.
    codes["description"] = codes["description"].str.replace("Â\x80\x93", "-", regex=False)
    codes["description"] = codes["description"].str.replace(r"[Â\x80-\x9f]", "", regex=True)
    codes["description"] = codes["description"].str.replace(r"\s+", " ", regex=True).str.strip()
    return codes.drop_duplicates(subset="code", keep="first")


def main():
    delays = load_delay_files()

    # dates come through as timestamps in the excel files and strings in the csv
    delays["Date"] = pd.to_datetime(delays["Date"], errors="coerce").dt.strftime("%Y-%m-%d")

    codes = load_codes()

    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)
    with sqlite3.connect(DB_PATH) as con:
        delays.to_sql("delays_raw", con, index=False)
        codes.to_sql("delay_codes", con, index=False)
        print(f"wrote {len(delays):,} raw rows to {DB_PATH}")

        # cleaning happens in SQL
        with open(os.path.join("sql", "01_clean.sql")) as f:
            con.executescript(f.read())
        n = con.execute("SELECT COUNT(*) FROM delays").fetchone()[0]
        print(f"delays table: {n:,} rows after cleaning")


if __name__ == "__main__":
    main()
