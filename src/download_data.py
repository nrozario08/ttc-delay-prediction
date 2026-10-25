"""Download the TTC subway delay files from the City of Toronto Open Data portal.

Files land in data/raw/. Run from the project root:

    python src/download_data.py
"""
import os
import time
import urllib.error
import urllib.request

BASE = "https://ckan0.cf.opendata.inter.prod-toronto.ca/dataset/996cfe8d-fb35-40ce-b569-698d51fc683b/resource"

# (local filename, resource path on the portal)
FILES = [
    ("delays_2019.xlsx", "1df6aace-fa16-40e9-a0d3-416a912bfaf1/download/ttc-subway-delay-data-2019.xlsx"),
    ("delays_2020.xlsx", "1ba66ead-cddf-453d-859d-b349d3286f02/download/ttc-subway-delay-2020.xlsx"),
    ("delays_2021.xlsx", "c6e4f5eb-6ed7-4db1-944f-87406faa5a09/download/ttc-subway-delay-2021.xlsx"),
    ("delays_2022.xlsx", "441143ca-8194-44ce-a954-19f8141817c7/download/ttc-subway-delay-2022.xlsx"),
    ("delays_2023.xlsx", "2fbec48b-33d9-4897-a572-96c9f002d66a/download/ttc-subway-delay-2023.xlsx"),
    ("delays_2024.xlsx", "2ee1a65c-da06-4ad1-bdfb-b1a57701e46a/download/ttc-subway-delay-2024.xlsx"),
    ("delays_2025_on.csv", "0b6e5c52-e993-46d6-8d74-8602ee224457/download/ttc-subway-delay-data-since-2025.csv"),
    ("delay_codes.csv", "b2d8f5e0-0997-46b5-8abd-caa685a0290b/download/code-descriptions.csv"),
    # older workbook, has the subway + SRT code lists that the csv above is missing
    ("delay_codes_old.xlsx", "3900e649-f31e-4b79-9f20-4731bbfd94f7/download/ttc-subway-delay-codes.xlsx"),
]

OUT_DIR = os.path.join("data", "raw")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    for name, path in FILES:
        dest = os.path.join(OUT_DIR, name)
        if os.path.exists(dest):
            print(f"skipping {name}, already there")
            continue
        print(f"downloading {name} ...")
        fetch(f"{BASE}/{path}", dest)
    print("done")


def fetch(url, dest, tries=6):
    # the portal throws random 502s on the big files, so just try again a few times
    for attempt in range(1, tries + 1):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=120) as resp:
                data = resp.read()
            with open(dest, "wb") as f:
                f.write(data)
            return
        except urllib.error.URLError as e:
            print(f"  attempt {attempt} failed: {e}")
            if attempt == tries:
                raise
            time.sleep(10)


if __name__ == "__main__":
    main()
