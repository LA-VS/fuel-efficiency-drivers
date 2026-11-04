"""Download the raw UCI Auto MPG file into data/raw."""

from fuel_efficiency.data import fetch

if __name__ == "__main__":
    print(f"Saved {fetch(force=True)}")
