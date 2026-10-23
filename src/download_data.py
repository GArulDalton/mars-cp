"""Download the four public datasets into data/ (about 25 MB compressed)."""
import urllib.request, zipfile
from paths import DATA
URLS = {
    "elec.csv": "https://raw.githubusercontent.com/scikit-multiflow/streaming-datasets/master/elec.csv",
    "daily-min-temperatures.csv": "https://raw.githubusercontent.com/jbrownlee/Datasets/master/daily-min-temperatures.csv",
    "household_power_consumption.zip": "https://raw.githubusercontent.com/jbrownlee/Datasets/master/household_power_consumption.zip",
    "machine_temperature_system_failure.csv": "https://raw.githubusercontent.com/numenta/NAB/master/data/realKnownCause/machine_temperature_system_failure.csv",
}
for name, url in URLS.items():
    dst = DATA / name
    if not dst.exists():
        print("downloading", name); urllib.request.urlretrieve(url, dst)
z = DATA / "household_power_consumption.zip"
if not (DATA / "household_power_consumption.txt").exists():
    zipfile.ZipFile(z).extractall(DATA)
print("data ready in", DATA)
