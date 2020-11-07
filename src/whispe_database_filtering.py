from os import listdir
from os.path import isfile, join
from src.utils import load_audio_file
from src.filters import inverse_filter
from tqdm import tqdm
import soundfile as sf

path = "../data/whi-spe-dataset"
filter_order = 10

list_of_wavefiles = [
    f for f in listdir(path) if (isfile(join(path, f)) and f.endswith("n.wav"))
]

print("Start inverse filtering...")
for wavfile in tqdm(list_of_wavefiles):

    tmp_s, sr = load_audio_file(join(path, wavfile))
    tmp_s_filtered = inverse_filter(tmp_s, filter_order)
    new_wavfile = wavfile.strip("n.wav") + "_filtered.wav"
    sf.write(join(path, new_wavfile), tmp_s_filtered, sr)

print("Inverse filtering completed!")
