import os
import glob
import pretty_midi
import soundfile as sf
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor

from preprocessing.constants import GIANTMIDI_DATA_PATH, SAMPLE_RATE

# Define the path to a piano SoundFont (.sf2) file. 
# You can download a free piano SoundFont like "FluidR3_GM.sf2" or "Yamaha Grand Lite"
SOUNDFONT_PATH = "C:/Users/alau2/Documents/AMT-backend/preprocessing/UprightPianoKW-20220221.sf2"

def process_midi(midi_path):
    basename = os.path.splitext(os.path.basename(midi_path))[0]
    base_dir = os.path.dirname(os.path.dirname(midi_path))
    wav_path = os.path.join(base_dir, 'wav', f"{basename}.wav")
    
    # Skip conversion if the WAV file already exists
    if os.path.exists(wav_path):
        return True 
        
    try:
        pm = pretty_midi.PrettyMIDI(midi_path)
        
        # Use SoundFont if available to synthesize realistic piano sound
        if os.path.exists(SOUNDFONT_PATH):
            audio_data = pm.fluidsynth(fs=SAMPLE_RATE, sf2_path=SOUNDFONT_PATH)
        else:
            print(f"Warning: SoundFont not found at {SOUNDFONT_PATH}. Falling back to default sine waves.")
            audio_data = pm.synthesize(fs=SAMPLE_RATE)
            
        sf.write(wav_path, audio_data, SAMPLE_RATE)
        return True
    except Exception as e:
        print(f"Error converting {midi_path}: {e}")
        return False
        
def main():
    midis_dir = os.path.join(GIANTMIDI_DATA_PATH, 'midis')
    if not os.path.exists(midis_dir):
        print(f"Directory not found: {midis_dir}")
        return
        
    wavs_dir = os.path.join(GIANTMIDI_DATA_PATH, 'wav')
    os.makedirs(wavs_dir, exist_ok=True)

    midi_files = glob.glob(os.path.join(midis_dir, '*.mid')) + glob.glob(os.path.join(midis_dir, '*.midi'))
    print(f"Found {len(midi_files)} MIDI files in {midis_dir}. Starting conversion...")
    
    with ProcessPoolExecutor() as executor:
        list(tqdm(executor.map(process_midi, midi_files), total=len(midi_files), desc="Converting MIDIs to WAVs"))
        
if __name__ == "__main__":
    main()
    