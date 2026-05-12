import os

# --- CONFIGURATION ---
INPUT_FOLDER  = 'data/dataset/downsampled_5fps_videos'   # videos fed into DLC GUI
OUTPUT_FOLDER = 'data/dataset/deeplabcut_really_labeled'  # where .h5 files land

# ---------------------------------------------------------------
# Find all video stems that already have a .h5 output file
# e.g. "GR9-hmzDUY4_snip_10_superanimal_quadruped_hrnet_w32_...h5"
#       → stem = "GR9-hmzDUY4_snip_10"
# ---------------------------------------------------------------
done_stems = set()
for fname in os.listdir(OUTPUT_FOLDER):
    if fname.endswith('.h5'):
        # strip everything from "_superanimal_..." onward
        stem = fname.split('_superanimal_')[0]
        done_stems.add(stem)

print(f"Found {len(done_stems)} completed videos in output folder.\n")

# ---------------------------------------------------------------
# Delete matching source videos
# ---------------------------------------------------------------
deleted = []
skipped = []

for fname in os.listdir(INPUT_FOLDER):
    if not fname.lower().endswith(('.mp4', '.avi', '.mov')):
        continue

    stem = os.path.splitext(fname)[0]   # strip .mp4 etc.
    if stem in done_stems:
        path = os.path.join(INPUT_FOLDER, fname)
        os.remove(path)
        deleted.append(fname)
        print(f"[DELETED] {fname}")
    else:
        skipped.append(fname)

print(f"\n--- DONE ---")
print(f"Deleted : {len(deleted)} videos")
print(f"Remaining (not yet processed): {len(skipped)} videos")