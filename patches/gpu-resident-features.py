"""Patch openwakeword to hold the training features in VRAM instead of mmap.

Measured on the trainer VM (20 GB RAM, RTX 3090, 4 cores) during the training
stage: GPU utilisation 14%, CPU 37% idle, 7.2 GB of 8 GB swap in use, 213 MB RAM
free. Training is not compute-bound - it stalls on page faults.

The cause is the ACAV100M negative feature file: (5625000, 16, 96) float16, 17.28
GB, against 20 GB of RAM. `mmap_batch_generator` walks it SEQUENTIALLY, 1024 rows
per step, wrapping at the end - so at ~119 steps/s it completes a full pass every
~46 s, roughly 9 passes in 50,000 steps. The working set sits just under total RAM,
so each pass evicts what the next one needs and the kernel swaps to keep up.

17.28 GB fits in 24 GB of VRAM with the model and activations (~3.5 GB), leaving
~4 GB spare - provided the Kokoro containers are stopped first, since their two
CUDA contexts hold ~2.4 GB and are idle by the time training starts:

    docker compose stop kokoro kokoro2

This changes NO training dynamics: same arrays, same sequential order, same batch
composition. Only where the bytes live changes.

Three edits:

1. `mmap_batch_generator.__init__` - copy each array to a CUDA tensor, in chunks so
   the host never holds more than one chunk. np.load without mmap_mode would need a
   17 GB host allocation on a machine that has 213 MB free.

2. `mmap_batch_generator.__next__` - slice on the GPU. Arrays keep their on-disk
   dtype (ACAV100M is float16; the positive/negative feature files are float32), so
   slices are cast to float32 before concatenation - torch.cat requires a single
   dtype, and storing ACAV100M as float32 would need 34.6 GB. The cast is on a
   1024x16x96 slice, so it is negligible. Labels stay on the CPU: they are a small
   list, and the DataLoader's default_convert turns them into a tensor as before.

3. `train.py` - `num_workers=0`. CUDA tensors cannot cross a fork, so the current
   `num_workers=os.cpu_count()//2` would fail outright. With the data resident in
   VRAM there is no IO left to overlap, so the workers have nothing to do anyway.

There is deliberately NO fallback to the mmap path. A silent fallback would leave
the run looking healthy while delivering none of the benefit - which is exactly how
the onnxruntime CPU fallback cost 36 minutes of an 83-minute run before anyone
noticed the warning. If the allocation fails, this should fail loudly.
"""
import sys

path = sys.argv[1]
target = "train" if path.endswith("train.py") else "data"

with open(path) as f:
    content = f.read()

edits = []

if target == "data":
    # 1. Load each feature array into VRAM, chunked so the host never holds it all.
    old_load = """        self.data = {label: np.load(fl, mmap_mode='r') for label, fl in data_files.items()}"""
    new_load = '''        # GPU-resident features: see patches/gpu-resident-features.py. Copied in
        # chunks because the host cannot hold a 17 GB array. No fallback - if this
        # cannot allocate, the run must fail rather than quietly crawl on mmap.
        import torch as _torch
        if not _torch.cuda.is_available():
            raise RuntimeError(
                "gpu-resident-features patch is applied but CUDA is unavailable. "
                "Training would silently fall back to a path this patch removed."
            )
        self.data = {}
        for label, fl in data_files.items():
            _src = np.load(fl, mmap_mode='r')
            _dst = _torch.empty(tuple(_src.shape),
                                dtype=getattr(_torch, str(_src.dtype)),
                                device="cuda")
            _chunk = 65536
            for _i in range(0, _src.shape[0], _chunk):
                _dst[_i:_i + _chunk] = _torch.from_numpy(
                    np.ascontiguousarray(_src[_i:_i + _chunk])).cuda()
            print(f"  loaded {label} into VRAM: {tuple(_src.shape)} {_src.dtype} "
                  f"({_dst.element_size() * _dst.nelement() / 1e9:.2f} GB)")
            self.data[label] = _dst
            del _src'''
    edits.append((old_load, new_load))

    # 2. Concatenate on the GPU. Slices keep their stored dtype, so cast to float32.
    old_cat = """            return np.vstack(X), np.array(y)"""
    new_cat = """            # GPU-resident: X holds CUDA tensors of differing dtypes (ACAV100M is
            # float16, the generated features float32), so cast before concatenating.
            import torch as _torch
            return _torch.cat([_x.to(_torch.float32) for _x in X]), np.array(y)"""
    edits.append((old_cat, new_cat))

else:
    # 3. CUDA tensors cannot cross a fork, and there is no IO left to overlap.
    old_loader = """        X_train = torch.utils.data.DataLoader(IterDataset(batch_generator),
                                              batch_size=None, num_workers=n_cpus, prefetch_factor=16)"""
    new_loader = """        # num_workers must be 0 with GPU-resident features: CUDA tensors cannot
        # cross a fork, and with the data already in VRAM there is no IO to overlap.
        X_train = torch.utils.data.DataLoader(IterDataset(batch_generator),
                                              batch_size=None, num_workers=0)"""
    edits.append((old_loader, new_loader))

applied = 0
for old, new in edits:
    if old not in content:
        if new.split("\n")[0].strip() in content or "GPU-resident" in content:
            print(f"Already patched: {path}")
            sys.exit(0)
        print(f"ERROR: patch target not found in {path}:\n{old[:120]}")
        sys.exit(1)
    content = content.replace(old, new, 1)
    applied += 1

with open(path, "w") as f:
    f.write(content)

print(f"Patched: {path} ({applied} edit(s))")
