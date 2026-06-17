"""
Script 06 — GPU Memory Monitor During Training
-----------------------------------------------
nvidia-smi shows a snapshot of memory at one point in time.  During a real
training run, VRAM usage evolves:
  1. Baseline (Python + CUDA runtime loaded)
  2. Model parameters loaded
  3. Optimizer state allocated (Adam doubles memory per parameter)
  4. Forward pass: activations fill memory
  5. Backward pass: peak usage
  6. After optimizer.step() and zero_grad(): partial release

This script prints a memory report after each epoch so you can watch the
growth pattern and catch memory leaks early.

AICR hardware context:
  B200          — 192 GB VRAM  (very large; even big models fit in FP32)
  RTX PRO 6000  — 96 GB VRAM   (still large by most standards)

Having lots of VRAM does NOT mean you should leave it mostly empty.
Low memory usage + low utilisation = wasted hardware.
Goal: utilisation 80–100%, memory 70–90%.

Usage:
    python 06_memory_monitor.py    (GPU recommended; runs on CPU with warnings)
"""

import time
import torch
import torch.nn as nn

DEVICE          = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS          = 5
BATCH_SIZE      = 64
STEPS_PER_EPOCH = 25

SEP = "=" * 68

print(SEP)
print("  GPU MEMORY MONITOR — AICR")
print(SEP)
print(f"  Device : {DEVICE}")

if DEVICE == "cuda":
    p = torch.cuda.get_device_properties(0)
    total_gb = p.total_memory / 1024 ** 3
    print(f"  GPU    : {p.name}  ({total_gb:.1f} GB VRAM)")
    print()
    print("  AICR GPU memory context:")
    if total_gb > 150:
        print("  ↳ B200 (192 GB) — extremely large VRAM.  Even FP32 LLMs fit here.")
        print("    However, unused VRAM means unused compute — aim for 70–90% usage.")
    elif total_gb > 60:
        print("  ↳ RTX PRO 6000 (96 GB) — large VRAM.  Most workloads fit comfortably.")
        print("    Use BF16 to roughly halve memory and unlock Tensor Cores.")
    else:
        print(f"  ↳ {total_gb:.1f} GB available.")
print()


def report_memory(tag: str, reset_peak: bool = False) -> None:
    if DEVICE != "cuda":
        print(f"  [{tag}]  (no CUDA — memory stats unavailable)")
        return

    if reset_peak:
        torch.cuda.reset_peak_memory_stats()

    allocated_mb = torch.cuda.memory_allocated() / 1024 ** 2
    reserved_mb  = torch.cuda.memory_reserved() / 1024 ** 2
    peak_mb      = torch.cuda.max_memory_allocated() / 1024 ** 2
    total_mb     = torch.cuda.get_device_properties(0).total_memory / 1024 ** 2
    pct          = allocated_mb / total_mb * 100

    bar_len  = 32
    bar_fill = int(bar_len * pct / 100)
    bar      = "█" * bar_fill + "░" * (bar_len - bar_fill)

    print(f"  [{tag}]")
    print(f"    Allocated  : {allocated_mb:7.1f} MB  ({pct:4.1f}%)  [{bar}]")
    print(f"    Reserved   : {reserved_mb:7.1f} MB  (PyTorch allocator cache)")
    print(f"    Peak (epoch): {peak_mb:7.1f} MB")

    # Diagnostic hint
    if pct < 40:
        print(f"    ⚑  < 40% VRAM used — consider increasing batch size")
    elif pct > 90:
        print(f"    ⚑  > 90% VRAM — close to OOM; reduce batch or enable BF16")
    print()


# ── Model ─────────────────────────────────────────────────────────────────────
class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
            nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.classifier = nn.Linear(256 * 4 * 4, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x).view(x.size(0), -1))


print("Tracking memory as the model loads and training begins ...\n")
print(SEP)

report_memory("before model load")

model = ConvNet().to(DEVICE)
report_memory("after model.to(device)")

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
report_memory("after optimizer init  (Adam stores 2 extra tensors per param)")

criterion = nn.CrossEntropyLoss()

print(SEP)
print("  TRAINING LOOP")
print(SEP)

for epoch in range(1, EPOCHS + 1):
    t0 = time.perf_counter()

    for step in range(STEPS_PER_EPOCH):
        x = torch.randn(BATCH_SIZE, 3, 32, 32, device=DEVICE)
        y = torch.randint(0, 10, (BATCH_SIZE,), device=DEVICE)
        optimizer.zero_grad(set_to_none=True)
        out  = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()

    elapsed = time.perf_counter() - t0
    print(f"  Epoch {epoch}/{EPOCHS}  ({elapsed:.2f} s)")
    report_memory(f"end of epoch {epoch}", reset_peak=True)

# ── Final analysis ────────────────────────────────────────────────────────────
if DEVICE == "cuda":
    peak_mb  = torch.cuda.max_memory_allocated() / 1024 ** 2
    total_mb = torch.cuda.get_device_properties(0).total_memory / 1024 ** 2
    pct      = peak_mb / total_mb * 100

    print(SEP)
    print("  FINAL ANALYSIS")
    print(SEP)
    print(f"  Peak VRAM used : {peak_mb:.0f} MB / {total_mb:.0f} MB  ({pct:.1f}%)")
    print()

    if pct < 40:
        print("  Verdict: MEMORY UNDERUTILISED")
        print("  → Increase batch size.  The B200 has enormous VRAM — use it.")
        print("    Larger batches improve GPU utilisation and training stability.")
    elif pct < 70:
        print("  Verdict: MODERATE MEMORY USAGE")
        print("  → There is room to increase batch size.  Try doubling it and")
        print("    re-run — check that GPU utilisation also increases.")
    elif pct <= 90:
        print("  Verdict: HEALTHY MEMORY USAGE")
        print("  → Good balance.  You are using the hardware effectively.")
        print("    Monitor GPU utilisation alongside memory in nvidia-smi.")
    else:
        print("  Verdict: HIGH MEMORY — NEAR OOM")
        print("  → Enable BF16 autocast to roughly halve activation memory.")
        print("  → Or enable gradient checkpointing (trades compute for memory).")
        print("  → Or reduce batch size as a last resort.")

print()
print("  HANDY SNIPPET — add to any training script:")
print()
print("    torch.cuda.reset_peak_memory_stats()")
print("    # ... epoch training loop ...")
print("    peak_gb  = torch.cuda.max_memory_allocated() / 1024**3")
print("    total_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3")
print("    print(f'Peak VRAM: {peak_gb:.1f} GB / {total_gb:.1f} GB  '")
print("          f'({100*peak_gb/total_gb:.0f}%)')")
print(SEP)
