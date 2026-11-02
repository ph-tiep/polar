"""Launch Phase 1 with RT initialized before TensorFlow (native compatibility)."""
import sys
sys.path.insert(0, '/workspace/polar')
import sionna.rt
sionna.__version__ = sionna.rt.__version__
import main_phase1
from pathlib import Path
# Keep generated results separate from tracked input data and figures.
output = Path('/workspace/polar-results/phase1')
for key, leaf in [('DATASET_DIR', 'dataset/140GHz'), ('FIGURE_DIR', 'figures'), ('LOG_DIR', 'logs')]:
    folder = output / leaf
    folder.mkdir(parents=True, exist_ok=True)
    setattr(main_phase1.config, key, str(folder))
if __name__ == '__main__':
    main_phase1.main(run_mode='debug')
