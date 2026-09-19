"""Update only supplementary figures that depend on the reconciled summaries."""
from pathlib import Path
import os
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(os.environ.get('GSE202537_PROJECT_ROOT', Path(__file__).resolve().parents[2])).resolve()
OUT = Path(os.environ.get('REVISION_FIGURE_OUTPUT', ROOT / 'results/figures/consistency_v3.0.1'))
OUT.mkdir(parents=True, exist_ok=True)
os.environ['REVISION_FIGURE_TABLES'] = str(ROOT/'results/revision_consistency_v5')
os.environ['REVISION_FIGURE_OUTPUT'] = str(OUT)
sys.path.insert(0, str(ROOT))
import revision_figures as rf

def save(fig, original_name):
    name = {'Figure-S12':'Figure-S3', 'Figure-S9':'Figure-S5'}[original_name]
    if name == 'Figure-S3':
        fig.axes[3].set_position([.08, .50, .36, .18])
        fig.axes[4].set_position([.64, .50, .32, .18])
        for ax in fig.axes[:3]:
            for label in ax.texts:
                if 'sign concordance' in label.get_text():
                    label.set_text(label.get_text().replace('): sign', ')\nsign'))
    for suffix in ['png', 'pdf', 'jpg']:
        kwargs = {'pil_kwargs': {'quality': 95}} if suffix == 'jpg' else {}
        fig.savefig(OUT/f'{name}.{suffix}', dpi=600, bbox_inches='tight', **kwargs)
    plt.close(fig)
    print('Updated', name)
rf.save = save
rf.figureS12()
rf.figureS9()
