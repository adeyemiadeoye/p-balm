import matplotlib
import seaborn as sns

from matplotlib import style
import scienceplots

def setup_matplotlib(font_scale=2.5, style_name="science", grid=True):
    if grid:
        sns.set_style(style.library[style_name])
        matplotlib.rcParams.update({'axes.axisbelow': True, 'axes.grid': True, 'grid.alpha': 0.5, 'grid.color': 'k', 'grid.linestyle': '--', 'grid.linewidth': 0.5, 'legend.fancybox': True, 'legend.framealpha': 1.0, 'legend.frameon': True, 'legend.numpoints': 1})
    else:
        sns.set_style(style.library[style_name])
        matplotlib.rcParams.update({'axes.grid': False, 'legend.fancybox': True, 'legend.frameon': True, 'legend.numpoints': 1})

    sns.set_context("paper", font_scale=font_scale,
                    rc={"lines.linewidth": 3,
                    "mathtext.fontset": "stix",
                    })
    
    matplotlib.rcParams['text.usetex'] = True
    matplotlib.rcParams['text.latex.preamble'] = (
        r'\usepackage{amsmath,amssymb}'
        r'\boldmath'
        r'\bfseries'
    )
