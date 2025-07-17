import matplotlib

def setup_matplotlib(font_size=20, font_size_label=24):
    matplotlib.rcParams['text.usetex'] = True
    matplotlib.rcParams['text.latex.preamble'] = (
        r'\usepackage{amsmath,amssymb}'
        r'\boldmath'
        r'\bfseries'
    )

    matplotlib.rcParams['font.size'] = font_size
    matplotlib.rcParams['axes.titlesize'] = font_size
    matplotlib.rcParams['axes.labelsize'] = font_size
    matplotlib.rcParams['xtick.labelsize'] = font_size_label
    matplotlib.rcParams['ytick.labelsize'] = font_size_label
    matplotlib.rcParams['legend.fontsize'] = font_size
    matplotlib.rcParams['figure.titlesize'] = font_size
    matplotlib.rcParams['lines.linewidth'] = 3
    matplotlib.rcParams['legend.frameon'] = True
    matplotlib.rcParams['mathtext.fontset'] = 'cm'
    matplotlib.rcParams['mathtext.rm'] = 'serif'
    matplotlib.rcParams['mathtext.it'] = 'serif:italic'
    matplotlib.rcParams['mathtext.bf'] = 'serif:bold'
    matplotlib.rcParams['axes.labelweight'] = 'bold'
    matplotlib.rcParams['axes.titleweight'] = 'bold'
