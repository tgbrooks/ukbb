import pandas
import numpy
import matplotlib
import re

def read_actigraphy(file):
    data = pandas.read_csv(file, sep=",", index_col=0, parse_dates=[0])

    data.rename(columns={data.columns[0]:"acceleration"}, inplace=True)# Acceleration column name contains some extra information, ignore it

    # clear the imputed values since they are junk
    is_imputed = data.imputed.astype(bool)
    data.loc[is_imputed] = float("NaN")
    data.loc[is_imputed, 'imputed'] = 1 # restore the 'imputed' value since we just overwrote it as NaN

    return data

# Q-value utility
def BH_FDR(ps):
    ''' Benjamini-Hochberg FDR control

    Converts p values to q values'''

    # For the purposes of comparison, an implementation of Benjamini Hochberg correction
    sort_order = numpy.argsort(ps)

    adjusted = numpy.zeros(ps.shape)
    adjusted[sort_order] = numpy.array(ps)[sort_order]*len(ps)/numpy.arange(1,len(ps)+1)

    # Make monotone, skipping NaNs
    m = 1
    for r in sort_order[::-1]:
        if numpy.isfinite(adjusted[r]):
            m = min(adjusted[r], m)
            adjusted[r] = m

    return adjusted # the q-values

# Legend-making utilities
def legend_from_colormap(fig, colormap, maxlength=35, names=None, **kwargs):
    if names is None:
        names = {cat:cat for cat in colormap.keys()}
    legend_elts = [matplotlib.lines.Line2D(
                            [0],[0],
                            marker="o", markerfacecolor=c, markersize=10,
                            label=truncate(names[cat],maxlength) if not pandas.isna(cat) else "NA",
                            c=c, lw=0)
                        for cat, c in colormap.items()]
    fig.legend(handles=legend_elts, **kwargs)

def legend_of_pointscale(fig, offset, values_to_size, values_to_show, fmt="{}", **kwargs):
    legend_elts = [matplotlib.lines.Line2D(
                            [0],[0],
                            marker="o", markerfacecolor='k',
                            markersize=numpy.sqrt(offset + values_to_size * value),
                            label=fmt.format(value),
                            c='k', lw=0)
                        for value in values_to_show]
    fig.legend(handles=legend_elts, **kwargs)

## Prepare utility functions
def truncate(string, N):
    if N is None:
        return string
    if len(string) <= N:
        return string
    return string[:N-1] + "…" #Add elipsis unicode character, for brevity

def wrap(string, N):
    # Wrap to lines at most length N, preserving words
    words = string.split()
    lines = []
    current_line = []
    for word in words:
        if len(word) + sum(len(w) for w in current_line) > N:
            lines.append(' '.join(current_line))
            current_line = []
        current_line.append(word)
    lines.append(' '.join(current_line))
    return '\n'.join(lines)

def capitalize(string):
    # Capitalize first letter in each word, broken by any of a large number of characters
    split = re.split(r"([\s&,\\/]*)", string)
    return ''.join(s.capitalize() for s in split)