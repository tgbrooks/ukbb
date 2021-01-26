import numpy
import pandas
import matplotlib
import pylab
import scipy.stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
import seaborn as sns

import util
from phewas_tests import covariates, OLS

# Plot config by variables
plot_config = {
    "acceleration_RA": {
        "xbottom": 0.6,
        "xtop": 1.0,
        "point_width": 0.01,
        "bandwidth": 0.15,
        "label": "RA",
    },
    "amplitude": {
        "xbottom": 0.1,
        "xtop": 0.9,
        "point_width": 0.01,
        "bandwidth": 0.15,
        "label": "Amplitude",
    },
    "temp_RA": {
        "xbottom": -0.12,
        "xtop": 0.0,
        "point_width": 0.01,
        "bandwidth": 0.15,
        "label": "Temperature RA"
    }
}
# Labels for quintile plots
quintile_labels = ["First", "Second", "Third", "Fourth", "Fifth"]

def p_to_size(p_value):
    # maps a p-value to a point size for scatter plots
    CAP = 10 # -log P-values above this point are all the same size
    SCALE = 4
    MINIMUM = 1
    logp = -numpy.log10(p_value)
    logp = numpy.minimum(logp, CAP)
    return logp * SCALE + MINIMUM

def gaussian_kde(data, *args, **kwargs):
    # Gaussian KDE but drops NAs
    d = data[~data.isna()]
    return scipy.stats.gaussian_kde(d, *args, **kwargs)



def local_regression(x,y, out_x, bw=0.05):
    # Preform a local regression y ~ x and evaluate it at the provided points `out_x`
    reg = sm.nonparametric.KernelReg(exog=x, endog=y, var_type='c',
                                    bw=[bw],
                                    )
    fit, _ = reg.fit(out_x)
    return fit



class Plotter:
    def __init__(self, phecode_info, colormaps, activity_variables, activity_variable_descriptions, quantitative_variable_descriptions):
         self.phecode_info = phecode_info
         self.colormaps = colormaps
         self.activity_variables = activity_variables
         self.activity_variable_descriptions = activity_variable_descriptions
         self.quantitative_variable_descriptions = quantitative_variable_descriptions

         self.quintile_labels = quintile_labels

    def sex_difference_plot(self, d, color_by="phecode_category", cmap="Dark2", lim=0.5, ax=None, legend=True, labels=True, names=None):
        if color_by is not None:
            if type(cmap) == str:
                cats = d[color_by].unique()
                if cmap == "rainbow":
                    cmap = [pylab.get_cmap("rainbow")(i) for i in numpy.arange(len(cats))/len(cats)]
                else:
                    cmap = [pylab.get_cmap(cmap)(i) for i in range(len(cats))]
                colormap = {cat:color for cat, color in
                                    zip(cats, cmap)}
            else:
                colormap = cmap
            color = [colormap[c] for c in d[color_by]]
        else:
            color = None
        if ax is None:
            fig, ax = pylab.subplots(figsize=(9,9))
            just_ax = False
        else:
            fig = ax.figure
            just_ax = True
        # The points
        ax.scatter(
            d.std_male_coeff,
            d.std_female_coeff,
            label="phenotypes",
            s=p_to_size(numpy.minimum(d.p_male, d.p_female)),
            c=color)
        ax.spines['bottom'].set_color(None)
        ax.spines['top'].set_color(None)
        ax.spines['left'].set_color(None)
        ax.spines['right'].set_color(None)
        ax.axvline(c="k", lw=1)
        ax.axhline(c="k", lw=1)
        if labels:
            #ax.set_title("Effect sizes by sex\nAmong signifcant associations")
            ax.set_xlabel("Effect size in males")
            ax.set_ylabel("Effect size in females")
            bbox = {'facecolor': (1,1,1,0.8), 'edgecolor':(0,0,0,0)}
            ax.annotate("Male Effect Larger", xy=(0.8*lim,0), ha="center", bbox=bbox, zorder=3)
            ax.annotate("Male Effect Larger", xy=(-0.8*lim,0), ha="center", bbox=bbox, zorder=3)
            ax.annotate("Female Effect Larger", xy=(0,0.8*lim), ha="center", bbox=bbox, zorder=3)
            ax.annotate("Female Effect Larger", xy=(0,-0.5*lim), ha="center", bbox=bbox, zorder=3)
        ax.set_aspect("equal")
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        # Diagonal y=x line
        bound = max(abs(numpy.min([ax.get_xlim(), ax.get_ylim()])),
                    numpy.max([ax.get_xlim(), ax.get_ylim()]))
        diag = numpy.array([-bound, bound])
        ax.plot(diag, diag, linestyle="--", c='k', zorder=-1, label="diagonal")
        ax.plot(diag, -diag, linestyle="--", c='k', zorder=-1, label="diagonal")
        if color_by is not None and legend:
            if just_ax:
                util.legend_from_colormap(ax, colormap, ncol=2, fontsize="small", names=names)
            else:
                util.legend_from_colormap(fig, colormap, ncol=2, fontsize="small", names=names)
        return fig, ax

    def age_effect_plot(self, d, legend=True, labels=True, color_by="phecode_category", cmap="Dark2", lim=0.45, ax=None, names=None):
        if ax is None:
            fig, ax = pylab.subplots(figsize=(9,9))
            just_ax = False
        else:
            fig = ax.figure
            just_ax = True
        if color_by == "phecode_category":
            colormap = self.colormaps['phecode_cat']
            color = [colormap[c] for c in d[color_by]]
        elif color_by is not None:
            if type(cmap) == str:
                cats = d[color_by].unique()
                if cmap == "rainbow":
                    cmap = [pylab.get_cmap("rainbow")(i) for i in numpy.arange(len(cats))/len(cats)]
                else:
                    cmap = [pylab.get_cmap(cmap)(i) for i in range(len(cats))]
                colormap = {cat:color for cat, color in
                                    zip(cats, cmap)}
            else:
                colormap = cmap
            color = [colormap[c] for c in d[color_by]]
        else:
            color = None
        color = [colormap[c] for c in d[color_by]]
        # The points
        ax.scatter(
            d.age_55_std_effect,
            d.age_70_std_effect,
            s=p_to_size(numpy.minimum(d.p_overall, d.p_age)),
            c=color)
        ax.spines['bottom'].set_color(None)
        ax.spines['top'].set_color(None)
        ax.spines['left'].set_color(None)
        ax.spines['right'].set_color(None)
        ax.axvline(c="k", lw=1)
        ax.axhline(c="k", lw=1)
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        #ax.set_xticks(numpy.linspace(-0.4,0.4,11))
        #ax.set_yticks(numpy.linspace(-0.4,0.4,11))
        # Diagonal y=x line
        bound = max(abs(numpy.min([ax.get_xlim(), ax.get_ylim()])),
                    numpy.max([ax.get_xlim(), ax.get_ylim()]))
        diag = numpy.array([-bound, bound])
        ax.plot(diag, diag, linestyle="--", c='k', zorder=-1, label="diagonal", linewidth=1)
        ax.plot(diag, -diag, linestyle="--", c='k', zorder=-1, label="diagonal", linewidth=1)
        ax.set_aspect("equal")
        if labels:
            #ax.set_title("Effect sizes by age\nAmong signifcant associations")
            ax.set_xlabel("Effect size at 55")
            ax.set_ylabel("Effect size at 70")
            bbox = {'facecolor': (1,1,1,0.8), 'edgecolor':(0,0,0,0)}
            ax.annotate("Age 55 Effect Larger", xy=(0.8*lim,0), ha="center", bbox=bbox, zorder=3)
            ax.annotate("Age 55 Effect Larger", xy=(-0.8*lim,0), ha="center", bbox=bbox, zorder=3)
            ax.annotate("Age 70 Effect Larger", xy=(0,0.8*lim), ha="center", bbox=bbox, zorder=3)
            ax.annotate("Age 70 Effect Larger", xy=(0,-0.8*lim), ha="center", bbox=bbox, zorder=3)
            ax.annotate("Equal Effects", xy=(0.8*lim,0.8*lim), ha="center", va="center", bbox=bbox, zorder=3, rotation=45)
            ax.annotate("Opposite Effects", xy=(0.8*lim,-0.8*lim), ha="center", va="center", bbox=bbox, zorder=3, rotation=-45)
        if legend:
            if just_ax:
                util.legend_from_colormap(ax, colormap, ncol=2, fontsize="small", names=names, loc="upper left")
            else:
                util.legend_from_colormap(fig, colormap, ncol=2, fontsize="small", names=names, loc="upper left")
        return fig,ax

    # Fancy style plot
    # Only really works for highly abundant phenotypes like hypertension (401)
    def fancy_case_control_plot(self, data, code, var="acceleration_RA", normalize=False, confidence_interval=False, rescale=True, annotate=False):
        CONTROL_COLOR = "teal"
        CASE_COLOR = "orange"
        UNCERTAIN_COLOR = (0.8, 0.8, 0.8)

        case = data[code] == True
        config = plot_config[var]
        xbottom = config['xbottom']
        xtop = config['xtop']
        point_width = config['point_width']
        bandwidth = config['bandwidth']
        eval_x = numpy.linspace(xbottom, xtop, int(0.5/point_width + 1))

        case_scaling = (case).sum() * point_width if rescale else 1
        control_scaling = (~case).sum() * point_width if rescale else 1
        case_avg = data[var][case].mean()
        control_avg = data[var][~case].mean()
        total_incidence = case.sum()/len(case)

        def densities_and_incidence(data):
            case_density = gaussian_kde(data[var][case], bw_method=bandwidth)(eval_x) * case_scaling
            control_density = gaussian_kde(data[var][~case], bw_method=bandwidth)(eval_x) * control_scaling
            incidence = case_density / (control_density  + case_density)
            return case_density, control_density, incidence
        
        case_density, control_density, incidence = densities_and_incidence(data)

        if confidence_interval:
            N = 40
            incidences = numpy.empty((len(eval_x), N))
            for i in range(N):
                sample = data.sample(len(data), replace=True)
                _, _, incidence = densities_and_incidence(sample)
                incidences[:,i] = incidence
            incidences = numpy.sort(incidences, axis=1)
            lower_bound = incidences[:,0]
            upper_bound = incidences[:,-1]
            middle = incidences[:,incidences.shape[1]//2]

        fig, (ax1,ax3,ax2) = pylab.subplots(nrows=3, sharex=True,
                                            gridspec_kw = {"hspace":0.1,
                                                        "height_ratios":[0.2,0.2,0.6]})

        # Plot the data
        ax1.fill_between(eval_x, 0, control_density, color=CONTROL_COLOR)
        ax3.fill_between(eval_x, 0, case_density, color=CASE_COLOR)
        if confidence_interval:
            ax2.fill_between(eval_x, lower_bound, middle, color='lightgray')
            ax2.fill_between(eval_x, middle, upper_bound, color='lightgray')
        ax2.plot(eval_x, middle, color='k')

        # Plot avgs
        ax1.axvline(control_avg, c='k', linestyle="--")
        ax3.axvline(case_avg, c='k', linestyle="--")
        ax2.axhline(total_incidence, c='k', linestyle="--")

        # Label plot
        ax1.set_ylabel(f"Controls\nN={(~case).sum()}")
        ax2.set_ylabel(f"Prevalence\n(overall={total_incidence:0.1%})")
        ax3.set_ylabel(f"Cases\nN={case.sum()}") 
        ax2.set_xlabel(config['label'])

        ax1.spines['left'].set_visible(False)
        ax1.spines['right'].set_visible(False)
        ax1.spines['top'].set_visible(False)
        ax1.tick_params(bottom=False)
        ax3.tick_params(bottom=False)
        ax1.yaxis.set_ticks([])
        #ax2.xaxis.set_ticks_position('none')
        ax2.yaxis.set_ticks_position('right')
        if not normalize:
            ax2.yaxis.set_ticks([0, 0.25, 0.5, 0.75, 1])
            ax2.yaxis.set_ticklabels(["0%", "25%", "50%", "75%","100%"])
        else:
            ax2.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(xmax=1))
        ax3.spines['left'].set_visible(False)
        ax3.spines['right'].set_visible(False)
        ax3.spines['bottom'].set_visible(False)
        ax3.yaxis.set_ticks([])

        # Set axis limits
        ax1.set_xlim(xbottom, xtop)
        if not normalize:
            max_density = max(numpy.max(case_density), numpy.max(control_density))
            ax1.set_ylim(0, max_density)
            ax3.set_ylim(0, max_density)
            ax2.set_ylim(0, 1)
        else:
            ax1.set_ylim(0)
            ax3.set_ylim(0)
            ax2.set_ylim(0, numpy.minimum(numpy.max(middle)*1.3, 1.0))
        ax3.invert_yaxis()

        if annotate:
            ax1.annotate("Control mean",
                            xy=(control_avg, numpy.max(control_density)/2),
                            xytext=(-50,0),
                            textcoords="offset pixels",
                            ha="right",
                            va="center",
                            arrowprops={"arrowstyle": "->"})
            ax3.annotate("Case mean",
                            xy=(case_avg, numpy.max(control_density)/2),
                            xytext=(-50,0),
                            textcoords="offset pixels",
                            ha="right",
                            va="center",
                            arrowprops={"arrowstyle": "->"})
            ax2.annotate("Overall prevalence",
                            xy=((xtop*0.9 + xbottom*0.1), total_incidence),
                            xytext=(0,25),
                            textcoords="offset pixels",
                            ha="right",
                            va="center",
                            arrowprops={"arrowstyle": "->"},
                            )
            i = len(eval_x)//5
            ax2.annotate("95% confidence interval",
                            xy=(eval_x[i], upper_bound[i]),
                            xytext=(0,25),
                            textcoords="offset pixels",
                            ha="center",
                            va="bottom",
                            arrowprops={"arrowstyle": "->"},#"-[, lengthB=5.0"},
                            )

        try:
            ax1.set_title(self.phecode_info.loc[code].phenotype)
        except KeyError:
            ax1.set_title(code)
        return fig

    def incidence_rate_by_category(self, data, code, categories, var="acceleration_RA", normalize=False, confidence_interval=False, rescale=False):
        # Break up individuals by categorical variable (eg: sex, age bins)
        # and plot the incidence rate of the phecode by the variable
        case = data[code] == True

        config = plot_config[var]
        xbottom = config['xbottom']
        xtop = config['xtop']
        point_width = config['point_width']
        bandwidth = config['bandwidth']
        eval_x = numpy.linspace(xbottom, xtop, int(0.5/point_width + 1))

        case_scaling = (case).sum() * point_width if rescale else  point_width
        control_scaling = (~case).sum() * point_width if rescale else point_width
        case_avg = data[var][case].median()
        control_avg = data[var][~case].median()
        total_incidence = case.sum()/len(case)

        def densities_and_incidence(data):
            case_kde = gaussian_kde(data[var][case], bw_method=bandwidth)
            case_density = case_kde(eval_x) * case_scaling * case_kde.n
            control_kde = gaussian_kde(data[var][~case], bw_method=bandwidth)
            control_density = control_kde(eval_x) * control_scaling * control_kde.n
            if not normalize:
                incidence = case_density / (control_density  + case_density)
            if normalize:
                incidence = case_density / total_incidence / 2 / (control_density + case_density / total_incidence / 2)
            return case_density, control_density, incidence

        lower_bounds = []
        upper_bounds = []
        middle_values = []
        all_categories = data[categories].cat.categories
        for value in all_categories:
            category_data = data[data[categories] == value]

            if confidence_interval:
                N = 40
            else:
                N = 1
            incidences = numpy.empty((len(eval_x), N))
            for i in range(N):
                sample = category_data.sample(len(category_data), replace=True)
                _, _, incidence = densities_and_incidence(sample)
                incidences[:,i] = incidence
            incidences = numpy.sort(incidences, axis=1)
            lower_bound = incidences[:,0]
            upper_bound = incidences[:,-1]
            middle = incidences[:,incidences.shape[1]//2]


            lower_bounds.append(lower_bound)
            upper_bounds.append(upper_bound)
            middle_values.append(middle)

        fig, ax = pylab.subplots()

        # Plot the data
        for lower_bound, upper_bound, middle, cat in zip(lower_bounds, upper_bounds, middle_values, all_categories):
            ax.fill_between(eval_x, lower_bound, upper_bound, color='lightgrey', alpha=0.3, label=None)
            ax.plot(eval_x, middle, label=cat)

        # Plot avgs
        ax.axhline(total_incidence, c='k', linestyle="--")

        # Label plot
        ax.set_ylabel(f"Prevalence\n(overall={total_incidence:0.1%})")
        ax.set_xlabel(config['label'])

        ax.yaxis.set_ticks_position('right')
        ax.yaxis.set_ticks([0, 0.25, 0.5, 0.75, 1])
        ax.yaxis.set_ticklabels(["0%", "25%", "50%", "75%","100%"])

        # Set axis limits
        ax.set_xlim(xbottom, xtop)
        ax.set_ylim(0, 1)
        try:
            ax.set_title(self.phecode_info.loc[code].phenotype + ("\n(normalized)" if normalize else ""))
        except KeyError:
            ax.set_title(code)

        fig.legend()
        return fig

    # By-age plots
    def age_plot(self, data, var, phecode, difference=False):
        CONTROL_COLOR = "teal"
        CASE_COLOR = "orange"
        fig, ax = pylab.subplots()
        age_at_actigraphy = (data.actigraphy_start_date - data.birth_year_dt) / pandas.to_timedelta("1Y")
        eval_x = numpy.arange(numpy.floor(numpy.min(age_at_actigraphy)), numpy.ceil(numpy.max(age_at_actigraphy))+1, 3)
        cases = (data[phecode] == 1)
        controls = (data[phecode] == 0)
        def reg_with_conf_interval(subset):
            main = local_regression(age_at_actigraphy.iloc[subset], data.iloc[subset][var], eval_x, bw=3.0)
            samples = []
            for _ in range(40):
                s = numpy.random.choice(subset, size=len(subset))
                samples.append(local_regression(age_at_actigraphy.iloc[s], data.iloc[s][var], eval_x, bw=3.0))
            samples = numpy.array(samples)
            samples = numpy.sort(samples, axis=0)
            bottom = samples[0,:]
            top = samples[-1,:]
            return main, bottom, top
        case_mid, case_bottom, case_top = reg_with_conf_interval(numpy.where(cases)[0])
        control_mid, control_bottom, control_top = reg_with_conf_interval(numpy.where(controls)[0])
        if difference == False:
            ax.plot(eval_x, case_mid, label="cases", c=CASE_COLOR)
            ax.plot(eval_x, control_mid, label="controls", c=CONTROL_COLOR)
            ax.fill_between(eval_x, case_bottom, case_top, color=CASE_COLOR, alpha=0.5)
            ax.fill_between(eval_x, control_bottom, control_top, color=CONTROL_COLOR, alpha=0.5)
        else:
            ax.plot(eval_x, case_mid - control_mid, c="k")
            ax.fill_between(eval_x, case_bottom - control_mid, case_top - case_bottom, color="k", alpha=0.5)
        ax.set_xlabel("Age")
        ax.set_ylabel(var)
        ax.set_title(self.phecode_info.loc[phecode].phenotype)
        fig.legend()
        return fig, ax

    def survival_curve(self, data, ax, **kwargs):
        start_date = pandas.to_datetime(data.date_of_death).min()
        N = len(data)
        data = data[~data.date_of_death.isna()]
        date = pandas.to_datetime(data.date_of_death).sort_values()
        date_ = pandas.concat((pandas.Series([start_date]), date))
        ax.step(date_,
                (1 - numpy.concatenate(([0], numpy.arange(len(data))))/N)*100,
                where='post',
                **kwargs)

    def quintile_survival_plot(self, data, var, var_label=None, ax=None):
        if var_label is None:
            var_label = var
        quintiles = pandas.qcut(data[var], 5)
        if ax is None:
            fig, ax = pylab.subplots(figsize=(8,6))
            just_ax = False
        else:
            fig = ax.figure
            just_ax = True
        years = range(2015, 2022)
        yearly_counts = {} # Number of deaths so far
        for quintile, label in list(zip(quintiles.cat.categories, quintile_labels))[::-1]:
            quintile_data = data[quintiles == quintile]
            self.survival_curve(quintile_data, ax, label= label + " Quintile")
            yearly_counts[label] = {year: (pandas.to_datetime(quintile_data.date_of_death) < pandas.to_datetime(str(year) + "-01-01")).sum()
                                for year in years }

        ax.set_title(f"Survival by {var_label}")
        ax.set_ylabel("Survival Probability")
        ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter())
        ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%Y"))
        #ax2 = ax.twinx() # The right-hand side axis label
        #scale = len(data)/5
        #ax2.set_ylim(ax.get_ylim()[0]*scale, ax.get_ylim()[1]*scale)
        #ax2.set_ylabel("Participants")
        if not just_ax:
            fig.legend(loc=(0.15,0.15))
            fig.tight_layout()
        return fig, pandas.DataFrame(yearly_counts)

    def quintile_survival_plot_by_cat(self, data, var, cat_var, var_label=None):
        cats = data[cat_var].astype("category").cat.categories
        quintiles = pandas.qcut(data[var], 5)
        ns = data[cat_var].value_counts()
        cats = cats[ns[cats] > 500] # Drops categories that don't have enough people
        years = [pandas.to_datetime(str(year) + "-01-01") for year in range(2015, 2022)]
        yearly_counts = {cat:{} for cat in cats}
        fig, axes = pylab.subplots(ncols=len(cats), sharex=True, sharey=True, figsize=(10,5))
        for ax, cat in zip(axes.flatten(), cats):
            d = data[data[cat_var] == cat]
            q =  d.index.map(quintiles)
            for quintile, label in list(zip(quintiles.cat.categories, quintile_labels))[::-1]:
                quintile_d = d[q == quintile]
                quintile_label = (label + " Quintile") if ax == axes.flatten()[0] else None # Only label first
                self.survival_curve(quintile_d, ax, label=quintile_label)
                yearly_counts[cat][label] = {year: (pandas.to_datetime(quintile_d.date_of_death) < year).mean()
                                                for year in years }
            ax.set_title(cat)
            ax.set_ylabel("Survival Probability")
            ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter())
            ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%Y"))
        fig.legend()
        fig.suptitle(var_label)
        return fig, {cat: pandas.DataFrame(counts) for cat,counts in yearly_counts.items()}

    def categorical_survival_plot(self, data, var, var_label=None, min_N=None):
        if var_label is None:
            var_label = var
        fig, ax = pylab.subplots(figsize=(8,6))
        value = data[var].astype("category")
        for cat in value.cat.categories:
            d = data[value == cat]
            if min_N is not None and len(d) < min_N:
                continue # Skip this category
            self.survival_curve(d, ax, label= cat)
        fig.legend(loc=(0.15,0.15))
        ax.set_ylabel("Survival Probability")
        ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter())
        fig.tight_layout()
        return fig

    ## Investigate phenotypes by diagnosis age
    def plot_by_diagnosis_date(self, data, ICD10_codes, phecode, phenotype_name, icd10_entries, phecode_tests):
        # gather icd10 entries by the date of first diagnosis
        Q_CUTOFF = 0.01
        NUM_GROUPS = 20 # number of equal-sized groupsgroups to break the diagnosis dates into
        GROUPS_PER_POINT = 5 # Number of those groups to use at once: use 1 for non-overlapping, higher numbers to get overlap of adjacent
        in_icd10_codes = numpy.any([icd10_entries.ICD10.str.startswith(code) for code in ICD10_codes], axis=0)
        diag_dates = icd10_entries[in_icd10_codes].groupby("ID").first_date.first()
        activity_vars_to_test = phecode_tests[(phecode_tests.phecode == phecode) & (phecode_tests.q < Q_CUTOFF)].activity_var.unique()

        date_data = data.copy()
        date_data['diag_date'] = pandas.to_datetime(pandas.Series(data.index.map(diag_dates), data.index))
        date_data['diag_date_minus_actigraphy'] = (date_data.diag_date - date_data.actigraphy_start_date) / pandas.to_timedelta("1Y")
        date_data['diag_date_group'] = pandas.qcut(date_data.diag_date_minus_actigraphy, NUM_GROUPS)
        by_date_list = []
        for i, date_group in enumerate(date_data.diag_date_group.cat.categories):
            date_groups = date_data.diag_date_group.cat.categories[i:i+GROUPS_PER_POINT]
            if len(date_groups) < GROUPS_PER_POINT:
                continue # Insufficient groups, reached end of the study
            mid_point = date_groups[GROUPS_PER_POINT//2].mid
            date_data['in_group'] = (date_data.diag_date_group.isin(date_groups)).astype(int)
            # Compare only to people who do not have the phecode from any source
            date_data['use'] = date_data.in_group | (date_data[phecode] == 0)
            for activity_var in activity_vars_to_test:
                if activity_var.startswith("self_report"):
                    continue # Not based off actigraphy, skip
                covariate_formula = ' + '.join(c for c in covariates if c != 'sex')
                fit = OLS(f"{activity_var} ~ in_group + sex * ({covariate_formula})",
                            data=date_data,
                            subset=date_data.use)
                p = fit.pvalues["in_group"]
                coeff = fit.params["in_group"]
                std_effect = coeff / data[activity_var].std()
                N_cases = data.loc[~data[activity_var].isna(), phecode].sum()
                by_date_list.append({
                                        "group": date_group,
                                        "group_mid": mid_point,
                                        "activity_var": activity_var,
                                        "p": p,
                                        "coeff": coeff,
                                        "std_effect": std_effect,
                                        "N_cases": N_cases,
                                    })
        by_date = pandas.DataFrame(by_date_list)

        fig, ax = pylab.subplots(figsize=(10,7))
        def plot_var(data):
            data = data.sort_values(by="group_mid")
            cat = self.activity_variable_descriptions.Subcategory[data.activity_var.iloc[0]]
            color = self.colormaps['actigraphy_subcat'][cat]
            #cat = activity_variable_descriptions.Category[data.activity_var.iloc[0]]
            #color = color_by_actigraphy_cat[cat]
            ax.plot(-data.group_mid,
                    data.std_effect.abs(),
                    #-numpy.log10(data.p),
                    c = color, label=cat,
                    linewidth=3)
        ax.set_xlabel("Years since diagnosis")
        ax.set_ylabel("Effect size")
        ax.set_title(phenotype_name)
        by_date.groupby('activity_var').apply(plot_var)
        util.legend_from_colormap(fig, self.colormaps['actigraphy_subcat'], fontsize="small", loc="upper left")
        return fig, ax, by_date

    def circadian_component_plot(self, data, phenotypes, total_phenotypes, quantitative=False):
        ## Plot comparison of circadian versus sleep and physical activity
        colors = {c: self.colormaps['actigraphy_cat'][c] if c in self.colormaps['actigraphy_cat'] else (0.3, 0.3, 0.3, 1.0)
                    for c in ["Circadianness", "Physical activity", "Sleep", "Overall"]}
        colors_list = list(colors.values())
        fig, (ax1, ax2) = pylab.subplots(figsize=(8,9), ncols=2, sharey=True)
        yticks = {}
        pvalues = []
        ranks = []
        for rank, phenotype in enumerate(phenotypes):
            phenotype_name = self.phecode_info.loc[phenotype].phenotype if not quantitative else self.quantitative_variable_descriptions.loc[phenotype].Name
            if rank > 20:
                continue # Skip all but the most significant

            # or use hand-selected variables for all
            top_circ = "temp_RA"
            top_sleep = "main_sleep_ratio_mean"
            top_physical = "acceleration_overall"

            covariate_formula = ' + '.join(c for c in covariates)
            if not quantitative:
                # Logistic model
                results = smf.logit(f"Q({phenotype}) ~ {top_circ} + {top_sleep} + {top_physical} + {covariate_formula}", data=data).fit()
                marginals = results.get_margeff()
                ps = pandas.Series(results.pvalues, index=results.model.exog_names)[[top_circ, top_physical, top_sleep]]
                overall_p = results.f_test(f"{top_circ} = 0, {top_sleep} = 0, {top_physical} = 0").pvalue
                effs = pandas.Series(marginals.margeff, index=results.model.exog_names[1:])[[top_circ, top_physical, top_sleep]].abs()
                effs *= data[effs.index].std() # Standardize by the actigraphy variables used
                effs /= data[phenotype].mean() # Standardize by the overall prevalence
                ses = pandas.Series(marginals.margeff_se, index=results.model.exog_names[1:])[[top_circ, top_physical, top_sleep]]
                ses *= data[effs.index].std() # Standardized SEs too
                ses /= data[phenotype].mean()
            else:
                results = smf.ols(f"{phenotype} ~ {top_circ} + {top_sleep} + {top_physical} + {covariate_formula}", data=data).fit()
                ps = results.pvalues[[top_circ, top_physical, top_sleep]]
                overall_p = results.f_test(f"{top_circ} = 0, {top_sleep} = 0, {top_physical} = 0").pvalue
                effs = results.params[[top_circ, top_physical, top_sleep]].abs()
                effs *= data[effs.index].std() # Standardize by the actigraphy variables used
                effs /= data[phenotype].std() # Standardize by the phenotype variance
                ses = results.bse[[top_circ, top_physical, top_sleep]].abs()
                ses *= data[effs.index].std() # Standardize by the actigraphy variables used
                ses /= data[phenotype].std() # Standardize by the phenotype variance

            height = 0.15 #Height of the bar plots (1 unit is the spacing between variables)
            ys = numpy.linspace(rank-height*1.5, rank+height*1.5, 4)
            ax2.scatter(effs, ys[:3], c=colors_list[:3])
            for eff, se, y, c, p in zip(effs, ses, ys, colors_list, ps):
                ax2.plot([eff - 1.96*se, eff + 1.96*se], [y, y], c=c) # Draw effect sizes
            for y, c, p in zip(ys, colors_list, numpy.concatenate((ps, [overall_p]))):
                ax1.barh([y], height=height, width=[-numpy.log10(p)], color=c) # Draw p value bars

            yticks[rank] = util.wrap(phenotype_name, 30)
            pvalues.append(ps[top_circ])
            ranks.append(rank)
        # Compute an FDR = 0.05 cutoff - but we don't actually compute a p for every phenotype
        # so we will assume worst case, all others ps are 1
        #pvalues = numpy.array(pvalues + [1]*(phenotype_tests.phenotype.nunique() - len(pvalues)))
        pvalues = numpy.concatenate([pvalues, numpy.random.uniform(size=(total_phenotypes - len(pvalues)))])
        qvalues = util.BH_FDR(pvalues)
        qvalue_dict = {rank: q for rank, q in zip(ranks, qvalues[:len(ranks)])}
        # Cut off half way between pvalues of worst gene passing FDR < 0.05 and best gene not passing
        pvalue_cutoff = 0.5*pvalues[qvalues < 0.05].max() + 0.5*pvalues[qvalues >= 0.05].min()
        ax1.axvline(-numpy.log10(pvalue_cutoff), linestyle="--", color="k")
        ax1.set_yticks(list(yticks.keys()))
        ax1.set_yticklabels([name if qvalue_dict[rank] >= 0.05 else name + " (*)"
                                for rank, name in yticks.items()])
        ax1.set_xlabel("-log10 p-value")
        ax1.set_xlim(left=0, right=min(ax1.get_xlim()[1], 40)) # Cap log p-values at 40
        ax2.set_xlabel("Standardized Effect Size")
        ax2.axvline(0, linestyle="-", color="k", linewidth=1)
        ax2.set_xlim(left=0)
        util.legend_from_colormap(fig, colors)
        ax1.margins(0.5, 0.02)
        fig.tight_layout()

        return fig, (ax1, ax2)