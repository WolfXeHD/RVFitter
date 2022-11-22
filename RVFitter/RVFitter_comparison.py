import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sigfig


class RVFitter_comparison(object):
    """Docstring for RVFitter_comparison. """

    def __init__(self, list_of_fitters, output_folder='.'):
        self.list_of_fitters = list_of_fitters
        stars = [fitter.star for fitter in list_of_fitters]
        # check if all elements of stars are equal
        if len(set(stars)) != 1:
            raise ValueError("All stars must be equal")
        self.star = stars[0]
        self.output_folder = output_folder
        self.df = None

    def create_overview_df(self):
        df = pd.DataFrame()
        for fitter in self.list_of_fitters:
            if fitter.constraints_applied:
                suffix = fitter.shape_profile + "_with_constraints"
            else:
                suffix = fitter.shape_profile + "_without_constraints"
            df = self.add_parameters_to_df(df=df,
                                           fitter=fitter,
                                           suffix="_" + suffix)
        self.df = df

    def add_parameters_to_df(self, df, fitter, suffix):
        l_amp = []
        l_cen = []
        l_sig = []
        l_error_amp = []
        l_error_cen = []
        l_error_sig = []
        for parameter in fitter.df["parameters"]:
            amp = parameter["amp"]
            cen = parameter["cen"]
            sig = parameter["sig"]

            l_amp.append(fitter.result.params[amp].value)
            l_cen.append(fitter.result.params[cen].value)
            l_sig.append(fitter.result.params[sig].value)

            l_error_amp.append(fitter.result.params[amp].stderr)
            l_error_cen.append(fitter.result.params[cen].stderr)
            l_error_sig.append(fitter.result.params[sig].stderr)

        this_df = pd.DataFrame({
            "starname":
                fitter.df["starname"],
            "date" + suffix:
                fitter.df["date"],
            "line_profile" + suffix:
                fitter.df["line_profile"],
            "amp" + suffix:
                l_amp,
            "cen" + suffix:
                l_cen,
            "sig" + suffix:
                l_sig,
            "error_amp" + suffix:
                l_error_amp,
            "error_cen" + suffix:
                l_error_cen,
            "error_sig" + suffix:
                l_error_sig,
            "line_name" + suffix:
                fitter.df["line_name"]
        })

        df = pd.concat([df, this_df], axis=1)

        for col in df.columns:
            if "date" in col:
                check = df[col] == this_df["date" + suffix]
                if np.sum(check) != len(check):
                    raise Exception("Error: date mismatch")
                df["date"] = df[col]
            if "line_profile" in col:
                check = df[col] == this_df["line_profile" + suffix]
                if np.sum(check) != len(check):
                    raise Exception("Error: line_profile mismatch")
                df["line_profile"] = df[col]
            if "line_name" in col:
                check = df[col] == this_df["line_name" + suffix]
                if np.sum(check) != len(check):
                    raise Exception("Error: line_name mismatch")
                df["line_name"] = df[col]

        return df

    def get_df_for_latex(self, variable, add_starname=False):
        if variable not in ["amp", "cen", "sig"]:
            raise Exception("Error: variable not in ['amp', 'cen', 'sig']")

        columns_constraints = [col for col in self.df.columns if (variable in col)
                               and ("with_constraints" in col)]
        columns_no_constraints = [col for col in self.df.columns if (variable in col)
                                  and ("without_constraints" in col) and ("error" not in col)]
        errors_no_constraints = [col for col in self.df.columns if (variable in col)
                                 and ("without_constraints" in col) and ("error" in col)]
        dates = self.df["date"].unique()

        df_for_comparison = pd.DataFrame()
        new_df = {}
        for date in dates:
            new_df['date'] = date
            temp_df = self.df[self.df["date"] == date]
            if add_starname:
                new_df["starname"] = self.star
            for col, ecol in zip(columns_no_constraints, errors_no_constraints):
                new_df[col] = temp_df[col].median()
                new_df[ecol] = temp_df[ecol].std()
            for col in columns_constraints:
                new_df[col] = temp_df[col].median()

            df_for_comparison = df_for_comparison.append(
                new_df, ignore_index=True)
        if add_starname:
            l_names = ["starname", "date"]
        else:
            l_names = ["date"]

        for shape_profile in ["gaussian", "lorentzian", "voigt"]:
            for constraints_applied in [True, False]:
                this_fitter = self.get_fitter(
                    shape_profile=shape_profile, constraints_applied=constraints_applied)
                if this_fitter is None:
                    continue
                else:
                    subscript = this_fitter.subscript
                    name = "$v_{" + subscript + "}$ (km/s)"
                    l_names.append(name)
                    df_for_comparison[name] = df_for_comparison.apply(lambda x: RVFitter_comparison.adjust_table(x, variable=variable,
                                                                                                                 shape_profile=shape_profile,
                                                                                                                 constraint=constraints_applied,
                                                                                                                 apply_cutoff=False), axis=1)
        return df_for_comparison[l_names]

    def get_fitter(self, shape_profile, constraints_applied):
        for this_fitter in self.list_of_fitters:
            if (this_fitter.shape_profile == shape_profile) and (this_fitter.constraints_applied == constraints_applied):
                return this_fitter
        return None

    def write_overview_table(self, variable, table_name=None):
        if table_name is None:
            table_name = os.path.join(
                self.output_folder, "results_" + self.star + "_" + variable + ".tex")
        this_df = self.get_df_for_latex(variable=variable)
        RVFitter_comparison.write_df_to_table(input_df=this_df,
                                              filename=table_name)

    @staticmethod
    def write_df_to_table(input_df, filename):
        table_data = input_df.to_latex(
            escape=False, index=False, column_format="c" * len(input_df.columns))
        with open(filename, "w", encoding='utf-8') as f:
            f.write(table_data)
        print(filename, "written.")

    @staticmethod
    def adjust_table(x, variable, shape_profile, constraint, apply_cutoff=False):
        if constraint:
            constraint_string = "with"
        else:
            constraint_string = "without"

        error_var = f"error_{variable}_{shape_profile}_{constraint_string}_constraints"
        value_var = f"{variable}_{shape_profile}_{constraint_string}_constraints"

        if np.isnan(x[value_var]):
            return "$ NaN $"
        if np.isnan(x[error_var]):
            return "$ {value:.0f} \pm NaN $".format(value=sigfig.round(x[value_var], decimals=1))

        if apply_cutoff:
            val = sigfig.round(x[value_var], x[error_var],
                               cutoff=1000, separation=' \pm ')
        else:
            val = sigfig.round(x[value_var], x[error_var], separation=' \pm ')
        return "$" + val + "$"

    def compare_fit_results_1D(self, variable, fig_and_ax=None):
        if variable not in ["amp", "cen", "sig"]:
            raise Exception("Error: variable not in ['amp', 'cen', 'sig']")

        columns_to_plot = [col for col in self.df.columns if (
            variable in col) and ("error" not in col)]

        dates = self.df["date"].unique()

        if fig_and_ax is None:
            if len(dates) > 1:
                fig, axes = plt.subplots(len(dates), 1, sharex=True)
            else:
                fig, axes = plt.subplots(1, 1)
                axes = np.array([axes])
        else:
            fig, axes = fig_and_ax
        fig.suptitle('Comparison of fit results for ' + self.star)
        plt.subplots_adjust(hspace=0.3, bottom=0.25, left=0.15)

        for ax, date in zip(axes, dates):
            ax.set_title(date, fontsize=10)
            this_df = self.df[self.df["date"] == date]
            #sort dataframe by column
            this_df["line_profile"] = this_df["line_profile"].astype(int)
            this_df = this_df.sort_values(by='line_profile')
            labels = this_df["line_name"].values
            this_df["labels"] = this_df.apply(
                lambda x: x["line_name"] + " " + str(x["line_profile"]), axis=1)
            for column in columns_to_plot:
                if None not in this_df['error_' + column].values:
                    p = ax.errorbar(this_df[column].values, list(range(len(this_df[column]))), xerr=this_df['error_' + column].values,
                                    fmt='o', label=column, capsize=2)
                else:
                    p = ax.errorbar(this_df[column].values, list(
                        range(len(this_df[column]))), fmt='o', label=column)
                color = p[0].get_color()
                ax.axvline(np.median(this_df[column]),
                           color=color, linestyle='-')
                ax.axvspan(np.median(this_df[column]) - np.std(this_df[column])/2.,
                           np.median(this_df[column]) +
                           np.std(this_df[column])/2.,
                           color=color, alpha=0.2)

            ax.set_yticks(list(range(len(this_df))))
            ax.set_yticklabels(this_df["labels"])
        handles, labels = axes[-1].get_legend_handles_labels()
        if variable == 'cen':
            axes[-1].set_xlabel('Velocity (km/s)')
            filename = "compare_results_velocity.png"
        else:
            axes[-1].set_xlabel(variable)
            filename = "compare_results_{variable}.png"

        fig.legend(handles, labels, ncol=2, loc='lower center')

        this_filename = os.path.join(self.output_folder, filename)
        fig.savefig(this_filename)
        print(this_filename, "saved.")
        plt.close(fig)

    def plot_fits_and_residuals(self,
                                color_dict=None,
                                figname=None):

        if color_dict is None:
            color_dict={0: "red", 1: "blue", 2: "green",
                        3: "orange", 4: "purple", 5: "black"},
        if figname is None:
            figname = os.path.join(
                self.output_folder, "fits_and_residuals.png")
        fig, ax_dict = self.list_of_fitters[0].get_fig_and_ax_dict()

        for idx, this_fitter in enumerate(self.list_of_fitters):
            if idx == 0:
                this_fitter.plot_data_and_residuals(fig=fig, ax_dict=ax_dict)
            this_fitter.plot_fit_and_residuals(fig=fig,
                                               ax_dict=ax_dict,
                                               add_legend_label=False,
                                               add_legend_model=True,
                                               plot_dict={"zorder": 2.5,
                                                          "markersize": "1",
                                                          "color": color_dict[idx],
                                                          },
                                               plot_dict_res={
                                                   "marker": ".",
                                                   "linestyle": "None",
                                                   "color": color_dict[idx],
                                                   "markersize": "2"})
        handles, labels = ax_dict[list(ax_dict.items())[
            0][0]].get_legend_handles_labels()
        labels = [this_fitter.label for this_fitter in self.list_of_fitters]
        fig.legend(handles, labels, ncol=2, loc='lower center')

        fig.savefig(figname)
        print(figname, "saved.")
        # plt.close(fig)

    def plot_fits_and_data(self, color_dict=None, filename=None):
        if color_dict is None:
            color_dict = {0: "red", 1: "blue", 2: "green", 3: "orange"}
        fig, axes = self.list_of_fitters[0].get_fig_and_axes()
        for idx, this_fitter in enumerate(self.list_of_fitters):
            if idx == 0:
                this_fitter.plot_data(fig=fig, axes=axes)
            this_fitter.plot_fit(fig=fig, axes=axes,
                                 plot_dict={"zorder": 2.5, "color": color_dict[idx], "label": this_fitter.label})
        handles, labels = axes[-1, -1].get_legend_handles_labels()
        fig.legend(handles, labels, ncol=2, loc='lower center')
        if filename is not None:
            fig.savefig(filename)
