import pandas as pd

def read_line_list(filename):
    with open(filename, "r") as f:
        data = f.read()

    lines = []
    for line in data.splitlines():
        if line.startswith("#"):
            continue
        elif line.strip() == "":
            continue
        else:
            try:
                splitted_line = line.split(" ")
                this_dict = {}
                this_dict["line_name"] = splitted_line[0]
                this_dict["line_profile"] = float(splitted_line[1])
                this_dict["line_width"] = float(splitted_line[2])
                lines.append(this_dict)
            except:
                raise Exception(f"Could not parse line {line}")
    return lines

def manipulate_df_by_line_list(df, line_list):
    if line_list is not None:
        l_dfs = []
        for line in line_list:
            masker1 = df["line_name"] == line["line_name"]
            masker2 = df["line_profile"] == line["line_profile"]
            full_masker = masker1 & masker2
            this_df = df[full_masker]
            if len(this_df) == 0:
                df_name = df.query(
                    f"(line_name == \"{line['line_name']}\")"
                )

                if len(df_name) == 0:
                    available_lines = df["line_name"].unique()
                    raise Exception(
                        f"Line {line['line_name']} not found in dataframe. Available lines are: {available_lines}")
                else:
                    available_line_profiles = df_name["line_profile"].unique()
                    raise Exception("No data for line {} with profile {}. Available profiles are {}".format(
                        line["line_name"], line["line_profile"], available_line_profiles))

            l_dfs.append(this_df)
        df = pd.concat(l_dfs)
    else:
        print("No line list provided. Using all lines in dataframe.")
    return df
