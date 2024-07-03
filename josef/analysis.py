# Python modules
import sys
import json
import yaml
import math
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats

HERE = Path(__file__).parent
sys.path.append(str(HERE.parent))

sns.set_theme(context="talk", style="ticks")

# 'viral' as package
from viral.single_session import (
    load_data,
    plot_rewarded_vs_unrewarded_licking,
    plot_speed,
    remove_bad_trials,
    summarise_trial,
)
from viral.multiple_sessions import (
    plot_performance_across_days,
    learning_metric,
    speed_difference,
    licking_difference,
)

from viral.gsheets_importer import gsheet2df
from viral.models import SessionSummary, MouseSummary, SessionSummaryCount
from josef.constants import HERE, DATA_PATH, SPREADSHEET_ID

#############################################################
# SETTINGS CODE #
# JB: Parsing the config.yaml file
with open(HERE.parent / "josef" / "config.yaml", "r") as file:
    settings = yaml.safe_load(file)

# JB: Setting the minimal number of trials a session has to have to be included into the analysis
SESSION_MIN_TRIALS = settings["SESSION_MIN_TRIALS"]

MICE_LIST = settings["MICE_LIST"]
print(f"NOTE: Analysing -> {MICE_LIST}")

# Counting the number of subjects for correct plots
mice_count = len(MICE_LIST)

MICE_OPTIMAL_WEIGHT_LOSS = settings["MICE_OPTIMAL_WEIGHT_LOSS"]

#############################################################
# CACHE CODE #
# JB: Here goes the code for caching mouse analysis data
def cache_mouse(mouse_name: str) -> None:
    # JB: This function takes the mouse_name, to then take the sessions from the worksheet
    # It then retrieves the trial data plus their weight at the day of the session and summarises the session
    # It calculates the weight changes
    # The name of the mouse and the summaries of the sessions get dumped into a .json file
    print(f"NOTE: Starting to cache mouse '{mouse_name}'.")
    metadata = gsheet2df(SPREADSHEET_ID, mouse_name, 1)
    session_summaries = []
    baseline_weight = float(get_baseline_weight(mouse_name))
    for _, row in metadata.iterrows():
        trials = []
        if "learning day" not in row["Type"].lower():
                continue
        
        # JB: Here, it gets tricky. We want to deal with "1 + 2" sessions, and jibberish sessions.
        if "+" in row["Session Number"]:
            # Session splits is a list of the splitted sessions
            session_splits = (row["Session Number"]).split("+")
            session_splits = [session.strip() for session in session_splits]
            # It then retrieves the data of both split sessions and combines the trials
            # Note: trials is the whole list
            print(session_splits)
            for session in session_splits:
                session_number = (
                session
                if len(session) == 3
                else f"00{session}"
                )
                session_path = DATA_PATH / mouse_name / row["Date"] / session_number
                try:
                    trials_split = load_data(session_path)
                    trials_split = remove_bad_trials(trials)
                    trials.extend(trials_split)
                except:
                    print(
                        f"ERROR: ExceptionError: There went something wrong getting the trials."
                    )
                    print(
                        f"Mouse:'{mouse_name}'. Date:'{row["Date"]}'. Session: '{session_number}'."
                    )
        else:
        # Here, we have the standard case, just one session for a day
            session_number = (
                row["Session Number"]
                if len(row["Session Number"]) == 3
                else f"00{row['Session Number']}"
            )

            session_path = DATA_PATH / mouse_name / row["Date"] / session_number

            # JB: These lines get the trial data and raise an ExceptionError if there is an issue getting the trial data
            try:
                trials = load_data(session_path)
                trials = remove_bad_trials(trials)
            except:
                print(
                    f"ERROR: ExceptionError: There went something wrong getting the trials."
                )
                print(
                    f"Mouse:'{mouse_name}'. Date:'{row["Date"]}'. Session: '{session_number}'."
                )

        # JB: Here, we get the weight at the day of the session, the baseline weight to then calculate the relative weight change
        # Initialising baseline_weight before the try block
        try:
            session_weight = float(row["Weight"])
        except:
            print(f"ERROR: ExceptionError: There went something wrong getting 'session_weight'.")
        try:
            weight_change = float(calculate_weight_change(baseline_weight, session_weight))
        except:
            print(f"ERROR: Exception:Error: There went someting wrong getting 'weight_change'.")
        try:
            session_summaries.append(
                    SessionSummary(
                        name=row["Type"],
                        trials=[summarise_trial(trial) for trial in trials],
                        weight=float(session_weight),  # Add your very important weight here
                        weight_change=float(weight_change)
                    )
            )
        except:
             print(f"ERROR: ExceptionError: There went something wrong with creating a 'SessionSummary' for '{row["Type"]}'.")
        
        # JB: Now the cache file gets saved as a .json file.
        with open(
            HERE.parent / "josef" / "data" / "cache" / f"{mouse_name}.json",
            "w",
        ) as f:
            json.dump(
                MouseSummary(
                    sessions=session_summaries,
                    name=mouse_name,
                    baseline_weight=baseline_weight
                ).model_dump(),
                f,
            )
    print(f"NOTE: Successfully cached mouse '{mouse_name}'.")

def load_cache_mouse(mouse_name: str) -> None:
    ## JB: This function takes the mouse name and retrieves the MouseSummary from the cache file
    with open(
        HERE.parent / "josef" / "data" / "cache" / f"{mouse_name}.json", "r"
    ) as f:
        return MouseSummary.model_validate_json(f.read())
    
def cache_weight_changes_all(MICE_LIST: list) -> None:
    # JB: This function takes a list of mice, takes their weight changes and dumps them into a single list
    # It then dumps it into a .json file for further analysis
    weight_changes_all = []
    for mouse_name in MICE_LIST:
        mouse_weight_changes = get_weight_changes(mouse_name)
        mouse_data = [mouse_name, mouse_weight_changes]
        weight_changes_all.append(mouse_data)
    weight_changes_all_df = pd.DataFrame(data = weight_changes_all, columns=["mouse_name", "weight_changes"])

    # JB: This was to test if the dataframe was created correctly
    # datapath = HERE.parent / "josef" / "data" / "weight_changes_all.csv"
    # weight_changes_all_df.to_csv(datapath)

    
    # JB: Now the cache file gets saved as a .json file.
    weight_changes_all_json = weight_changes_all_df.to_json()
    with open(
        HERE.parent / "josef" / "data" / "weight_changes_all.json",
            "w",
        ) as f:
            json.dump(weight_changes_all_json, f)
    print(f"NOTE: Successfully cached 'weight_changes_all'")
    
def list_weight_changes_all():
    # JB: This function takes the cache .json file with all the weight changes and transforms it into a plain list.
    # These lines retrieve the dataframe out of the .json file
    with open(HERE.parent / "josef" / "data" / "weight_changes_all.json") as f:
        weight_changes_all_json = json.load(f)
        weight_changes_all_df = pd.read_json(weight_changes_all_json)

    # Here we get the dataframe and return a plain list of weight_changes
    # First, we get a pandas series and turn it into a list
    weight_changes_all = pd.Series.to_list((weight_changes_all_df["weight_changes"]))
    # Then, we flatten the list (thanks to ChatGPT)
    weight_changes_all = [item for sublist in weight_changes_all for item in sublist]

    return weight_changes_all

def cache_number_of_trials_weights_weight_changes_all(MICE_LIST: list):
    # JB: This function takes a list of mice, takes their weight, weight changes and trials counts
    # It then dumps it into a .json file for further analysis
    number_of_trials_weights_weight_changes_all = []
    for mouse_name in MICE_LIST:
        session_summaries_count = get_number_of_trials_weights_weight_changes(mouse_name)
        number_of_trials_weights_weight_changes_all.append([mouse_name, session_summaries_count])
    number_of_trials_weights_weight_changes_all_df = pd.DataFrame(data = number_of_trials_weights_weight_changes_all, columns=["mouse_name", "weight_trials_data"])

    # JB: Now the cache file gets saved as a .json file.
    number_of_trials_weights_weight_changes_all_json = number_of_trials_weights_weight_changes_all_df.to_json()
    with open(
        HERE.parent / "josef" / "data" / "number_of_trials_weights_weight_changes_all.json",
            "w",
        ) as f:
            json.dump(number_of_trials_weights_weight_changes_all_json, f)
    print(f"NOTE: Successfully cached 'number_of_trials_weights_weight_changes_all_json'")

def load_number_of_trials_weights_weight_changes_all():
    # JB: This function takes the cache .json file with all the weights, weight changes, trials counts and transforms it into a dataframe
    # These lines retrieve the dataframe out of the .json file and returns it
    with open(HERE.parent / "josef" / "data" / "number_of_trials_weights_weight_changes_all.json") as f:
        number_of_trials_weights_weight_changes_all_json = json.load(f)
        number_of_trials_weights_weight_changes_all_df = pd.read_json(number_of_trials_weights_weight_changes_all_json)

    print(type(number_of_trials_weights_weight_changes_all_df))
    return number_of_trials_weights_weight_changes_all_df

#############################################################
# STATISTICS CODE #
## BASIC FUNCTIONS

def get_baseline_weight(mouse_name: str):
    ## JB: This function takes the mouse's name, gets the baseline weight and returns it
    # Remember: The baseline weight is the weight before the beginning of the water restriction
    # Ergo, it is equal to the weight on the day "Take bottle out"
    metadata = gsheet2df(SPREADSHEET_ID, mouse_name, 1)
    # Initialising the value
    baseline_weight = None
    try:
        for _, row in metadata.iterrows():
            if row["Type"] == "Take bottle out":
                baseline_weight = float(row["Weight"])
        return baseline_weight
    except:
        print(f"ERROR: ExceptionError: Could not get 'baseline_weight' for '{mouse_name}'.")

def calculate_weight_change(baseline_weight: float, session_weight: float):
    # JB: This function takes the baseline weight and the session weight to calculate the relative weight change.
    # weight_change = abs((baselineWeight - sessionWeight) / baselineWeight)
    try:
        weight_change = float((session_weight - baseline_weight) / baseline_weight)
        return weight_change
    except:
        print(f"ERROR: ExceptionError: There went something wrong calculating 'weight_change'.")

def get_weight_changes(mouse_name: str):
    ## JB: This function takes the mouse's name, loads the cache, calculates the weight change for each session and then returns a list of weight changes
    mouse = load_cache_mouse(mouse_name)
    # JB: This line states that a sessions has to have more than SESSION_MIN_TRIALS.
    sessions = [session for session in mouse.sessions if len(session.trials) > SESSION_MIN_TRIALS]
    # JB: This line openes an empty list of weight_changes for a mouse
    weight_changes = []
    try:
        baseline_weight = mouse.baseline_weight
        for session in sessions:
            session_weight = session.weight
            weight_change = calculate_weight_change(baseline_weight, session_weight)
            weight_changes.append(weight_change)
        return weight_changes
    except:
        print(f"ERROR: ExceptionError: There went something wrong getting the weight_changes for '{mouse_name}'.")

def calculate_weight_optimum(mouse_name: str):
    # JB: This function takes the baseline weight, calculates the 15% weight loss optimum and returns this value
    # I could think about taking the "MouseSummary" model or the baseline weight as input depending on the function that calls this function
    # (Could save some CPU)
    mouse = load_cache_mouse(mouse_name)
    baseline_weight = mouse.baseline_weight
    weight_optimum = (baseline_weight * 0.85)
    return weight_optimum

def get_number_of_trials_weights_weight_changes(mouse_name: str):
    # JB: This function loads some trial data, counts the number of trials (all; rewarded; unrewarded) as well as the session weight
    # It then returns a list
    print(f"NOTE: Starting to get number of trials and weights for '{mouse_name}'.")
    metadata = gsheet2df(SPREADSHEET_ID, mouse_name, 1)
    try:
        baseline_weight = get_baseline_weight(mouse_name)
    except:
        print(f"ERROR: ExceptionError: There went something wrong getting 'baseline_weight' for '{mouse_name}")
    session_summaries_count = []
    for _, row in metadata.iterrows():
        if "learning day" not in row["Type"].lower():
                continue
        
        session_number = (
            row["Session Number"]
            if len(row["Session Number"]) == 3
            else f"00{row['Session Number']}"
        )

        session_path = DATA_PATH / mouse_name / row["Date"] / session_number

        # JB: These lines get the trial data and raise an ExceptionError if there is an issue getting the trial data
        try:
            trials = load_data(session_path)
            trials = remove_bad_trials(trials)
        except:
            print(
                f"ERROR: ExceptionError: There went something wrong getting the trials."
            )
            print(
                f"Mouse:'{mouse_name}'. Date:'{row["Date"]}'. Session: '{session_number}'."
            )
        
        # JB: Here, we get the weight at the day of the session, and the number of
        # Initialising the data etc. before the try block
        session_weight = None
        try:
            session_weight = float(row["Weight"])
        except:
            print(f"ERROR: ExceptionError: There went something wrong getting 'session_weight'.")

        try:
            weight_change = float(calculate_weight_change(baseline_weight, session_weight))
        except:
            print(f"ERROR: Exception:Error: There went someting wrong getting 'weight_change'.")

        try:
            # Here I want to count the trials
            session_trials_all = 0
            session_trials_rewarded = 0
            session_trials_unrewarded = 0
            for trial in trials:
                session_trials_all += 1
                if trial.texture_rewarded:
                    session_trials_rewarded += 1
                else:
                    session_trials_unrewarded +=1
        except:
            print(f"ERROR: ExceptionError: There went something wrong counting the trials for session '{row["Type"]}'")
        
        # The model is nice, yet we get trouble forming it into a pd dataframe
        # try:
        #     session_summaries_count.append(
        #             SessionSummaryCount(
        #                 name=row["Type"],
        #                 # trials= [summarise_trial(trial) for trial in trials]
        #                 weight=float(session_weight),  # Add your very important weight here
        #                 trials_all = int(session_trials_all),
        #                 trials_rewarded = int(session_trials_rewarded),
        #                 trials_unrewarded = int(session_trials_unrewarded)
        #             )
        #     )
        # except:
        #      print(f"ERROR: ExceptionError: There went something wrong with creating a 'SessionSummaryCount'.")

        # I will try it with a list
        session_summaries_count.append([
                        row["Type"],
                        # trials= [summarise_trial(trial) for trial in trials]
                        session_weight,  # Add your very important weight here
                        weight_change, 
                        int(session_trials_all),
                        int(session_trials_rewarded),
                        int(session_trials_unrewarded)]
            )

    # session_summaries_count_df = pd.DataFrame(data=session_summaries_count, columns=["name", "weight", "trials_all", "trials_rewarded", "trials_unrewarded"])
    # print(session_summaries_count_df)
    # return session_summaries_count_df
    return session_summaries_count

## ACTUAL STATISTICAL CALCULATIONS

def statistical_weight_changes_all():
    # JB: This function takes the list of all weight changes, then does some basic calculations and prints them.
    weight_changes_all = list_weight_changes_all()
    
    # Here are the actual calculations and store them into a dictionary
    statistics_weight_changes = {
        "min_weight_changes": np.min(weight_changes_all),
        "max_weight_changes": np.max(weight_changes_all),
        "mean_weight_changes": np.mean(weight_changes_all),
        "firstQuartile_weight_changes": np.quantile(weight_changes_all, 0.25),
        "secondQuartile_weight_changes": np.quantile(weight_changes_all, 0.5),
        "thirdQuartile_weight_changes": np.quantile(weight_changes_all, 0.75),
        "forthQuartile_weight_changes": np.quantile(weight_changes_all, 1),
    }

    # Printing
    print(statistics_weight_changes)

    return statistics_weight_changes

def calculate_correlation_learning_weight_changes(MICE_LIST: list):
    # JB: This function creates a dataframe of learning metric matched to relative weight change
    # It calculates the correlation and prints this value
    data_list = []
    for idx, mouse_name in enumerate(MICE_LIST):
        mouse = load_cache_mouse(mouse_name)
        # JB: This line states that a sessions has to have more than 30 trials which can be an issue with our mice.
        sessions = [session for session in mouse.sessions if len(session.trials) > SESSION_MIN_TRIALS]
        for session in sessions:
            data_list.append([session.weight_change, learning_metric(session.trials)])
    
    data_dataframe = pd.DataFrame(data=data_list, columns=["session_weight_change", "learning_metric"])
    
    # Here we calculate the correlation as proposed by ChatGPT
    data_correlation = data_dataframe["session_weight_change"].corr(data_dataframe["learning_metric"])
    print(f"STATISTICS: Correlation between learning metric and relative weight change = '{data_correlation}'.")

def calculate_regression_learning_weight_changes(MICE_LIST: list):
    # JB: This function creates a dataframe of learning metric matched to relative weight change
    # It calculates the regression and prints these values
    data_list = []
    for idx, mouse_name in enumerate(MICE_LIST):
        mouse = load_cache_mouse(mouse_name)
        # JB: This line states that a sessions has to have more than 30 trials which can be an issue with our mice.
        sessions = [session for session in mouse.sessions if len(session.trials) > SESSION_MIN_TRIALS]
        for session in sessions:
            data_list.append([session.weight_change, learning_metric(session.trials)])
    
    data_dataframe = pd.DataFrame(data=data_list, columns=["session_weight_change", "learning_metric"])
    
    # Here we calculate the regression as proposed by ChatGPT
    slope, intercept, r_value, p_value, std_err = stats.linregress(data_dataframe["session_weight_change"], data_dataframe["learning_metric"])
    print("STATISTICS: Regression Analysis")
    print(f"Slope = '{slope}'")
    print(f"Intercept = '{intercept}'")
    print(f"R-squared = '{r_value**2}'")
    print(f"P-value = '{p_value}'")
    print(f"Standard error = '{std_err}'")

def calculate_correlation_learning_weight(MICE_LIST: list):
    # JB: This function creates a dataframe of learning metric matched to session weight
    # It calculates the correlation and prints this value
    data_list = []
    for idx, mouse_name in enumerate(MICE_LIST):
        mouse = load_cache_mouse(mouse_name)
        # JB: This line states that a sessions has to have more than 30 trials which can be an issue with our mice.
        sessions = [session for session in mouse.sessions if len(session.trials) > SESSION_MIN_TRIALS]
        for session in sessions:
            data_list.append([session.weight, learning_metric(session.trials)])
    
    data_dataframe = pd.DataFrame(data=data_list, columns=["session_weight", "learning_metric"])
    
    # Here we calculate the correlation as proposed by ChatGPT
    data_correlation = data_dataframe["session_weight"].corr(data_dataframe["learning_metric"])
    print(f"STATISTICS: Correlation between learning metric and session weight = '{data_correlation}'.")

def calculate_regression_learning_weight(MICE_LIST: list):
    # JB: This function creates a dataframe of learning metric matched to session weight
    # It calculates the regression and prints these values
    data_list = []
    for idx, mouse_name in enumerate(MICE_LIST):
        mouse = load_cache_mouse(mouse_name)
        # JB: This line states that a sessions has to have more than 30 trials which can be an issue with our mice.
        sessions = [session for session in mouse.sessions if len(session.trials) > SESSION_MIN_TRIALS]
        for session in sessions:
            data_list.append([session.weight, learning_metric(session.trials)])
    
    data_dataframe = pd.DataFrame(data=data_list, columns=["session_weight", "learning_metric"])
    
    # Here we calculate the regression as proposed by ChatGPT
    slope, intercept, r_value, p_value, std_err = stats.linregress(data_dataframe["session_weight"], data_dataframe["learning_metric"])
    print("STATISTICS: Regression Analysis")
    print(f"Slope = '{slope}'")
    print(f"Intercept = '{intercept}'")
    print(f"R-squared = '{r_value**2}'")
    print(f"P-value = '{p_value}'")
    print(f"Standard error = '{std_err}'")

def calculate_regression_trials_weight_changes_all(MICE_LIST: list):
    # JB: This function takes the dataframe of number of trials matched to relative weight change
    # It calculates the regression and prints these values
    session_summaries_count_all = load_number_of_trials_weights_weight_changes_all()
    # x is for relative weight change
    x = []
    y = []
    # y is for number of trials_all
    for idx, mouse_name in enumerate(MICE_LIST):
        # This is the row with the data of one mouse
        # mouse_rows = session_summaries_count_all[session_summaries_count_all["mouse_name"] == mouse_name]
        # weight_trials_data = mouse_rows[1]
        weight_trials_data = session_summaries_count_all.loc[session_summaries_count_all["mouse_name"] == mouse_name, "weight_trials_data"].iloc[0]
        try:
            for trial_data in weight_trials_data:
                # We are making sure, that we are not appending our fabulous list with 'None' entries
                if trial_data[2] and trial_data[3]:
                    x.append(trial_data[2])
                    y.append(trial_data[3])
        except:
            print(f"ERROR: ExceptionError: There seems to be an issue with the data of '{mouse_name}'.")
    # Here we calculate the regression as proposed by ChatGPT
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    print("STATISTICS: Regression Analysis")
    print(f"Slope = '{slope}'")
    print(f"Intercept = '{intercept}'")
    print(f"R-squared = '{r_value**2}'")
    print(f"P-value = '{p_value}'")
    print(f"Standard error = '{std_err}'")

#############################################################
# PLOTTING CODE #

## Weight
def plot_scatter_learning_weights_multiple(MICE_LIST: list) -> None:
    # JB: This function creates a scatter plot of learning metric matched to weight for all subjects in one plot
    plt.figure(figsize=(10, 4))
    for idx, mouse_name in enumerate(MICE_LIST):
        mouse = load_cache_mouse(mouse_name)
        # JB: This line states that a sessions has to have more than 30 trials which can be an issue with our mice.
        sessions = [session for session in mouse.sessions if len(session.trials) > SESSION_MIN_TRIALS]
        # s = marker_size
        plt.scatter(
            [session.weight for session in sessions],
            [learning_metric(session.trials) for session in sessions],
            s = 18
        )
        # numpy.arange([start, ]stop, [step, ]dtype=None, *, device=None, like=None)
        x = np.arange(18, 28, 0.5)
        plt.xticks(x, fontsize=12)
        plt.legend(
            loc="center left", bbox_to_anchor=(1, 0.5), title="Subjects", labels=MICE_LIST
        )
        plt.title("Learning metric in relation to weight")
        if idx == 0:
            plt.ylabel("Learning metric")
            # JB:
            plt.xlabel("Weight in g")

    plt.tight_layout()
    plt.savefig(HERE.parent / "josef" / "plots" / "scatter_learning_weights_multiple.png", dpi = 300)

def plot_scatter_learning_weights_single(MICE_LIST: list) -> None:
    # JB: This plot creates several plots for every single mouse, matching their learning metric to their weight
    plt.figure(figsize=(15, 4))
    for idx, mouse_name in enumerate(MICE_LIST):
        mouse = load_cache_mouse(mouse_name)
        weight_optimum = calculate_weight_optimum(mouse_name)
        # JB: This line states that a sessions has to have more than 30 trials which can be an issue with our mice. Changed it to ten.
        sessions = [session for session in mouse.sessions if len(session.trials) > SESSION_MIN_TRIALS]
        plt.subplot(1, mice_count, idx + 1)
        plt.scatter(
            [session.weight for session in sessions],
            [learning_metric(session.trials) for session in sessions],
            s = 18
        )
        # JB: I want to add a line, which shows the 15% weight loss optimium
        # plt.axhline(0, color="black", linestyle="--")
        # plt.vlines(weight_optimum, ymin=-1, ymax=2.5, color="k", linestyles="dashed", label="weight optimum")
        plt.axvline(weight_optimum, ymin=0, ymax=1, linestyle="--", color="black") #, label="weight optimum")
        plt.title(mouse_name)
        if idx == 0:
            plt.ylabel("Learning metric")
            # JB:
            plt.xlabel("Weight in g")
    plt.suptitle("Learning metric relative to weight and weight optimum")
    plt.tight_layout()
    plt.savefig(HERE.parent / "josef" / "plots" / "scatter_learning_weights_single.png", dpi = 300) 

def plot_scatter_learning_weights_all(MICE_LIST: list) -> None:
    # JB: This function creates ONE big scatter plot, realting learnign metric to weights
    plt.figure(figsize=(10, 4))
    # JB: Here I create a list that matches session weight and learning metric
    data_list = []
    for idx, mouse_name in enumerate(MICE_LIST):
        mouse = load_cache_mouse(mouse_name)
        # JB: This line states that a sessions has to have more than 30 trials which can be an issue with our mice.
        sessions = [session for session in mouse.sessions if len(session.trials) > SESSION_MIN_TRIALS]
        for session in sessions:
            data_list.append([session.weight, learning_metric(session.trials)])
    
    data_dataframe = pd.DataFrame(data=data_list, columns=["session_weight", "learning_metric"])
    plt.scatter(data_dataframe["session_weight"], data_dataframe["learning_metric"], s=18)
    # numpy.arange([start, ]stop, [step, ]dtype=None, *, device=None, like=None)
    # x = np.arange(18, 28, 0.5)
    # plt.xticks(x, fontsize=12)
    plt.title("Learning metric in relation to weight")
    plt.ylabel("Learning metric")
    plt.xlabel("Weight in g")

    plt.tight_layout()
    plt.savefig(HERE.parent / "josef" / "plots" / "scatter_learning_weights_all.png", dpi = 300)

def plot_regression_line_learning_weights_all(MICE_LIST: list) -> None:
    # JB: This function creates ONE big scatter plot, realting learning metric to weights with a REGRESSION LINE
    plt.figure(figsize=(10, 4))
    # JB: Here I create a list that matches session weight and learning metric
    data_list = []
    for idx, mouse_name in enumerate(MICE_LIST):
        mouse = load_cache_mouse(mouse_name)
        # JB: This line states that a sessions has to have more than 30 trials which can be an issue with our mice.
        sessions = [session for session in mouse.sessions if len(session.trials) > SESSION_MIN_TRIALS]
        for session in sessions:
            data_list.append([session.weight, learning_metric(session.trials)])
    data_dataframe = pd.DataFrame(data=data_list, columns=["session_weight", "learning_metric"])
    sns.lmplot(x="session_weight", y="learning_metric", data=data_dataframe, aspect=1.5, height=6)
    plt.title("Learning metric relative to session weight with regression line")
    plt.xlabel("Session weight in g")
    plt.ylabel("Learning metric")

    plt.tight_layout()
    plt.savefig(HERE.parent / "josef" / "plots" / "regression_line_learning_weights_all.png", dpi = 300)

def plot_scatter_trials_weights_single(MICE_LIST: list) -> None:
    # JB: This function creates scatter plots of number of trials matched to weight for all subjects in one plot for each condition
    session_summaries_count_all = load_number_of_trials_weights_weight_changes_all()
    plt.figure(figsize=(18, 5)) 
    for idx, mouse_name in enumerate(MICE_LIST):
        # This is the row with the data of one mouse
        # mouse_rows = session_summaries_count_all[session_summaries_count_all["mouse_name"] == mouse_name]
        # weight_trials_data = mouse_rows[1]
        weight_trials_data = session_summaries_count_all.loc[session_summaries_count_all["mouse_name"] == mouse_name, "weight_trials_data"].iloc[0]
        weight_optimum = calculate_weight_optimum(mouse_name)
        # x is for weight
        x = []
        y1 = []
        y2 = []
        y3 = []
        # y1 is for number of trials_all
        # y2 is for number of trials_rewarded
        # y3 is for number of trials_rewarded
        try:
            for trial_data in weight_trials_data:
                # We are making sure, that we are not appending our fabulous list with 'None' entries
                if trial_data[1] and trial_data[3] and trial_data[4] and trial_data[5]:
                    x.append(trial_data[1])
                    y1.append(trial_data[3])
                    y2.append(trial_data[4])
                    y3.append(trial_data[5])
        except:
            print(f"ERROR: ExceptionError: There seems to be an issue with the data of '{mouse_name}'.")
        
        plt.subplot(1, mice_count, idx + 1)
        plt.scatter(x, y1,s = 18, c = "blue")
        plt.scatter(x, y2,s = 18, c = "green")
        plt.scatter(x, y3, s = 18,c = "red")
        plt.axvline(weight_optimum, ymin=0, ymax=1, linestyle="--", color="black") #, label="weight optimum")
        plt.title(mouse_name)
        if idx == 0:
            plt.ylabel("n trials")
            # JB:
            plt.xlabel("Weight in g")
    labels = ["all", "rewarded", "unrewarded"]
    plt.legend(
        loc="center left", bbox_to_anchor=(1, 0.5), title="Conditions", labels=labels
    )
    plt.suptitle("Number of all trials - weight")
    plt.tight_layout(h_pad = 5)
    plt.savefig(HERE.parent / "josef" / "plots" / "scatter_trials_weight_single.png", dpi = 300)

## Weight changes
def plot_distribution_weight_changes() -> None:
    # JB: This function takes the list of all weight chnages and creates a distribution plot
    weight_changes_all = list_weight_changes_all()
    data = np.array(weight_changes_all)
    plt.figure()
    plt.hist(data, weights=np.zeros_like(data) + 1.0 / data.size)
    plt.xlabel("Relative weight change")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.title("Distribution of weight change")
    plt.savefig(HERE.parent / "josef" / "plots" / "distribution_weight_changes", dpi = 300)

def plot_boxplot_weight_changes():
    # JB: This function takes the list of all weight chnages and creates a boxplot
    weight_changes_all = list_weight_changes_all()
    data = np.array(weight_changes_all)
    plt.figure()
    plt.boxplot(data)
    plt.ylabel("Relative weight change")
    plt.tight_layout()
    plt.title("Distribution of weight change")
    plt.savefig(HERE.parent / "josef"/ "plots" / "boxplot_weight_changes", dpi = 300)

def plot_learning_weight_changes_multiple(MICE_LIST: list) -> None:
    plt.figure(figsize=(10, 4))
    for idx, mouse_name in enumerate(MICE_LIST):
        mouse = load_cache_mouse(mouse_name)
        # JB: This line states that a sessions has to have more than 30 trials which can be an issue with our mice.
        sessions = [session for session in mouse.sessions if len(session.trials) > SESSION_MIN_TRIALS]

        # I have to create a dataframe with two columns: 1) relative weight change 2) learning metric
        # I do not :)
        # data = []
        # for session in sessions:
        #     weight_change = session.weight_change
        #     learning_metric = learning_metric(session.trials)
        #     data.append([weight_change, learning_metric])
        # dataframe = pd.DataFrame(data=data, columns=[weight_change, learning_metric])
        # print(dataframe)

        # s = marker_size
        plt.scatter(
            [session.weight_change for session in sessions],
            [learning_metric(session.trials) for session in sessions],
            s = 18
        )
        if idx == 0:
            plt.ylabel("Learning metric")
            plt.xlabel("Relative weight change")
        # numpy.arange([start, ]stop, [step, ]dtype=None, *, device=None, like=None)
        # x = np.arange(18, 28, 0.5)
        # plt.xticks(x, fontsize=12)
    plt.axvline(-0.15, ymin=0, ymax=1, linestyle="--", color="black", label="weight optimum")
    plt.legend(
        loc="center left", bbox_to_anchor=(1, 0.5), title="Subjects", labels=MICE_LIST
    )
    plt.title("Learning metric in relation to weight change")
    if idx == 0:
        plt.ylabel("Learning metric")
        # JB:
        plt.xlabel("Relative weight change")
    plt.tight_layout()
    plt.savefig(HERE.parent / "josef" / "plots" / "learning_weight_changes_multiple.png", dpi = 300)

def plot_learning_weight_changes_single(MICE_LIST: list) -> None:
    plt.figure(figsize=(15, 4))
    for idx, mouse_name in enumerate(MICE_LIST):
        mouse = load_cache_mouse(mouse_name)
        # JB: This line states that a sessions has to have more than 30 trials which can be an issue with our mice. Changed it to ten.
        sessions = [session for session in mouse.sessions if len(session.trials) > SESSION_MIN_TRIALS]
        plt.subplot(1, mice_count, idx + 1)
        plt.scatter(
            [session.weight_change for session in sessions],
            [learning_metric(session.trials) for session in sessions],
            s = 18
        )
        # Trying to properly scale the x-axis
        try:
            xdata = [session.weight_change for session in sessions]
            # Round to nearest .05
            xmin = math.floor((np.min(xdata))*20) / 20
            xmax = math.ceil((np.max(xdata))*20) / 20
            x = np.arange(xmin, xmax, 0.05)
            plt.xticks(x, fontsize=10)
        except:
            print(f"ERROR: ExceptionError: Could not define the x-axis for '{mouse_name}'. Probably lacking data here.")
        # JB: I want to add a line, which shows the 15% weight loss optimium
        # plt.axhline(0, color="black", linestyle="--")
        # plt.vlines(weight_optimum, ymin=-1, ymax=2.5, color="k", linestyles="dashed", label="weight optimum")
        plt.axvline(MICE_OPTIMAL_WEIGHT_LOSS, ymin=0, ymax=1, linestyle="--", color="black") #, label="weight optimum")
        plt.title(mouse_name)
        if idx == 0:
            plt.ylabel("Learning metric")
            # JB:
            plt.xlabel("Weight changes")
    plt.suptitle("Learning metric relative to weigh changes")
    plt.tight_layout()
    plt.savefig(HERE.parent / "josef" / "plots" / "learning_weight_changes_single.png", dpi = 300)

def plot_learning_weight_changes_all(MICE_LIST: list) -> None:
    # JB: This function creates ONE big scatter plot, realting learnign metric to weights
    plt.figure(figsize=(10, 4))
    # JB: Here I create a list that matches session weight and learning metric
    data_list = []
    for idx, mouse_name in enumerate(MICE_LIST):
        mouse = load_cache_mouse(mouse_name)
        # JB: This line states that a sessions has to have more than 30 trials which can be an issue with our mice.
        sessions = [session for session in mouse.sessions if len(session.trials) > SESSION_MIN_TRIALS]
        for session in sessions:
            data_list.append([session.weight_change, learning_metric(session.trials)])
    
    data_dataframe = pd.DataFrame(data=data_list, columns=["session_weight_change", "learning_metric"])
    plt.scatter(data_dataframe["session_weight_change"], data_dataframe["learning_metric"], s=18)
    # numpy.arange([start, ]stop, [step, ]dtype=None, *, device=None, like=None)
    # x = np.arange(18, 28, 0.5)
    # plt.xticks(x, fontsize=12)
    plt.title("Learning metric in relation to relative weight change")
    plt.ylabel("Learning metric")
    plt.xlabel("Relative weight change")

    plt.tight_layout()
    plt.savefig(HERE.parent / "josef" / "plots" / "learning_weight_changes_all.png", dpi = 300)

def plot_regression_line_learning_weight_changes_all(MICE_LIST: list) -> None:
    # JB: This function creates ONE big scatter plot, realting learning metric to weight changes with a REGRESSION LINE
    plt.figure(figsize=(10, 4))
    # JB: Here I create a list that matches session weight change and learning metric
    data_list = []
    for idx, mouse_name in enumerate(MICE_LIST):
        mouse = load_cache_mouse(mouse_name)
        # JB: This line states that a sessions has to have more than 30 trials which can be an issue with our mice.
        sessions = [session for session in mouse.sessions if len(session.trials) > SESSION_MIN_TRIALS]
        for session in sessions:
            data_list.append([session.weight_change, learning_metric(session.trials)])
    data_dataframe = pd.DataFrame(data=data_list, columns=["session_weight_change", "learning_metric"])
    sns.lmplot(x="session_weight_change", y="learning_metric", data=data_dataframe, aspect=1.5, height=6)
    plt.title("Learning metric relative to relative weight change with regression line")
    plt.xlabel("Relative weight change")
    plt.ylabel("Learning metric")

    plt.tight_layout()
    plt.savefig(HERE.parent / "josef" / "plots" / "regression_line_learning_weight_changes_all.png", dpi = 300)

def plot_scatter_trials_weight_changes_single(MICE_LIST: list) -> None:
    # JB: This function creates scatter plots of number of trials matched to weight changes for all subjects for each condition
    session_summaries_count_all = load_number_of_trials_weights_weight_changes_all()
    plt.figure(figsize=(18, 5)) 
    for idx, mouse_name in enumerate(MICE_LIST):
        # This is the row with the data of one mouse
        # mouse_rows = session_summaries_count_all[session_summaries_count_all["mouse_name"] == mouse_name]
        # weight_trials_data = mouse_rows[1]
        weight_trials_data = session_summaries_count_all.loc[session_summaries_count_all["mouse_name"] == mouse_name, "weight_trials_data"].iloc[0]
        # x is for relative weight change
        x = []
        y1 = []
        y2 = []
        y3 = []
        # y1 is for number of trials_all
        # y2 is for number of trials_rewarded
        # y3 is for number of trials_rewarded
        try:
            for trial_data in weight_trials_data:
                # We are making sure, that we are not appending our fabulous list with 'None' entries
                if trial_data[2] and trial_data[3] and trial_data[4] and trial_data[5]:
                    x.append(trial_data[2])
                    y1.append(trial_data[3])
                    y2.append(trial_data[4])
                    y3.append(trial_data[5])
        except:
            print(f"ERROR: ExceptionError: There seems to be an issue with the data of '{mouse_name}'.")
        
        plt.subplot(1, mice_count, idx + 1)
        plt.scatter(x, y1,s = 18, c = "blue")
        plt.scatter(x, y2,s = 18, c = "green")
        plt.scatter(x, y3, s = 18,c = "red")
        plt.title(mouse_name)
        if idx == 0:
            plt.ylabel("n trials")
            # JB:
            plt.xlabel("Relative weight change")
    labels = ["all", "rewarded", "unrewarded"]
    plt.legend(
        loc="center left", bbox_to_anchor=(1, 0.5), title="Conditions", labels=labels
    )
    
    plt.suptitle("Number of all trials - relative weight change")
    plt.tight_layout(h_pad = 5)
    plt.savefig(HERE.parent / "josef" / "plots" / "scatter_trials_weight_changes_single.png", dpi = 300)

def plot_trials_weight_changes_all(MICE_LIST: list) -> None:
    # JB: This function creates a plot with number of trials against relative weight change for each condition
    session_summaries_count_all = load_number_of_trials_weights_weight_changes_all()
    plt.figure(figsize=(12, 5))
    # x is for relative weight change
    x = []
    y1 = []
    y2 = []
    y3 = []
    # y1 is for number of trials_all
    # y2 is for number of trials_rewarded
    # y3 is for number of trials_rewarded
    for idx, mouse_name in enumerate(MICE_LIST):
        # This is the row with the data of one mouse
        # mouse_rows = session_summaries_count_all[session_summaries_count_all["mouse_name"] == mouse_name]
        # weight_trials_data = mouse_rows[1]
        weight_trials_data = session_summaries_count_all.loc[session_summaries_count_all["mouse_name"] == mouse_name, "weight_trials_data"].iloc[0]
        try:
            for trial_data in weight_trials_data:
                # We are making sure, that we are not appending our fabulous list with 'None' entries
                if trial_data[2] and trial_data[3] and trial_data[4] and trial_data[5]:
                    x.append(trial_data[2])
                    y1.append(trial_data[3])
                    y2.append(trial_data[4])
                    y3.append(trial_data[5])
        except:
            print(f"ERROR: ExceptionError: There seems to be an issue with the data of '{mouse_name}'.")
    
    datadictionary = {
        "x": x,
        "all": y1,
        "rewarded": y2,
        "unrewarded": y3
    }
    dataframe = pd.DataFrame(data= datadictionary, columns=["x", "all", "rewarded", "unrewarded"])
    dataframe = pd.melt(dataframe, id_vars=["x"], value_vars=["all", "rewarded", "unrewarded"],
                  var_name="Condition", value_name="Value")
    sns.lineplot(data=dataframe, x="x", y="Value", hue="Condition")
    plt.xlabel("Relative weight change")
    plt.ylabel("n trials")
    plt.title("Number of trials - relative weight change")
    plt.tight_layout(h_pad = 5)
    plt.savefig(HERE.parent / "josef" / "plots" / "trials_weight_changes_all.png", dpi = 300)

def plot_regression_line_trials_weight_changes_all(MICE_LIST: list) -> None:
    # JB: This function creates ONE big scatter plot, relating number of all trials against relative weight change
    session_summaries_count_all = load_number_of_trials_weights_weight_changes_all()
    plt.figure()
    # x is for relative weight change
    x = []
    y = []
    # y is for number of trials_all
    for idx, mouse_name in enumerate(MICE_LIST):
        # This is the row with the data of one mouse
        # mouse_rows = session_summaries_count_all[session_summaries_count_all["mouse_name"] == mouse_name]
        # weight_trials_data = mouse_rows[1]
        weight_trials_data = session_summaries_count_all.loc[session_summaries_count_all["mouse_name"] == mouse_name, "weight_trials_data"].iloc[0]
        try:
            for trial_data in weight_trials_data:
                # We are making sure, that we are not appending our fabulous list with 'None' entries
                if trial_data[2] and trial_data[3]:
                    x.append(trial_data[2])
                    y.append(trial_data[3])
        except:
            print(f"ERROR: ExceptionError: There seems to be an issue with the data of '{mouse_name}'.")
    
    datadictionary = {
        "x": x,
        "y": y
    }
    dataframe = pd.DataFrame(data= datadictionary, columns=["x", "y"])
    sns.lmplot(data=dataframe, x="x", y="y") #, aspect=1.5, height=6)
    plt.xlabel("Relative weight change")
    plt.ylabel("n trials")
    plt.title("Number of trials - relative weight change")
    # plt.tight_layout(h_pad = 5)
    plt.savefig(HERE.parent / "josef" / "plots" / "regression_line_trials_weight_changes_all.png", dpi = 300, bbox_inches="tight")

#############################################################
# MAIN CODE #

# Not best practice
# mice_list = ("J011", "J013", "J015", "J016", "J017")


cache_mouse("J016")

for mouse_name in MICE_LIST:
    cache_mouse(mouse_name)
    # weight_changes = get_weight_changes(mouse_name)
    # print(weight_changes)

# cache_weight_changes_all(MICE_LIST)

# plot_scatter_learning_weights_multiple(MICE_LIST)
# plot_scatter_learning_weights_single(MICE_LIST)
# plot_learning_weight_changes_multiple(MICE_LIST)
# plot_learning_weight_changes_single(MICE_LIST)
# statistical_weight_changes_all()
# plot_distribution_weight_changes()
# plot_boxplot_weight_changes()
# calculate_weight_optimum("J011")

# plot_scatter_learning_weights_all(MICE_LIST)
# plot_learning_weight_changes_all(MICE_LIST)

# calculate_correlation_learning_weight_changes(MICE_LIST)
# calculate_regression_learning_weight_changes(MICE_LIST)

# plot_regression_line_learning_weight_changes_all(MICE_LIST)

# calculate_correlation_learning_weight(MICE_LIST)
# calculate_regression_learning_weight(MICE_LIST)

# plot_regression_line_learning_weights_all(MICE_LIST)

# get_number_of_trials_weight("J011")

# cache_number_of_trials_weights_weight_changes_all(MICE_LIST)

# plot_scatter_trials_weights_single(MICE_LIST)
# plot_scatter_trials_weight_changes_single(MICE_LIST)

# plot_trials_weight_changes_all(MICE_LIST)
# plot_regression_line_trials_weight_changes_all(MICE_LIST)
# calculate_regression_trials_weight_changes_all(MICE_LIST)