# Python modules
import sys
import json
import yaml
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

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
from viral.models import SessionSummary, MouseSummary
from josef.constants import HERE, DATA_PATH, SPREADSHEET_ID

#############################################################
# SETTINGS CODE #
# JB: Parsing the config.yaml file
with open(HERE.parent / "josef" / "config.yaml", "r") as file:
    settings = yaml.safe_load(file)

# JB: Setting the minimal number of trials a session has to have to be included into the analysis
SESSION_MIN_TRIALS = settings["SESSION_MIN_TRIALS"]

# JB: Setting the minimal number of trials a session has to have to be included into the analysis
MICE_LIST = settings["MICE_LIST"]
print(MICE_LIST)

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
             print(f"ERROR: ExceptionError: There went something wrong with creating a 'SessionSummary'.")
        
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
        for session in sessions:
            session_weight = session.weight
            baseline_weight = mouse.baseline_weight
            weight_change = calculate_weight_change(baseline_weight, session_weight)
            weight_changes.append(weight_change)
        return weight_changes
    except:
        print(f"ERROR: ExceptionError: There went something wrong getting the weight_changes for '{mouse_name}'.")

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

#############################################################
# PLOTTING CODE #

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

def plot_scatter_learning_weights_multiple(MICE_LIST: list) -> None:
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
        if idx == 0:
            plt.ylabel("Learning Metric")
            plt.xlabel("Weight in g")
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

#############################################################
# MAIN CODE #

# Not best practice
# mice_list = ("J011", "J013", "J015", "J016", "J017")



# for mouse_name in MICE_LIST:
#     cache_mouse(mouse_name)
#     weight_changes = get_weight_changes(mouse_name)
#     print(weight_changes)

# cache_weight_changes_all(MICE_LIST)

plot_scatter_learning_weights_multiple(MICE_LIST)

# statistical_weight_changes_all()
# plot_distribution_weight_changes()
# plot_boxplot_weight_changes()