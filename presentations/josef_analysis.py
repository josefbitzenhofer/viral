from pathlib import Path
from matplotlib import pyplot as plt
import sys
import json
import seaborn as sns
import numpy as np
import pandas as pd


HERE = Path(__file__).parent
sys.path.append(str(HERE.parent))

sns.set_theme(context="talk", style="ticks")

from viral.single_session import (
    load_data,
    plot_rewarded_vs_unrewarded_licking,
    plot_speed,
    remove_bad_trials,
    summarise_trial,
)
from viral.multiple_sessions import (
    # load_cache,
    plot_performance_across_days,
    learning_metric,
    speed_difference,
    licking_difference,
)
from viral.gsheets_importer import gsheet2df
from viral.constants import DATA_PATH, HERE, SPREADSHEET_ID
from viral.models import SessionSummary, MouseSummary

from typing import List

#############################################################
# CACHE CODE

## Do not use this one, it saves weights, but NO weight changes.
def cache_mouse_w_weight(mouse_name: str) -> None:
    # Takes a mouse's name, gets the metadata and the trials and saves the name and the session summaries (incl. weight!).
    metadata = gsheet2df(SPREADSHEET_ID, mouse_name, 1)
    session_summaries = []
    for _, row in metadata.iterrows():
        if "learning day" not in row["Type"].lower():
            continue

        session_number = (
            row["Session Number"]
            if len(row["Session Number"]) == 3
            else f"00{row['Session Number']}"
        )

        session_path = DATA_PATH / mouse_name / row["Date"] / session_number
        try:
            trials = load_data(session_path)
        except:
            print(
                f"ERROR: ExceptionError: There went something wrong getting the trials in session '{session_number}' for '{mouse_name}'."
            )
        trials = remove_bad_trials(trials)
        # JB: We handle missing values. The row gets excluded.
        try:
            session_summaries.append(
                SessionSummary(
                    name=row["Type"],
                    trials=[summarise_trial(trial) for trial in trials],
                    weight=float(row["Weight"]),  # Add your very important weight here
                )
            )
        except:
            print(f"It seems like at least one value is missing here.")

    with open(HERE.parent / "data" / "josef" / f"{mouse_name}.json", "w") as f:
        json.dump(
            MouseSummary(
                sessions=session_summaries,
                name=mouse_name,
            ).model_dump(),
            f,
        )

def load_cache(mouse_name: str) -> MouseSummary:
    with open(HERE.parent / "data" / "josef" / f"{mouse_name}.json", "r") as f:
        return MouseSummary.model_validate_json(f.read())

def cache_sessions_weight_changes(mouse_name: str) -> None:
    metadata = gsheet2df(SPREADSHEET_ID, mouse_name, 1)
    session_summaries = []
    for idx, mouse_name in enumerate(myMice):
        for _, row in metadata.iterrows():
            if "learning day" not in row["Type"].lower():
                continue

            session_number = (
                row["Session Number"]
                if len(row["Session Number"]) == 3
                else f"00{row['Session Number']}"
            )

            session_path = DATA_PATH / mouse_name / row["Date"] / session_number
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

            # try:
            #     weight = float(row["Weight"])
            #     print(weight)
            #     baselineWeight = myMice_baselineWeights.get(mouse_name)
            #     weight_change = calculate_weight_change(baselineWeight, weight)
            # except:
            #     print("ERROR: ExecptionError: There is no entry for 'Weight'.")

            
            # JB: We handle missing values. The row gets excluded.
            try:
                weight = float(row["Weight"])
                print(weight)
                baselineWeight = myMice_baselineWeights.get(mouse_name)
                weight_change = calculate_weight_change(baselineWeight, weight)
                print(weight_change)
                # trials=[summarise_trial(trial) for trial in trials]
                # name=row["Type"]
                # print(trials)
                # print(name)

                # session_summary = SessionSummary(
                #         name=row["Type"],
                #         trials=[summarise_trial(trial) for trial in trials],
                #         weight=weight,  # Add your very important weight here
                #         weight_change=float(weight_change))
                # print(session_summary)
                # session_summaries.append(session_summary)

                session_summaries.append(
                    SessionSummary(
                        name=row["Type"],
                        trials=[summarise_trial(trial) for trial in trials],
                        weight=weight,  # Add your very important weight here
                        weight_change=float(weight_change)
                    )
                )
            except:
                print(f"It seems like at least one value is missing here.")

        with open(
            HERE.parent / "data" / "josef" / "weight_changes" / f"{mouse_name}.json",
            "w",
        ) as f:
            json.dump(
                MouseSummary(
                    sessions=session_summaries,
                    name=mouse_name,
                ).model_dump(),
                f,
            )

def load_cache_sessions_weight_changes(mouse_name: str) -> None:
    with open(
        HERE.parent / "data" / "josef" / "weight_changes" / f"{mouse_name}.json", "r"
    ) as f:
        return MouseSummary.model_validate_json(f.read())

def readcache_weight_changes():
    with open(HERE.parent / "data" / "josef" / "weight_changes.json", "r") as f:
        return json.load(f)

#############################################################
# CALCULATIONS CODE
## Weight change for every single day
def calculate_weight_change(baselineWeight: float, sessionWeight: int):
    # weight_change = abs((baselineWeight - sessionWeight) / baselineWeight)
    weight_change = float((baselineWeight - sessionWeight) / baselineWeight)
    return weight_change

def get_weight_changes(myMice: list):
    weight_changes = []
    for idx, mouse_name in enumerate(myMice):
        mouse = load_cache(mouse_name)
        # JB: This line states that a sessions has to have more than 30 trials which can be an issue with our mice. Changed it to ten.
        sessions = [session for session in mouse.sessions if len(session.trials) > 5]
        for session in sessions:
            sessionWeight = session.weight
            baselineWeight = myMice_baselineWeights.get(mouse_name)
            weight_change = calculate_weight_change(baselineWeight, sessionWeight)
            weight_changes.append(weight_change)
    return weight_changes

def print_weight_changes(myMice: List):
    # Weight changes as a whole
    weight_changes = get_weight_changes(myMice)
    print(weight_changes)
    with open(HERE.parent / "data" / "josef" / "weight_changes.json", "w") as f:
        json.dump(weight_changes, f)


#############################################################
# PLOTTING CODE
## JB: try scatter plot
def plot_scatter(sessions: List[SessionSummary]) -> None:

    # plt.plot([x], [y])
    plt.scatter(
        [session.weight for session in sessions],
        [learning_metric(session.trials) for session in sessions],
    )

    # plt.xticks(
    #     range(len(sessions)),
    #     [f"Day {idx + 1}" for idx in range(len(sessions))],
    #     rotation=90,
    # )

    # plt.axhline(0, color="black", linestyle="--")
    # plt.title(MOUSE)

def scatter_single(myMice):
    plt.figure(figsize=(10, 4))
    # f, (ax, bx) = plt.subplots(, 1, sharey="row")
    for idx, mouse_name in enumerate(myMice):
        mouse = load_cache(mouse_name)
        # JB: This line states that a sessions has to have more than 30 trials which can be an issue with our mice. Changed it to ten.
        sessions = [session for session in mouse.sessions if len(session.trials) > 5]
        # sessions = mouse.sessions
        # print(sessions)
        plt.subplot(1, 5, idx + 1)
        # plt.ylim(-0.7, 2.7)
        plot_scatter(sessions)
        plt.title(mouse_name)
        if idx == 0:
            plt.ylabel("Learning Metric")
            # JB:
            plt.xlabel("Weight in g")

    plt.tight_layout()
    plt.savefig(HERE.parent / "plots" / "scatter" / "single") 

## JB: Scatter all mice in one plot
def scatter_multiple(myMice):
    plt.figure(figsize=(10, 4))
    # f, (ax, bx) = plt.subplots(, 1, sharey="row")
    for idx, mouse_name in enumerate(myMice):
        mouse = load_cache(mouse_name)
        # JB: This line states that a sessions has to have more than 30 trials which can be an issue with our mice. Changed it to ten.
        sessions = [session for session in mouse.sessions if len(session.trials) > 5]
        # sessions = mouse.sessions
        # print(sessions)
        # plt.subplot(1, 5, idx + 1)
        # plt.ylim(-0.7, 2.7)
        plot_scatter(sessions)
        # plt.title("Learning metric - weights")
        if idx == 0:
            plt.ylabel("Learning Metric")
            # JB:
            plt.xlabel("Weight in g")
        # numpy.arange([start, ]stop, [step, ]dtype=None, *, device=None, like=None)
        x = np.arange(18, 28, 0.5)
        plt.xticks(x, fontsize=12)
        plt.legend(
            loc="center left", bbox_to_anchor=(1, 0.5), title="Subjects", labels=myMice
        )

    plt.tight_layout()
    plt.savefig(HERE.parent / "plots" / "scatter" / "multiple")

def across_days(myMice: list):
    plt.figure(figsize=(10, 4))
    # f, (ax, bx) = plt.subplots(, 1, sharey="row")
    for idx, mouse_name in enumerate(myMice):

        mouse = load_cache(mouse_name)
        # JB: This line states that a sessions has to have more than 30 trials which can be an issue with our mice. Changed it to ten.
        sessions = [session for session in mouse.sessions if len(session.trials) > 10]
        print(sessions)
        plt.subplot(1, 3, idx + 1)
        plt.ylim(-0.7, 2.7)
        plot_performance_across_days(sessions)
        plt.title(mouse_name)
        if idx == 0:
            plt.ylabel("Learning Metric")

    plt.tight_layout()
    plt.savefig(HERE.parent / "plots" / "learning_metric_across_days.png")


# def get_trials(MOUSE:str) -> None:
#     mouse = load_cache(mouse_name)
#     sessions = [session for session in mouse.sessions if len(session.trials) > 30]

# def get_weight(MOUSE: str) -> None:
#     SESSION_PATH = DATA_PATH / MOUSE / DATE / SESSION_NUMBER

#     trials = load_data(SESSION_PATH)
#     trials = remove_bad_trials(trials)

#############################################################
# JB: This code is for the weight analysis

#############################################################
# For a single mouse
# def plot_learningmetric_weights_single(sessions: List[SessionSummary]) -> None:
#     plt.plot()


# def learningmetric_weights_single(mouse_name: str):
#     mouse = load_cache(mouse_name)
#     sessions = [session for session in mouse.sessions if len(session.trials) > 10]
#     print(f"Number of sessions for '{mouse_name}': {len(sessions)} sessions.")
#     plt.subplot(1, 3)
#     plt.ylim(-0.7, 2.7)





############################
# Subgrouping
# Obviously not best practice, but here are the baseline weights:
myMice_baselineWeights = {
    "J011": 22.6,
    "J013": 27.9,
    "J015": 22.4,
    "J016": 23.3,
    "J017": 23.5,
}

def statistical_weight_changes():
    weight_changes = readcache_weight_changes()
    # Calculations
    min_weight_changes = np.min(weight_changes)
    max_weight_changes = np.max(weight_changes)
    mean_weight_changes = np.mean(weight_changes)
    firstQuartile_weight_changes = np.quantile(weight_changes, 0.25)
    secondQuartile_weight_changes = np.quantile(weight_changes, 0.5)
    thirdQuartile_weight_changes = np.quantile(weight_changes, 0.75)
    forthQuartile_weight_changes = np.quantile(weight_changes, 1)

    statistics_weight_changes = {
        "min_weight_changes": min_weight_changes,
        "max_weight_changes": max_weight_changes,
        "mean_weight_changes": mean_weight_changes,
        "firstQuartile_weight_changes": firstQuartile_weight_changes,
        "secondQuartile_weight_changes": secondQuartile_weight_changes,
        "thirdQuartile_weight_changes": thirdQuartile_weight_changes,
        "forthQuartile_weight_changes": forthQuartile_weight_changes,
    }

    print(statistics_weight_changes)

    return statistics_weight_changes

    # print(
    #     f"Minimum: '{min_weight_changes}'"
    # )
    # print(
    #     f"Mean = '{mean_weight_changes}'"
    # )
    # print(
    #     f"Maximum: '{max_weight_changes}'"
    # )
    # print(
    #     f"1st Quartile = '{firstQuartile_weight_changes}'"
    # )
    # print(
    #     f"2nd Quartile (Median) = '{secondQuartile_weight_changes}'"
    # )
    # print(
    #     f"3rd Quartile = '{thirdQuartile_weight_changes}'"
    # )
    # print(
    #     f"4th Quartile (Maximum) = '{forthQuartile_weight_changes}'"
    # )


def plot_distriution_weight_changes():
    weight_changes = readcache_weight_changes()
    data = np.array(weight_changes)
    # 1
    plt.hist(data, weights=np.zeros_like(data) + 1.0 / data.size)
    # plt.hist(
    #     weight_changes,
    #     # density = True,
    #     # stacked = True
    # )
    plt.xlabel("Relative weight change")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.title("Distribution of weight change")
    plt.savefig(HERE.parent / "plots" / "scatter" / "statistics_distribution")


def plot_boxplot_weight_changes():
    weight_changes = readcache_weight_changes()
    data = np.array(weight_changes)
    # 2
    plt.boxplot(data)
    # plt.xlabel("Relative weight change")
    plt.ylabel("Relative weight change")
    plt.tight_layout()
    plt.title("Distribution of weight change")
    plt.savefig(HERE.parent / "plots" / "scatter" / "statistics_boxplot")


def broad_learning_metric_weight_changes(myMice: list):
    
    # JB: Not at all tested so far
    # JB: should be a dataframe, I guess
    weight_changes_broad = []
    for idx, mouse_name in enumerate(myMice):
        print(mouse_name)
        mouse = load_cache_sessions_weight_changes(mouse_name)
        sessions = [session for session in mouse.sessions if len(session.trials) > 5]
        for session in sessions:
            session_weight_change = [session.weight_change]
            session_learning_metric = [learning_metric(session.trials)]
            weight_changes_broad.append([session_weight_change, session_learning_metric])

    broad_lmwc_df = pd.DataFrame(
        data=weight_changes_broad, columns=["session_weight_change", "session_learning_metric"]
    )
    # print(broad_lmwc_df)
    return broad_lmwc_df


def define_subgroups_weight_changes(myMice: list):
    statistics_weight_changes = statistical_weight_changes()
    weight_changes = readcache_weight_changes()

    # Probably you should make this a constants.py
    min_weight_changes = statistics_weight_changes.get(min_weight_changes)
    max_weight_changes = statistics_weight_changes.get(max_weight_changes)
    mean_weight_changes = statistics_weight_changes.get(mean_weight_changes)
    firstQuartile_weight_changes = statistics_weight_changes.get(
        firstQuartile_weight_changes
    )
    secondQuartile_weight_changes = statistics_weight_changes.get(
        secondQuartile_weight_changes
    )
    thirdQuartile_weight_changes = statistics_weight_changes.get(
        thirdQuartile_weight_changes
    )
    forthQuartile_weight_changes = statistics_weight_changes.get(
        forthQuartile_weight_changes
    )

    # Subgroups
    weight_changes_A = []
    weight_changes_B = []
    weight_changes_C = []
    weight_changes_D = []

    # Introduce both things: weight_changes_broad and learning metric
    weight_changes_broad = broad_learning_metric_weight_changes(myMice)
    weight_changes_broad = weight_changes_broad.reset_index()

    for index, row in weight_changes_broad.iterrows():
        # A:    Minimum -> Q1
        if (
            row["session_weight_change"] >= min_weight_changes
            and row["session_weight_change"] <= firstQuartile_weight_changes
        ):
            weight_changes_A.append([row["session_weight_change"], row["session_weight_change"]])
        # B:    Q1 -> Q2 (Median)
        if (
            row["session_weight_change"] >= firstQuartile_weight_changes
            and row["session_weight_change"] <= secondQuartile_weight_changes
        ):
            weight_changes_B.append([row["session_weight_change"], row["session_weight_change"]])
        # C:    Q2 -> Q3
        if (
            row["session_weight_change"] >= secondQuartile_weight_changes
            and row["session_weight_change"] <= thirdQuartile_weight_changes
        ):
            weight_changes_C.append([row["session_weight_change"], row["session_weight_change"]])
        # D:    Q3 -> Q4 (Maximum)
        if (
            row["session_weight_change"] >= thirdQuartile_weight_changes
            and row["session_weight_change"] <= forthQuartile_weight_changes
        ):
            weight_changes_D.append([row["session_weight_change"], row["session_weight_change"]])


def plot_broad_learning_metric_weight_changes() -> None:
    plt.figure(figsize=(10, 4))
    # f, (ax, bx) = plt.subplots(, 1, sharey="row")
    # plt.ylim(-0.7, 2.7)
    plt.scatter()
    plt.title("Learning metric in relation to relative weight change")
    # if idx == 0:
    #     plt.ylabel("Learning Metric")
    #     # JB:
    #     plt.xlabel("Weight in g")
    plt.ylabel()
    plt.xlabel()
    plt.tight_layout()
    plt.savefig(HERE.parent / "plots" / "scatter" / "learning_metric_WEIGHTS.png")


############################
def plot_performance_across_days_weights(sessions: List[SessionSummary]) -> None:

    # plt.plot([x], [y])
    plt.plot(
        [session.weight for session in sessions],
        [learning_metric(session.trials) for session in sessions],
    )

    # plt.xticks(
    #     range(len(sessions)),
    #     [f"Day {idx + 1}" for idx in range(len(sessions))],
    #     rotation=90,
    # )

    plt.axhline(0, color="black", linestyle="--")
    # plt.title(MOUSE)

def across_days_weights(myMice: List):
    plt.figure(figsize=(10, 4))
    # f, (ax, bx) = plt.subplots(, 1, sharey="row")
    for idx, mouse_name in enumerate(myMice):

        mouse = load_cache(mouse_name)
        # JB: This line states that a sessions has to have more than 30 trials which can be an issue with our mice. Changed it to ten.
        sessions = [session for session in mouse.sessions if len(session.trials) > 10]
        # print(sessions)
        # plt.subplot(1, 3, idx + 1)
        # number of rows, number of columns, ...
        plt.subplot(1, 5, idx + 1)
        plt.ylim(-0.7, 2.7)
        plot_performance_across_days_weights(sessions)
        plt.title(mouse_name)
        if idx == 0:
            plt.ylabel("Learning Metric")
            # JB:
            plt.xlabel("Weight in g")

    plt.tight_layout()
    plt.savefig(HERE.parent / "plots" / "learning_metric_across_days_WEIGHTS.png")


############################
# Main lines
myMice = ["J011", "J013", "J015", "J016", "J017"]

# for mouse in myMice:
#     cache_mouse_new(mouse)
# scatter_single(myMice)
# scatter_multiple(myMice)
print_weight_changes(myMice)
# statistical_weight_changes()
# plot_distriution_weight_changes()

# plot_boxplot_weight_changes()
# across_days_weights(myMice)
# across_days_weights()
#  cache_mouse_new("J011")
# cache_mouse("J011")


# Must run again
# for mouse in myMice:
    # cache_sessions_weight_changes(mouse)

# broad_learning_metric_weight_changes(myMice)