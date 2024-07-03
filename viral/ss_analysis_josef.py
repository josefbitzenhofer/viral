from pathlib import Path
import random
from typing import Dict, List
import matplotlib.pyplot as plt
import numpy as np
from models import TrialInfo
# We need to import Pandas
import pandas as pd

from utils import (
    degrees_to_cm,
    get_speed_positions,
    shaded_line_plot,
    licks_to_position,
)

from constants import DATA_PATH, ENCODER_TICKS_PER_TURN, WHEEL_CIRCUMFERENCE
import seaborn as sns
from scipy.stats import ttest_ind

sns.set_theme(context="talk", style="ticks")

DATA_PATH = Path("/Volumes/MarcBusche/James/Behaviour/online/Subjects")
MOUSE = "J011"
DATE = "2024-06-10"
SESSION_NUMBER = "001"
SESSION_PATH = DATA_PATH / MOUSE / DATE / SESSION_NUMBER

from single_session import (
    load_data,
    plot_lick_raster,
    get_anticipatory_licking,
    plot_trial_length,
    get_percent_timedout,
    disparity,
    plot_previous_trial_dependent_licking,
    plot_rewarded_vs_unrewarded_licking,
    plot_licking_habituation,
    plot_position_whole_session,
    plot_speed,
    remove_bad_trials,
    summarise_trial,

)

def AZspeed_ss_bp(trials: List[TrialInfo], sampling_rate) -> None:
    plt.figure()
    rewarded: List[SpeedPosition] = []
    not_rewarded: List[SpeedPosition] = []

    prev_rewarded = []
    prev_unrewarded = []

    # Defining first_position = 170 and last_position = 180 for the Anticipatory zone (AZ)
    first_position = 170
    last_position = 180
    step_size = 5
    for idx, trial in enumerate(trials):

        position = degrees_to_cm(np.array(trial.rotary_encoder_position))

        speed = get_speed_positions(
            position=position,
            first_position=first_position,
            last_position=last_position,
            step_size=step_size,
            sampling_rate=sampling_rate,
        )

        if trial.texture_rewarded:
            rewarded.append(speed)
        else:
            not_rewarded.append(speed)

        if trials[idx - 1].texture_rewarded:
            prev_rewarded.append(speed)
        else:
            prev_unrewarded.append(speed)

    # Accessing the 3rd item in each nested list -> speed, see ChatGPT
    # Rappel ! We start counting from 0, 1, 2, ... -> [2] is 3rd!
    # (1) Create two empty lists for our new array
    AZ_speed_ss_rewarded = []
    AZ_speed_ss_notrewarded = []

    # Iterate over each inner list in the rewarded list
    for inner_list in rewarded:
        # Iterate over each SpeedPosition object in the inner list
        for speed_position in inner_list:
            # Append the speed of the current SpeedPosition object to the speeds list
            AZ_speed_ss_rewarded.append(speed_position.speed)
    
    for inner_list in not_rewarded:
        for speed_position in inner_list:
            # Append the speed of the current SpeedPosition object to the speeds list
            AZ_speed_ss_notrewarded.append(speed_position.speed)

    # print(AZ_speed_ss_rewarded)
    # print(AZ_speed_ss_notrewarded)

    AZ_speed_ss_rewarded_array = np.array(AZ_speed_ss_rewarded)
    AZ_speed_ss_notrewarded_array = np.array(AZ_speed_ss_notrewarded)

    # Use "concatenation" see ChatGPT
    AZ_speed_ss_array = np.concatenate([AZ_speed_ss_rewarded, AZ_speed_ss_notrewarded])
    condition_labels = ["Rewarded"] * len(AZ_speed_ss_rewarded) + ["Not Rewarded"] * len(AZ_speed_ss_notrewarded)

    sns.boxplot(x=condition_labels, y=AZ_speed_ss_array, palette="Blues", width=0.5)
    plt.xlabel("Condition")
    plt.ylabel("Speed in AZ (cm/s)")
    plt.tight_layout()

    # Rappel!
    plt.show()

def AZspeed_ss_db(trials: List[TrialInfo], sampling_rate) -> None:
# Massively reusing code from the AZspeed_ss_bp subroutine
    plt.figure()

    rewarded: List[SpeedPosition] = []
    not_rewarded: List[SpeedPosition] = []

    prev_rewarded = []
    prev_unrewarded = []

    # Defining first_position = 170 and last_position = 180 for the Anticipatory zone (AZ)
    first_position = 170
    last_position = 180
    step_size = 5
    for idx, trial in enumerate(trials):

        position = degrees_to_cm(np.array(trial.rotary_encoder_position))

        speed = get_speed_positions(
            position=position,
            first_position=first_position,
            last_position=last_position,
            step_size=step_size,
            sampling_rate=sampling_rate,
        )

        if trial.texture_rewarded:
            rewarded.append(speed)
        else:
            not_rewarded.append(speed)

        if trials[idx - 1].texture_rewarded:
            prev_rewarded.append(speed)
        else:
            prev_unrewarded.append(speed)

    # Accessing the 3rd item in each nested list -> speed, see ChatGPT
    # Rappel ! We start counting from 0, 1, 2, ... -> [2] is 3rd!
    # (1) Create two empty lists for our new array
    AZspeed_ss_rewarded = []
    AZspeed_ss_notrewarded = []

    # Iterate over each inner list in the rewarded list
    for inner_list in rewarded:
        # Iterate over each SpeedPosition object in the inner list
        for speed_position in inner_list:
            # Append the speed of the current SpeedPosition object to the speeds list
            AZspeed_ss_rewarded.append(speed_position.speed)
    
    for inner_list in not_rewarded:
        for speed_position in inner_list:
            # Append the speed of the current SpeedPosition object to the speeds list
            AZspeed_ss_notrewarded.append(speed_position.speed)

    # AZspeed_ss_rewarded_array = np.array(AZspeed_ss_rewarded)
    # AZspeed_ss_notrewarded_array = np.array(AZspeed_ss_notrewarded)

    # AZ_speed_ss_array = np.concatenate([AZ_speed_ss_rewarded, AZ_speed_ss_notrewarded])
    # condition_labels = ["Rewarded"] * len(AZ_speed_ss_rewarded) + ["Not Rewarded"] * len(AZ_speed_ss_notrewarded)

    # I create a dictionary and transform it in "long-shape" (ChatGPT)
    AZspeed_ss_dict = {
        "Condition": ["Rewarded"] * len(AZspeed_ss_rewarded) + ["Unrewarded"] * len(AZspeed_ss_notrewarded),
        "Speed in AZ (cm/s)": AZspeed_ss_rewarded + AZspeed_ss_notrewarded
    }
    # Then, I create the correct dataframe
    AZspeed_ss_df = pd.DataFrame(AZspeed_ss_dict)

    # Checking if the dataframe looks ok
    # print(speed_AZ_df.head()) # I think this was ok

    # Creating the distribution plot
    # Setting colors
    AZspeed_ss_colors = ["#9B00AE", "#87CEFA"]
    AZspeed_ss_colorpalette = sns.set_palette(sns.color_palette(AZspeed_ss_colors))
    sns.histplot(data=AZspeed_ss_df, x="Speed in AZ (cm/s)", hue="Condition", multiple="layer", stat="density", palette = AZspeed_ss_colorpalette)
    # sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
    plt.ylabel("Frequency")
    plt.legend([],[], frameon=False)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    trials = load_data(SESSION_PATH)
    disparity(trials)

    print(f"Number of trials: {len(trials)}")
    print(f"Percent Timed Out: {get_percent_timedout(trials)}")
    trials = remove_bad_trials(trials)

    # az_speed_histogram([summarise_trial(trial) for trial in trials])

    print(f"Number of trials after removing timed out: {len(trials)}")

    # plot_rewarded_vs_unrewarded_licking(trials)
    # plot_speed(trials, sampling_rate=30)
    # plot_trial_length(trials)

    # plot_previous_trial_dependent_licking(trials)
    
    # plot_speed_AZ_boxplot(trials, sampling_rate=30)
    # plot_speed_AZ_bp2(trials, sampling_rate=30)
    # plot_speed_AZ_dbp(trials, sampling_rate=30)

    AZspeed_ss_bp(trials, sampling_rate=30)
    AZspeed_ss_db(trials, sampling_rate=30)