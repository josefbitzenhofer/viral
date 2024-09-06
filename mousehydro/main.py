import yaml
import sys
import pickle
import math
import os.path
import pandas as pd
from datetime import date
from pathlib import Path
from pydantic import BaseModel
from typing import Any, List

from googleapiclient.discovery import build
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request

HERE = Path(__file__).parent
sys.path.append(str(HERE.parent))

from viral.single_session import (
    load_data,
    remove_bad_trials,
    summarise_trial,
)
from viral.models import (
    TrialInfo,
    TrialSummary,
)
from viral.gsheets_importer import (
    gsheet2df,
    CRED_PATH,
    TOKEN_PATH
)
from viral.constants import(
    DATA_PATH,
    SPREADSHEET_ID
)
from josef.analysis import (
    calculate_weight_change,
    get_baseline_weight
)

class SessionSummaryHydration(BaseModel):
    name: str
    trials: List[TrialSummary]
    weight: float                       
    weight_change: float                
    session_trials_all: int            
    session_trials_rewarded: int       
    session_trials_unrewarded: int     
    session_trials_licked: int

class MouseCalculation(BaseModel):
    mouse_name: str
    session_weight: float
    volume_needed: float
    n_licks: int
    reward_size: float
    reward_received: float
    top_up_needed: float

with open(HERE / "config.yaml", "r") as file:
    settings = yaml.safe_load(file)

REWARD_SIZE = settings["REWARD_SIZE"]
WATER_PER_DAY = settings["WATER_PER_DAY"]
MICE_LIST = settings["MICE_LIST"]
DATE_AUTO = settings["DATE_AUTO"]
DATE_CUSTOM = settings["DATE_CUSTOM"]
CREATE_CSV = settings["CREATE_CSV"]

# Calculations #
def calculate_reward_received(n_licks: int) -> float:
    """Return reward received"""
    reward_received = n_licks * REWARD_SIZE
    return reward_received

def calculate_volume_needed(mouse_weight: float) -> float:
    """Return volume needed"""
    mouse_weight_kg = mouse_weight * 0.001             # In the spreadsheet, the mouse weight is in g
    volume_needed = mouse_weight_kg * WATER_PER_DAY
    return volume_needed

def calculate_top_up_needed(volume_needed: float, reward_received: float) -> float:
    """
    Return top up needed
    Args:
        volume needed (float):      The volume of water needed for the mouse on this day according to weight
        reward_received (float):    The volume of water received by the mouse on this day by reward licking in trials
    Returns:
        float:                      The top up needed for the mouse on this day
    """
    top_up_needed = volume_needed - reward_received
    return top_up_needed

# GETTING DATA #
def get_current_date() -> str:
    """Return current date"""
    current_date = str(date.today())
    return current_date

def create_session_summary_hydration(mouse_name: str, date: str) -> SessionSummaryHydration | None:
    """
    Return SessionSummaryHydration
    Args:
        mouse_name (str):       The mouse's name (e.g. 'J011')
        date (str):             The date of the session
    Returns:
        BaseModel:              A summary of the session for the mouse
    Raises:
        FileNotFoundError:      If the trial files could not be found
    """
    mouse_metadata =  gsheet2df(SPREADSHEET_ID, mouse_name, 1)   # spreadsheet_id, sheet_name, header_row
    baseline_weight = float(get_baseline_weight(mouse_name))
    session_summary = None
    for _, row in mouse_metadata.iterrows():
        trials: List[TrialInfo] = []
        if row["Date"] == date:
            session_number = row["Session Number"]
            # Dealing with split sessions (e.g. '1 + 2')
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
                        trials_split = remove_bad_trials(trials_split)
                        trials.extend(trials_split)
                    except FileNotFoundError as e:
                        print(
                            f"ERROR: ExceptionError: There went something wrong getting the trials."
                        )
                        print(
                            f"Mouse:'{mouse_name}'. Date:'{row["Date"]}'. Session: '{session_number}'."
                        )
                        print(f"Exception: '{e}'.")
            else:
            # Standard case, just one session for a day
                session_number = (
                    row["Session Number"]
                    if len(row["Session Number"]) == 3
                    else f"00{row['Session Number']}"
                )

                session_path = DATA_PATH / mouse_name / row["Date"] / session_number

                try:
                    trials = load_data(session_path)
                    trials = remove_bad_trials(trials)
                except FileNotFoundError as e:
                    print(
                        f"ERROR: ExceptionError: There went something wrong getting the trials."
                    )
                    print(
                        f"Mouse:'{mouse_name}'. Date:'{row["Date"]}'. Session: '{session_number}'."
                    )
                    print(f"Exception: '{e}'.")
            
            # To make this usable
            trials_summarised: List[TrialSummary] = []
            for trial in trials:
                trials_summarised.append(summarise_trial(trial))
            
            # Count the trials and the licks
            session_trials_all = 0
            session_trials_rewarded = 0
            session_trials_unrewarded = 0
            session_trials_licked = 0
            for trial_summary in trials_summarised:
                print(f"Type: {type(trial)}")
                session_trials_all += 1
                if trial_summary.rewarded:
                    session_trials_rewarded += 1
                if trial_summary.reward_drunk:
                    session_trials_licked += 1
                else:
                    session_trials_unrewarded +=1

            # Here, we get the weight at the day of the session, the baseline weight to then calculate the relative weight change
            try:
                session_weight = float(row["Weight"])
            except:
                print(f"ERROR: ExceptionError: There went something wrong getting 'session_weight'.")
            try:
                weight_change = float(calculate_weight_change(baseline_weight, session_weight))
            except:
                print(f"ERROR: ExceptionError: There went someting wrong getting 'weight_change'.")
            try:
                session_summary = SessionSummaryHydration(
                            name=row["Type"],
                            trials=trials_summarised,
                            weight=float(session_weight), 
                            weight_change=float(weight_change),
                            session_trials_all=int(session_trials_all),
                            session_trials_rewarded =int(session_trials_rewarded),
                            session_trials_unrewarded=int(session_trials_unrewarded),
                            session_trials_licked =int(session_trials_licked),
                        )
                print(f"NOTE: SessionSummaryHydration successfully created for '{mouse_name}' on '{date}'.")
            except:
                print(f"ERROR: ExceptionError: There went something wrong creating 'SessionSummary'.")
    return session_summary

def get_n_licks(session_summary: SessionSummaryHydration) -> int:
    """Return number of licks"""
    return session_summary.session_trials_licked

def get_session_weight(session_summary: SessionSummaryHydration) -> float:
    """Return the mouse's weight on the day of the session"""
    return session_summary.weight

# INTERACTING WITH THE SPREADSHEET #
def build_service() -> Any:
    """Return Google Sheet service"""
    # Unfortunately had to copy these lines of code from the "gsheets_importer.py" file as the according function only returns a dataframe
    creds = None
    # JB: Had to change the scope so it is no longer read only!
    # SCOPES = ["https://www.googleapis.com/auth/spreadsheets.readonly"]
    SCOPES = ["https://www.googleapis.com/auth/spreadsheets"]
    # The file token.pickle stores the user's access and refresh tokens, and is
    # created automatically when the authorization flow completes for the first
    # time.

    if os.path.exists(TOKEN_PATH):
        with open(TOKEN_PATH, "rb") as token:
            creds = pickle.load(token)
    # If there are no (valid) credentials available, let the user log in.
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(CRED_PATH, SCOPES)
            # JR - this is needed to authenticate through ssh
            # flow = InstalledAppFlow.from_client_secrets_file("credentials.json", SCOPES)
            creds = flow.run_local_server(port=0)
        # Save the credentials for the next run
        with open(TOKEN_PATH, "wb") as token:
            pickle.dump(creds, token)

    return build("sheets", "v4", credentials=creds, cache_discovery=False)

def get_row_number(sheet_name: str, date: str) -> int | None:
    """"Return row number of according session in the worksheet"""
    row_name = date

    service = build_service()
    sheet = service.spreadsheets()
    result = sheet.values().get(spreadsheetId=SPREADSHEET_ID, range=sheet_name).execute()
    values = result.get('values', [])

    try:
        for idx, row in enumerate(values):
            if row[0] == row_name:
                return idx + 1                  # Apparently, Google starts indexing with '1'
    except:
        print(f"ERROR: ExceptionError: Could not get the row number for sheet '{sheet_name}', row '{row_name}'.")
    
    # Return 'None' if the data is not found
    return None

def dump_top_up_needed(mouse_name: str, date: str, top_up_needed: float) -> None:
    """Dumps the needed top-up volume into the spreadsheet"""
    # Reusing 'gsheets_importer.py' code:
    # Call the Sheets API
    service = build_service()
    
    # Select our mouse
    sheet_name = mouse_name
    # If the sheet name contains numbers, it must be enclosed in single quotes
    if any(char.isdigit() for char in sheet_name):
        sheet_name = f"'{sheet_name}'"
    
    # Setting the range to the 'top up' cell of the session
    column_letter = "D"                             # This is static according to the column letter for "Top up" in the template 
    row_number = get_row_number(sheet_name, date)   # This is dynamic: the session's date

    # Checks, whether the row number could be retrieved
    if not row_number:
        print(f"ERROR: Are you sure, the row for session '{date}' in worksheet '{sheet_name}' exists?")

    # "A1 notation" = e.g. 'Sheet1!A1:D5' -> 'sheet_name!column_namerow_number'
    # Little cheat here, the worksheet name should not have single quotes, so instead, I am using the mouse_name
    # Question, is it 'alphanumeric combination'? https://developers.google.com/sheets/api/guides/concepts#cell
    range_name = f"{sheet_name}!{column_letter}{row_number}"

    # Set the value to the top up needed
    value = [[f"{top_up_needed} mL"]]

    body = {
        "values": value
    }

    # These lines actually update the value in the spreadsheet
    try:
        result = service.spreadsheets().values().update(spreadsheetId=SPREADSHEET_ID, range=range_name, valueInputOption="USER_ENTERED", body=body).execute()
        print(f"NOTE: Updated value for session '{date}' in worksheet '{sheet_name}' (range = '{range_name}') to '{top_up_needed} mL'.")
        print(f"NOTE: {result.get('updatedCells')} cells updated.")
    except:
        print(f"ERROR: ExceptionError: Could not update value for session '{date}' in worksheet '{sheet_name}'. ('{range_name}')")
    
def main() -> None:
    """Main code"""
    print(f"NOTE: Calculating top ups for '{MICE_LIST}'.")
    if DATE_AUTO:
        # If 'DATE_AUTO' is set to 'True', the script takes the current date
        date = str(get_current_date())
    if not DATE_AUTO:
        date = str(DATE_CUSTOM)
    print(f"NOTE: Automatic date selection was set to '{DATE_AUTO}'. The date selected is '{date}'.")
    print(f"NOTE: Automatic Excel sheet creation was set to '{CREATE_CSV}'.")
    CSV_mouse_calculations = []
    CSV_data_list = []
    for mouse_name in MICE_LIST:
        print(f"NOTE: Start calculating and updating top up for '{mouse_name}'.")
        session_summary = create_session_summary_hydration(mouse_name, date)
        if session_summary:
            n_licks = get_n_licks(session_summary)
            mouse_weight = get_session_weight(session_summary)
        reward_received = calculate_reward_received(n_licks)
        volume_needed = calculate_volume_needed(mouse_weight)
        top_up_needed = calculate_top_up_needed(volume_needed, reward_received)
        top_up_needed = math.ceil(top_up_needed * 20) / 20
        dump_top_up_needed(mouse_name, date, top_up_needed)
        if CREATE_CSV:
            CSV_mouse_calculations.append(MouseCalculation(
                mouse_name=str(mouse_name),
                session_weight=float(mouse_weight),
                volume_needed=float(volume_needed),
                n_licks=int(n_licks),
                reward_size=float(REWARD_SIZE),
                reward_received=float(reward_received),
                top_up_needed=float(top_up_needed)
            ))
    if CREATE_CSV:
        for mouse in CSV_mouse_calculations:
            CSV_data_list.append(mouse.__dict__)
        CSV_data_dataframe = pd.DataFrame(
            data=CSV_data_list,
            columns = ["mouse_name", "session_weight", "volume_needed", "n_licks", "reward_size", "reward_received", "top_up_needed"]
            )
        csv_directory = HERE / "csv"
        csv_file_directory =  os.path.join(csv_directory, f"{date}.csv")
        os.makedirs(csv_directory, exist_ok=True)
        with open(
            csv_file_directory,
            "w",
        ) as f:
            pd.DataFrame.to_csv(self=CSV_data_dataframe, path_or_buf=csv_file_directory)

if __name__ == "__main__":
    main()



# Test
# dump_top_up_needed("J011", "2024-05-14", 3.0)


    # session_summary = create_session_summary("J011", "2024-06-14")
    # n_licks = get_n_licks(session_summary)
    # session_weight = get_session_weight(session_summary)
    # print(f"'n_licks' = {n_licks}")
    # volume_needed = calculate_volume_needed(session_weight)
    # print(volume_needed)
    # reward_received = calculate_reward_received(n_licks)
    # print(reward_received)
    # top_up_needed = calculate_top_up_needed(volume_needed, reward_received)
    # print(top_up_needed)
    # # Make if DATE_TODAY etc.
    # # Make it write it into the right cell
    # # Comment the functions
    # # Let us know more

    # create_session_summary_hydration(mouse_name="J011", date="2024-06-17")
