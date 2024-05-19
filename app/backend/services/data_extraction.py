import os
from services.data_loader import DataLoader
from schemas import schemas
from services.plot_service import plot_raw_optimized, encode_image_to_base64
from models.pipelines import pipelines
import mat73
import numpy as np

TEMP_FOLDER = "temp"
MODEL_FOLDER = "models"
HEAD_SIDE_RIGHT = "D"
HEAD_SIDE_LEFT = "G"
ARM_SIDE_RIGHT = "D"
ARM_SIDE_LEFT = "G"


async def save_file_temporarily(file):
    """
    Save the file temporarily in the TEMP_FOLDER and return the location of the file.

    Parameters:
    ----------
        file (UploadFile): The file to be saved temporarily.

    Returns:
    --------
        str: The location of the file.
    """
    if not os.path.exists(TEMP_FOLDER):
        os.makedirs(TEMP_FOLDER)
    temp_location = f"{TEMP_FOLDER}/{file.filename}"

    with open(temp_location, "wb+") as file_object:
        file_object.write(file.file.read())
    return temp_location


async def remove_temporary_files():
    """
    Remove all the files in the TEMP_FOLDER.

    Returns:
    --------
        None
    """
    if not os.path.exists(TEMP_FOLDER):
        os.makedirs(TEMP_FOLDER)
    for file in os.listdir(TEMP_FOLDER):
        os.remove(f"{TEMP_FOLDER}/{file}")


def extract_data_session(data_location):
    """
    Extract the session information from the data file and return it.

    Parameters:
    ----------
        data_location (str): The location of the data file.

    Returns:
    --------
        schemas.SessionInfo: The session information.
    """
    data_loader = DataLoader(data_location)

    session_info = schemas.SessionInfo(
        patient_id=data_loader.patient_id,
        stroke_side=data_loader.stroke_side,
        session_id=data_loader.session_id,
        signal_left_arm=schemas.SignalInfo(
            duration=data_loader.signal_left_arm_duration,
            first_movement_time=data_loader.signal_left_arm_first_movement_time,
            last_movement_time=data_loader.signal_left_arm_last_movement_time,
            electrodes=data_loader.left_arm_electrodes.tolist(),
        ),
        signal_right_arm=schemas.SignalInfo(
            duration=data_loader.signal_right_arm_duration,
            first_movement_time=data_loader.signal_right_arm_first_movement_time,
            last_movement_time=data_loader.signal_right_arm_last_movement_time,
            electrodes=data_loader.right_arm_electrodes.tolist(),
        ),
        models_available=list(pipelines.keys()),
        models_info=pipelines,
        data_location=data_location,
    )

    # Do the plotting
    for head_side, arm_side in [
        (HEAD_SIDE_RIGHT, ARM_SIDE_RIGHT),
        (HEAD_SIDE_LEFT, ARM_SIDE_LEFT),
        (HEAD_SIDE_RIGHT, ARM_SIDE_LEFT),
        (HEAD_SIDE_LEFT, ARM_SIDE_RIGHT),
    ]:
        raws = data_loader.get_raws(arm_side)
        if raws is not None:
            name = f"{TEMP_FOLDER}/raw_eeg_signal__arm_side={arm_side}__head_side={head_side}.png"
            plot_raw_optimized(
                raws,
                side=head_side,
                electrodes=None,
                subsample_factor=200,
                figsize=(15, 15),
                name=name,
            )
            encoded_string = encode_image_to_base64(name)

            head_side_ = "right" if head_side == HEAD_SIDE_RIGHT else "left"
            arm_side_ = "right" if arm_side == ARM_SIDE_RIGHT else "left"
            signal_arm_info = getattr(session_info, f"signal_{arm_side_}_arm")
            if signal_arm_info:
                setattr(
                    signal_arm_info, f"electrodes_{head_side_}_image", encoded_string
                )

            # os.remove(name)

    return session_info


def mat_to_npy(input_filename, output_folder_path, prefix):
    """
    Convert the .mat file to .npy and save it in the output_folder_path with the prefix.

    Parameters:
    ----------
        input_filename (str): The location of the .mat file.
        output_folder_path (str): The location of the output folder.
        prefix (str): The prefix of the file name.

    Returns:
    --------
        bool: True if the file was successfully converted and saved, False otherwise.
    """

    channels_list = [
        "AF3",
        "AF4",
        "AF7",
        "AF8",
        "AFz",
        "C1",
        "C2",
        "C3",
        "C4",
        "C5",
        "C6",
        "CP1",
        "CP2",
        "CP3",
        "CP4",
        "CP5",
        "CP6",
        "CPz",
        "Cz",
        "F1",
        "F2",
        "F3",
        "F4",
        "F5",
        "F6",
        "F7",
        "F8",
        "FC1",
        "FC2",
        "FC3",
        "FC4",
        "FC5",
        "FC6",
        "FCz",
        "FT7",
        "FT8",
        "Fp1",
        "Fp2",
        "Fpz",
        "Fz",
        "Iz",
        "O1",
        "O2",
        "Oz",
        "P1",
        "P2",
        "P3",
        "P4",
        "P5",
        "P6",
        "P7",
        "P8",
        "P9",
        "P10",
        "PO3",
        "PO4",
        "PO7",
        "PO8",
        "POz",
        "Pz",
        "T7",
        "T8",
        "TP7",
        "TP8",
    ]

    try:
        print(
            "Current file : " + os.path.basename(input_filename) + "\n"
        )  # check the file's name.
        patient_index = os.path.basename(input_filename)[3:8]
        print("patient_index = " + patient_index)

        # Loads data from the matlab structure in dictionary type structures.
        data_dict_patient = mat73.loadmat(input_filename, only_include="Save/patient")
        data_dict_SR = mat73.loadmat(input_filename, only_include="Save/rateDDD")
        data_dict_eeg = mat73.loadmat(input_filename, only_include="Save/VSVM/VS")

        # Only keeps the useful data.
        patient_number = data_dict_patient["Save"]["patient"]["nbr"]
        patho_side = data_dict_patient["Save"]["patient"]["pathoSide"]
        sampling_rate = data_dict_SR["Save"]["rateDDD"]
        DataEEG = data_dict_eeg["Save"]["VSVM"]["VS"]

        for trial in list(DataEEG.keys()):  # Stores each Trial in a specific file.

            # Stores the data for this Trial in a list named current_data
            current_data = [patient_number, patho_side, trial, int(sampling_rate)]
            current_data_dict = DataEEG[trial]

            for side in list(
                current_data_dict.keys()
            ):  # Stores the data for each side in current_data.

                side_specific_dict = current_data_dict[side]
                side_specific_data = [side, side_specific_dict["IdvrMVTrddd"], []]

                # Checking data _______________________________________________________________________________
                # Checks if all the motion_related data files have the same length, if so, computes the expected length of the channel files, stored in expected_channel_length.

                len_AC, len_VAC, len_AC3d, len_VAC3d = (
                    len(side_specific_dict["AngleCoude"]),
                    len(side_specific_dict["VitAngCoude"]),
                    len(side_specific_dict["AngleCoude3D"]),
                    len(side_specific_dict["VitAngCoude3D"]),
                )
                if not (
                    len_AC == len_VAC and len_AC == len_VAC3d and len_AC == len_AC3d
                ):
                    print(
                        "WARNING: données cinématiques de longueurs différentes ! [patient "
                        + patient_number
                        + ", "
                        + trial
                        + ", côté "
                        + side
                        + "]"
                    )
                    test_requirement = False
                else:
                    test_requirement = True
                    expected_channel_length = len_AC * 1024 / sampling_rate

                # Stop checking data __________________________________________________________________________

                already_shown = False
                for channel in channels_list:

                    side_specific_data[2].append([channel, side_specific_dict[channel]])

                    # Checking data _____________________________________________________________________________

                    if test_requirement:
                        if not (
                            already_shown
                        ):  # on envoie le warining une seule fois et pas 64 fois (pour chaque channel)
                            if (
                                abs(
                                    expected_channel_length
                                    - len(side_specific_dict[channel])
                                )
                                > 10
                            ):
                                print(
                                    "WARNING: taille de donnée EEG non cohérente avec la cinématique ! [patient "
                                    + patient_number
                                    + ", "
                                    + trial
                                    + ", côté "
                                    + side
                                    + ", channel "
                                    + channel
                                    + "]"
                                )
                                perc_err = abs(
                                    expected_channel_length
                                    - len(side_specific_dict[channel])
                                ) / len(side_specific_dict[channel])
                                print("Pourcentage d'erreur : " + str(perc_err))
                                print(
                                    "Attention, la même erreur est très probablement relevée pour les autres channels mais ne sera pas repportée ici."
                                )
                                already_shown = True

                    # Stop Checking data ________________________________________________________________________

                side_specific_motion_data = [
                    side_specific_dict["AngleCoude"],
                    side_specific_dict["VitAngCoude"],
                    side_specific_dict["AngleCoude3D"],
                    side_specific_dict["VitAngCoude3D"],
                ]

                side_specific_data.append(side_specific_motion_data)

                current_data.append(side_specific_data)

            current_data.append(patient_index)
            current_data = np.array(current_data, dtype=object)

            filename = prefix + "_" + trial + ".npy"
            np.save(output_folder_path + "/" + filename, current_data)

        print("\nFINISHED.\n")

        return True

    except:
        print(
            "ERROR: Probleme avec l'enregistrement du fichier "
            + os.path.basename(input_filename)
            + ".\n"
        )
        return False
