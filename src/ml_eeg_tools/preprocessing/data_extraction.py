import os
import numpy as np
import mat73


def mat_to_npy(input_filename, output_folder_path, prefix):
    """
    This function takes a .mat file as input and extracts the data from it. It then saves the data in a .npy file in the output_folder_path.
    The data is saved in a list of lists, with the following structure:
    [patient_number, patho_side, trial, sampling_rate, [side, IdvrMVTrddd, [[channel, data],...], [AngleCoude, VitAngCoude, AngleCoude3D, VitAngCoude3D]], patient_index]
    The function also checks the data for consistency and prints warnings if it finds any.

    Parameters
    ----------
    input_filename : str
        The path to the .mat file to be processed.
    output_folder_path : str
        The path to the folder where the .npy files will be saved.
    prefix : str
        The prefix to be used for the .npy files.

    Returns
    -------
    bool
        True if the function ran without errors, False otherwise.
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
