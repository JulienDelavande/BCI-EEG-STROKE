import matplotlib.pyplot as plt
import numpy as np
import base64
from services import data_manipulation_service


def plot_raw_optimized(
    raw,
    side=None,
    electrodes=None,
    subsample_factor=200,
    figsize=(15, 15),
    name="raw_eeg_signal.png",
):
    """
    Plot the raw EEG signal.

    Parameters:
    ----------
        raw (mne.io.Raw): The raw data.
        side (str): The side of the head ('D' for right and 'G' for left).
        electrodes (list): The list of electrodes to include.
        subsample_factor (int): The subsample factor.
        figsize (tuple): The size of the figure.
        name (str): The name of the file to save.

    Returns:
    --------
        str: The name of the file saved.
    """
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    fig.suptitle("Raw EEG Signal")
    fig.tight_layout()
    fig.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)

    # Récupérer les canaux EEG
    picks_eeg, picks_stim = data_manipulation_service.get_channels(
        raw, side, electrodes
    )

    # Sampler les données
    data, times = raw[picks_eeg, :]
    data = data[:, ::subsample_factor]
    times = times[::subsample_factor]

    if len(picks_stim) > 2:
        picks_stim = picks_stim[:2]

    data_stim, _ = raw[picks_stim, :]
    data_stim = data_stim[:, ::subsample_factor]

    # Remettre à l'échelle les données stim en centrant et réduisant sur chaque canal
    max_values = np.max(data_stim, axis=1)
    max_values[max_values == 0] = 1  # Éviter la division par zéro
    mean_values = np.mean(data_stim, axis=1)
    data_stim -= mean_values[:, np.newaxis]  # Centrer
    data_stim /= max_values[:, np.newaxis]  # Réduire
    data_stim *= 1e-4  # Mettre à l'échelle pour que les données stim soient de l'ordre de grandeur des données EEG

    # Concaténer les données EEG et les données stim
    data = np.concatenate((data, data_stim), axis=0)
    picks = np.concatenate((picks_eeg, picks_stim))

    # Dessiner les données
    offset = 100 * np.arange(len(picks))
    ax.plot(times, data.T * 1e6 + offset)

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Voltage (uV)")
    ax.set_yticks(offset)
    ax.set_yticklabels(raw.info["ch_names"][i] for i in picks)
    ax.set_xlim([times[0], times[-1]])
    ax.set_ylim([offset[0] - 100, offset[-1] + 200])

    # Ajouter une échelle verticale pour les microvolts
    scale_length_uv = 100  # 100 uV pour l'échelle
    scale_x_position = (
        times[-1] * 0.02
    )  # Position horizontale de l'échelle à 98% du max de l'axe des temps
    scale_y_start = offset[-1] + 50  # Position de début verticale de l'échelle
    scale_y_end = scale_y_start + scale_length_uv  # Fin de l'échelle

    # Dessiner l'échelle verticale
    ax.plot(
        [scale_x_position, scale_x_position],
        [scale_y_start, scale_y_end],
        color="k",
        lw=2,
    )
    ax.text(
        scale_x_position + 1,
        scale_y_start + scale_length_uv / 2,
        f"{scale_length_uv} uV",
        ha="left",
        va="center",
        color="k",
    )

    # sauvegarde des données
    plt.savefig(name)
    plt.close()

    return name


def encode_image_to_base64(image_path):
    """
    Encode the image to base64.

    Parameters:
    ----------
        image_path (str): The path of the image.

    Returns:
    --------
        str: The encoded image.
    """
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode("utf-8")
    return encoded_string


def density_plot(
    raw,
    density,
    threshold=0.5,
    plot=True,
    plot_density=False,
    name="temp/density_plot.png",
    channel="VAC",
    channel_label="Speed VAC",
):
    """
    Plot the density of the prediction.

    Parameters:
    ----------
        raw (mne.io.Raw): The raw data.
        density (list): The density of the prediction.
        threshold (float): The threshold for the prediction.
        plot (bool): Whether to plot the density.
        plot_density (bool): Whether to plot the density.
        name (str): The name of the file to save.
        channel (str): The channel to plot.
        channel_label (str): The label of the channel.

    Returns:
    --------
        list: The movement times."""
    sfreq = raw.info["sfreq"]
    # Mouvement prédits
    movement_times = []
    start = None
    for i, value in enumerate(density):
        if value > threshold and start is None:
            start = i
        elif value <= threshold and start is not None:
            end = i
            movement_times.append((start / sfreq, end / sfreq))
            start = None

    # Affichage des résultats
    if plot:
        fig, ax = plt.subplots(1, 1, figsize=(20, 5))

        # Affichage de la vitesse du bras
        ax.plot(raw.times, raw[channel][0][0], label=channel_label)
        ax.set_title("Prédiction pour un seuil de {:.2f}".format(threshold))
        ax.set_ylabel(channel_label)
        ax.set_xlabel("Time (s)")
        ax.legend(loc="upper right")

        if plot_density:
            # Affichage de la densité de prédiction
            ax2 = ax.twinx()
            ax2.plot(raw.times, density, label="Density", color="grey", alpha=0.5)
            ax2.set_ylabel("Density")
            ax2.set_ylim(-1, 2)
            ax2.legend(loc="lower right")

        # Ajout des bandes de couleur pour chaque intervalle de mouvement
        for start, end in movement_times:
            ax.axvspan(start, end, color="orange", alpha=0.4)

        plt.savefig(name)

    return movement_times
