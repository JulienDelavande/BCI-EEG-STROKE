from joblib import load
import numpy as np
import matplotlib.pyplot as plt


def density_on_prediction(raws, predictions, window_size, window_step):
    """
    Reconstruct the prediction on the temporal signal (prediction density).

    Parameters:
    ----------
        raws (mne.io.Raw): The raw data.
        predictions (np.array): The predictions.
        pipeline (dict): The pipeline information.

    Returns:
    --------
        list: The density of the prediction.
    """
    # Tableau de densité de prédiction
    time = raws.times
    window_size_idx = int(window_size * raws.info["sfreq"])
    window_step_idx = int(window_step * raws.info["sfreq"])

    n_windows = len(predictions)
    density = np.zeros(len(time))
    count = np.zeros(len(time))
    for i in range(n_windows):
        start = i * window_step_idx
        end = start + window_size_idx
        density[start:end] += predictions[i]  # /(size/steps)
        count[start:end] += 1
    # Normalisation

    density = [
        density[i] / count[i] if count[i] != 0 else 0 for i in range(len(density))
    ]

    return density



def density_plot(
    raw,
    density,
    threshold=0.5,
    plot_density=False,
    show=True,
    save_path=None,
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
    if show:
        plt.show()
    if save_path is not None:
        plt.savefig(save_path)

    return movement_times
