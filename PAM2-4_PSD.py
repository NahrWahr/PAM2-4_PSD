# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "marimo",
#     "numpy",
#     "plotly",
#     "scipy",
#     "pandas",
# ]
# ///

import marimo

__generated_with = "0.19.4"
app = marimo.App(width="medium")


@app.cell
def _():
    import numpy as np
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    from scipy import signal
    import marimo as mo
    return go, make_subplots, mo, np, signal


@app.cell
def _(mo):
    # --- UI CONTROLS ---
    baud_slider = mo.ui.slider(10, 56, value=28, label="Baud Rate (GBd)")

    # 1. Driver Characteristics
    risetime_slider = mo.ui.slider(0.01, 0.5, step=0.05, value=0.1, label="Driver Rise Time (UI)")

    # 2. Channel Characteristics (Transmission Line)
    bw_slider = mo.ui.slider(5, 50, value=20, label="Channel Bandwidth (GHz)")

    # 3. Noise Parameters
    tx_noise_slider = mo.ui.slider(0.0, 0.2, step=0.01, value=0.01, label="TX Noise (jitter/amp)")
    rx_noise_slider = mo.ui.slider(0.0, 0.2, step=0.01, value=0.01, label="RX Noise (thermal)")

    controls = mo.vstack([
        mo.md("## SERDES System: Driver + Channel + Noise"),
        mo.md("Compare the ideal 'Digital' data (dotted) vs the simulated 'Analog' signal at the receiver."),
        mo.hstack([baud_slider, risetime_slider, bw_slider]),
        mo.hstack([tx_noise_slider, rx_noise_slider])
    ])
    return (
        baud_slider,
        bw_slider,
        controls,
        risetime_slider,
        rx_noise_slider,
        tx_noise_slider,
    )


@app.cell
def _(
    baud_slider,
    bw_slider,
    np,
    risetime_slider,
    rx_noise_slider,
    signal,
    tx_noise_slider,
):
    # --- SIMULATION ENGINE ---

    # Parameters
    Rb = baud_slider.value * 1e9
    fs = Rb * 16  # 16x Oversampling
    num_symbols = 10**4

    # 1. Generate Ideal Symbols & Square Waves
    # PAM2
    bits_pam2 = np.random.randint(0, 2, num_symbols) * 2 - 1
    ideal_pam2 = np.repeat(bits_pam2, 16) # Perfect Square Wave

    # PAM4 (Normalized)
    syms_pam4 = np.random.randint(0, 4, num_symbols)
    map_pam4 = np.array([-1.0, -0.333, 0.333, 1.0])
    levels_pam4 = map_pam4[syms_pam4]
    ideal_pam4 = np.repeat(levels_pam4, 16) # Perfect Square Wave

    # 2. Apply Driver Slew Rate (Rise Time)
    # We use a Gaussian filter to smooth the square edges
    sigma = 16 * risetime_slider.value # 16 samples per UI
    win_len = int(6 * sigma)
    if win_len % 2 == 0: win_len += 1
    window = signal.windows.gaussian(win_len, std=sigma)
    window /= np.sum(window)

    # Convolve to get "Driver Output"
    pam2_driver = signal.convolve(ideal_pam2, window, mode='same')
    pam4_driver = signal.convolve(ideal_pam4, window, mode='same')

    # 3. Add TX Noise
    tx_noise_2 = np.random.normal(0, tx_noise_slider.value, len(pam2_driver))
    pam2_tx = pam2_driver + tx_noise_2
    # (Skipping PAM4 noise logic for brevity, focusing on PAM2/RX mainly)
    pam4_tx = pam4_driver + np.random.normal(0, tx_noise_slider.value, len(pam4_driver))

    # 4. Apply Channel LPF (Transmission Line)
    fc = bw_slider.value * 1e9
    sos = signal.butter(1, fc, 'low', fs=fs, output='sos')

    pam2_channel = signal.sosfilt(sos, pam2_tx)
    pam4_channel = signal.sosfilt(sos, pam4_tx)

    # 5. Add RX Noise
    rx_noise = np.random.normal(0, rx_noise_slider.value, len(pam2_channel))

    pam2_rx = pam2_channel + rx_noise
    pam4_rx = pam4_channel + rx_noise

    # Time Vector
    t = np.arange(len(pam2_rx)) / fs

    # 6. PSD & Clock Recovery (On Final RX Signal)
    f, psd_pam2 = signal.welch(pam2_rx, fs, nperseg=1024)
    _, psd_pam4 = signal.welch(pam4_rx, fs, nperseg=1024)

    psd_pam2_db = 10 * np.log10(psd_pam2 + 1e-15)
    psd_pam4_db = 10 * np.log10(psd_pam4 + 1e-15)

    edge_signal = np.abs(np.gradient(pam2_rx))
    f_edge, psd_edge_linear = signal.welch(edge_signal, fs, nperseg=2048)
    psd_edge_db = 10 * np.log10(psd_edge_linear + 1e-15)

    ref_clock = np.sin(2 * np.pi * Rb * t)
    corr_raw = np.corrcoef(pam2_rx, ref_clock)[0, 1]
    corr_edge = np.corrcoef(edge_signal, ref_clock)[0, 1]
    return (
        Rb,
        corr_edge,
        corr_raw,
        f,
        f_edge,
        ideal_pam2,
        ideal_pam4,
        pam2_rx,
        pam4_rx,
        psd_edge_db,
        psd_pam2_db,
        psd_pam4_db,
        t,
    )


@app.cell
def _(
    Rb,
    f,
    f_edge,
    go,
    ideal_pam2,
    ideal_pam4,
    make_subplots,
    np,
    pam2_rx,
    pam4_rx,
    psd_edge_db,
    psd_pam2_db,
    psd_pam4_db,
    t,
):
    # --- VISUALIZATION ---

    fig = make_subplots(
        rows=3, cols=1, 
        subplot_titles=(
            "Time Domain: Ideal TX (Dashed) vs Noisy RX (Solid)", 
            "RX Power Spectral Density", 
            "Clock Recovery Spectrum"
        ),
        vertical_spacing=0.12
    )

    # 1. Time Domain
    limit_idx = 300 

    # Plot Ideal Square Wave (White, Dashed)
    fig.add_trace(go.Scatter(x=t[:limit_idx]*1e9, y=ideal_pam2[:limit_idx], name="Ideal Digital Data", 
                             line=dict(color="white", width=1, dash="dash"), opacity=0.7), row=1, col=1)
    fig.add_trace(go.Scatter(x=t[:limit_idx]*1e9, y=ideal_pam4[:limit_idx], name="Ideal Digital Data", 
                             line=dict(color="white", width=1, dash="dash"), visible='legendonly', opacity=0.7), row=1, col=1)

    # Plot RX Signal (Colored)
    fig.add_trace(go.Scatter(x=t[:limit_idx]*1e9, y=pam2_rx[:limit_idx], name="RX Signal (PAM2)", 
                             line=dict(color="#00CC96", width=2)), row=1, col=1)

    # (Optional: Toggle PAM4 visibility via legend to avoid clutter, kept hidden by default here)
    fig.add_trace(go.Scatter(x=t[:limit_idx]*1e9, y=pam4_rx[:limit_idx], name="RX Signal (PAM4)", 
                             line=dict(color="#EF553B", width=2), visible='legendonly'), row=1, col=1)

    # 2. PSD Comparison
    freq_limit_idx = np.argmax(f > 2.5 * Rb)
    fig.add_trace(go.Scatter(x=f[:freq_limit_idx]/1e9, y=psd_pam2_db[:freq_limit_idx], 
                             name="PAM2 Spectrum", line=dict(color="#00CC96")), row=2, col=1)
    fig.add_trace(go.Scatter(x=f[:freq_limit_idx]/1e9, y=psd_pam4_db[:freq_limit_idx], 
                             name="PAM4 Spectrum", line=dict(color="#EF553B")), row=2, col=1)
    fig.add_vline(x=Rb/2e9, line_dash="dot", line_color="yellow", row=2, col=1, annotation_text="Nyquist")

    # 3. Clock Recovery
    freq_edge_limit = np.argmax(f_edge > 1.5 * Rb)
    fig.add_trace(go.Scatter(x=f[:freq_edge_limit]/1e9, y=psd_pam2_db[:freq_edge_limit], 
                             name="RX Spectrum", line=dict(color="gray", width=1), opacity=0.5), row=3, col=1)
    fig.add_trace(go.Scatter(x=f_edge[:freq_edge_limit]/1e9, y=psd_edge_db[:freq_edge_limit], 
                             name="Edge Spectrum", line=dict(color="#FFA15A", width=2)), row=3, col=1)

    fig.update_layout(
        height=900,
        template="plotly_dark",
        xaxis_title="Time (ns)",
        xaxis2_title="Frequency (GHz)",
        xaxis3_title="Frequency (GHz)",
        yaxis_title="Amplitude (V)",
        yaxis2_title="Power (dB/Hz)",
        yaxis3_title="Power (dB/Hz)",
    )
    pass
    return (fig,)


@app.cell
def _(controls, corr_edge, corr_raw, fig, mo):
    # --- METRICS DISPLAY ---

    score_card = mo.md(
        f"""
        ### Correlation Analysis (Lock Detector)

        To recover the clock, the receiver correlates the incoming signal with a local oscillator.

        * **Raw Data Correlation:** `{corr_raw:.5f}`  
            _The raw data is random. It has **no** correlation with a sine wave at the baud rate._

        * **Edge Signal Correlation:** `{corr_edge:.5f}`  
            _After edge detection (non-linearity), a strong spectral line appears. High correlation means the PLL can lock._
        """
    )

    mo.vstack([
        controls,
        fig,
        score_card
    ])
    return


if __name__ == "__main__":
    app.run()
