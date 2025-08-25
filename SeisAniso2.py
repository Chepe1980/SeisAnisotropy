import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.optimize import curve_fit
from scipy import signal
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import io
import warnings
warnings.filterwarnings('ignore')

# Set page configuration
st.set_page_config(
    page_title="VTI Anisotropy Analysis",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== SIDEBAR CONFIGURATION ====================
st.sidebar.title("Configuration Parameters")

# File upload
uploaded_file = st.sidebar.file_uploader("Upload CSV file", type=["csv"], 
                                        help="Upload well log data with columns for VP, VS, RHOB, and DEPTH")

# Debug info for uploaded file
if uploaded_file is not None:
    st.sidebar.info(f"File: {uploaded_file.name} ({uploaded_file.size} bytes)")

# Initialize default column names
vp_col = 'VP'
vs_col = 'VS'
rho_col = 'RHOB'
depth_col = 'DEPTH'
gr_col = None
phi_col = None
sw_col = None
rt_col = None
vclay_col = None

# Wavelet parameters
st.sidebar.header("Wavelet Settings")
wavelet_type = st.sidebar.selectbox("Wavelet Type", ["ricker", "bandpass"], index=0,
                                   help="Ricker: zero-phase wavelet with specified frequency. Bandpass: filtered impulse with frequency range.")
wavelet_frequency = st.sidebar.slider("Wavelet Frequency (Hz)", 10, 100, 30,
                                     help="Central frequency for Ricker wavelet")
freq_low = st.sidebar.slider("Low Frequency (Hz)", 5, 50, 10,
                            help="Lower cutoff frequency for bandpass wavelet")
freq_high = st.sidebar.slider("High Frequency (Hz)", 20, 100, 50,
                             help="Upper cutoff frequency for bandpass wavelet")
wavelet_length = st.sidebar.slider("Wavelet Length", 50, 200, 100,
                                  help="Number of samples in the wavelet")
dt = st.sidebar.slider("Time Sampling (s)", 0.001, 0.005, 0.002, 0.001,
                      help="Time sampling interval for synthetic seismic")

# Angle parameters
st.sidebar.header("Angle Settings")
angle_range_min = st.sidebar.slider("Minimum Angle (degrees)", 0, 30, 0,
                                   help="Minimum incidence angle for analysis")
angle_range_max = st.sidebar.slider("Maximum Angle (degrees)", 30, 60, 50,
                                   help="Maximum incidence angle for analysis")
angle_sampling = st.sidebar.slider("Angle Sampling (degrees)", 0.1, 2.0, 0.5, 0.1,
                                  help="Angle increment for calculation")
num_traces = st.sidebar.slider("Number of Traces", 20, 100, 50,
                              help="Number of traces in synthetic gather")
max_offset = st.sidebar.slider("Depth Offset (m)", 10, 50, 30,
                              help="Vertical extent around interface for display")

# Display parameters
st.sidebar.header("Display Settings")
colormap = st.sidebar.selectbox("Colormap", 
                               ["RdBu", "Viridis", "Plasma", "Inferno", "Magma", "Coolwarm", "Spectral", "Seismic"],
                               index=0)
show_confidence_intervals = st.sidebar.checkbox("Show Confidence Intervals", value=True,
                                               help="Display uncertainty ranges in AVO plots")

# Advanced parameters
with st.sidebar.expander("Advanced Parameters"):
    st.markdown("**Anisotropy Estimation Method**")
    anisotropy_method = st.selectbox("Method", ["default", "complex"], index=0,
                                   help="Method for estimating Thomsen parameters from logs")
    
    st.markdown("**Synthetic Data Generation**")
    synth_vp_min = st.slider("Synthetic VP Min (m/s)", 2000, 3000, 2200)
    synth_vp_max = st.slider("Synthetic VP Max (m/s)", 3000, 5000, 3800)
    synth_vp_vs_ratio = st.slider("VP/VS Ratio", 1.5, 2.2, 1.7, 0.1)
    synth_rho_min = st.slider("Density Min (g/cc)", 2.0, 2.4, 2.1, 0.1)
    synth_rho_max = st.slider("Density Max (g/cc)", 2.4, 3.0, 2.6, 0.1)

# Manual column selection (only show if file is uploaded)
if uploaded_file is not None:
    st.sidebar.header("Column Mapping")
    try:
        # Check if file is not empty
        if uploaded_file.size == 0:
            st.sidebar.error("Uploaded file is empty")
        else:
            # Try to read the CSV file with different encodings if needed
            try:
                uploaded_file.seek(0)
                df_preview = pd.read_csv(uploaded_file)
            except UnicodeDecodeError:
                # Try with different encoding if UTF-8 fails
                uploaded_file.seek(0)
                df_preview = pd.read_csv(uploaded_file, encoding='latin-1')
            except Exception as e:
                st.sidebar.error(f"Error reading file: {str(e)}")
                df_preview = pd.DataFrame()
            
            if df_preview.empty:
                st.sidebar.error("CSV file contains no data")
            else:
                available_columns = df_preview.columns.tolist()
                
                depth_col = st.sidebar.selectbox("Depth Column", available_columns, 
                                               index=available_columns.index('DEPTH') if 'DEPTH' in available_columns else 0)
                vp_col = st.sidebar.selectbox("VP Column", available_columns, 
                                            index=available_columns.index('VP') if 'VP' in available_columns else 0)
                vs_col = st.sidebar.selectbox("VS Column", available_columns, 
                                            index=available_columns.index('VS') if 'VS' in available_columns else 1)
                rho_col = st.sidebar.selectbox("Density Column", available_columns, 
                                             index=available_columns.index('RHOB') if 'RHOB' in available_columns else 2)
                
                # Optional columns
                gr_col = st.sidebar.selectbox("GR Column (optional)", [None] + available_columns, 
                                            index=0)
                phi_col = st.sidebar.selectbox("Porosity Column (optional)", [None] + available_columns, 
                                             index=0)
                sw_col = st.sidebar.selectbox("SW Column (optional)", [None] + available_columns, 
                                            index=0)
                rt_col = st.sidebar.selectbox("RT Column (optional)", [None] + available_columns, 
                                            index=0)
                vclay_col = st.sidebar.selectbox("VCLAY Column (optional)", [None] + available_columns,
                                               index=0)
                
    except Exception as e:
        st.sidebar.error(f"Error reading file: {str(e)}")

# Add sample data download option
sample_data = pd.DataFrame({
    'DEPTH': [1000, 1001, 1002, 1003, 1004, 1005, 1006, 1007, 1008, 1009],
    'VP': [2500, 2550, 2600, 2650, 2700, 2750, 2800, 2850, 2900, 2950],
    'VS': [1200, 1220, 1240, 1260, 1280, 1300, 1320, 1340, 1360, 1380],
    'RHOB': [2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9, 3.0],
    'GR': [45, 50, 55, 60, 65, 70, 75, 80, 85, 90]
})

csv_sample = sample_data.to_csv(index=False)
st.sidebar.download_button(
    "📋 Download Sample CSV",
    csv_sample,
    "sample_well_logs.csv",
    "text/csv",
    help="Download a sample CSV file to test the application"
)

# ==================== WAVELET GENERATION ====================
def generate_ricker_wavelet(frequency, length, dt):
    """Generate Ricker wavelet using the correct formula"""
    t = np.arange(-length//2, length//2) * dt
    t = t - np.mean(t)  # Center the wavelet
    
    # Ricker wavelet formula
    wavelet = (1 - 2 * (np.pi * frequency * t) ** 2) * np.exp(-(np.pi * frequency * t) ** 2)
    return wavelet / np.max(np.abs(wavelet))

def generate_bandpass_wavelet(freq_low, freq_high, length, dt):
    """Generate bandpass wavelet using scipy.signal"""
    # Create a signal with broadband frequency content
    impulse = np.zeros(length)
    impulse[length//2] = 1.0
    
    # Apply bandpass filter
    nyquist = 0.5 / dt
    low = freq_low / nyquist
    high = freq_high / nyquist
    b, a = signal.butter(4, [low, high], btype='band')
    wavelet = signal.filtfilt(b, a, impulse)
    return wavelet / np.max(np.abs(wavelet))

def generate_wavelet(wavelet_type='ricker', frequency=30, freq_low=10, freq_high=50, length=100, dt=0.002):
    """Generate wavelet based on type using scipy.signal"""
    if wavelet_type == 'ricker':
        return generate_ricker_wavelet(frequency, length, dt)
    elif wavelet_type == 'bandpass':
        return generate_bandpass_wavelet(freq_low, freq_high, length, dt)
    else:
        raise ValueError("Unknown wavelet type. Use 'ricker' or 'bandpass'")

# ==================== SYNTHETIC SEISMIC GENERATION ====================
def create_angle_gather_synthetic(reflection_coefficients, angles, wavelet, depth, num_traces=50, max_offset=30):
    """Create an angle gather synthetic seismic section"""
    # Create angle axis
    angle_axis = np.linspace(angles[0], angles[-1], num_traces)
    
    # Interpolate reflection coefficients
    interp_func = interp1d(angles, reflection_coefficients, kind='cubic', bounds_error=False, fill_value=0)
    rc_interp = interp_func(angle_axis)
    
    # Create synthetic seismic gather
    synthetic_gather = np.zeros((len(wavelet), num_traces))
    
    for i, rc_value in enumerate(rc_interp):
        # Create a spike at the reflection coefficient position
        spike = np.zeros(len(wavelet))
        spike[len(wavelet)//2] = rc_value
        
        # Convolve with wavelet
        trace = signal.convolve(spike, wavelet, mode='same', method='auto')
        synthetic_gather[:, i] = trace
    
    # Create depth/time axis (flipped vertically)
    depth_axis = np.linspace(depth - max_offset, depth + max_offset, len(wavelet))
    
    return angle_axis, depth_axis, synthetic_gather

# ==================== LOG PROCESSING ====================
def estimate_vclay_from_gr(gr, gr_min, gr_max, method='linear'):
    """Estimate clay volume from Gamma Ray log"""
    if method == 'linear':
        vclay = (gr - gr_min) / (gr_max - gr_min)
        vclay = np.clip(vclay, 0.0, 1.0)
    else:
        vclay = np.zeros_like(gr)
    return vclay

def preprocess_logs(df, vp_col='VP', vs_col='VS', rho_col='RHOB', 
                   gr_col=None, vclay_col=None, phi_col=None, sw_col=None, rt_col=None):
    """Preprocess logs and estimate missing parameters"""
    result_df = df.copy()
    
    # Handle optional columns
    gr_values = df[gr_col].values if gr_col and gr_col in df.columns else np.zeros(len(df))
    phi_values = df[phi_col].values if phi_col and phi_col in df.columns else np.zeros(len(df))
    
    # Estimate VCLAY from GR if not available
    if vclay_col and vclay_col in df.columns:
        vclay_used = vclay_col
    elif gr_col and gr_col in df.columns:
        gr = df[gr_col].values
        gr_min = np.nanpercentile(gr, 10)
        gr_max = np.nanpercentile(gr, 90)
        result_df['VCLAY_EST'] = estimate_vclay_from_gr(gr, gr_min, gr_max)
        vclay_used = 'VCLAY_EST'
    else:
        result_df['VCLAY_EST'] = 0.0
        vclay_used = 'VCLAY_EST'
    
    # Handle missing porosity
    if phi_col and phi_col in df.columns:
        phi_used = phi_col
    else:
        result_df['PHIT_EST'] = 0.15
        phi_used = 'PHIT_EST'
    
    return result_df, vclay_used, phi_used

# ==================== ELASTIC CONSTANTS AND THOMSEN PARAMETERS ====================
def estimate_thomsen_from_logs(vp, vs, vclay, porosity, method='default'):
    """Estimate Thomsen parameters from available logs"""
    epsilon = np.zeros_like(vp)
    gamma = np.zeros_like(vp)
    delta = np.zeros_like(vp)
    
    if method == 'default':
        epsilon = 0.1 * vclay + 0.05 * porosity
        gamma = 0.15 * vclay + 0.03 * porosity
        delta = 0.08 * vclay + 0.02 * porosity
        
        epsilon = np.clip(epsilon, 0.0, 0.3)
        gamma = np.clip(gamma, 0.0, 0.25)
        delta = np.clip(delta, -0.1, 0.2)
    elif method == 'complex':
        # More complex estimation based on empirical relationships
        epsilon = 0.12 * vclay + 0.06 * porosity + 0.002 * (vp - 2500)
        gamma = 0.18 * vclay + 0.04 * porosity + 0.001 * (vp - 2500)
        delta = 0.09 * vclay + 0.025 * porosity + 0.0015 * (vp - 2500)
        
        epsilon = np.clip(epsilon, 0.0, 0.35)
        gamma = np.clip(gamma, 0.0, 0.3)
        delta = np.clip(delta, -0.15, 0.25)
    
    return epsilon, gamma, delta

def calculate_elastic_constants(vp, vs, rho, epsilon, gamma, delta):
    """Calculate elastic constants for VTI media"""
    c33 = rho * vp**2
    c44 = rho * vs**2
    
    c11 = c33 * (1 + 2 * epsilon)
    c66 = c44 * (1 + 2 * gamma)
    
    # More stable calculation of c13
    delta_term = 2 * delta * c33 * (c33 - c44)
    # Ensure we don't take square root of negative values
    delta_term = np.maximum(delta_term, 0)
    c13 = np.sqrt(delta_term) + (c33 - 2 * c44)
    
    return {'c11': c11, 'c13': c13, 'c33': c33, 'c44': c44, 'c66': c66}

# ==================== VTI REFLECTION COEFFICIENT ====================
def vti_reflection_coefficient(theta, A_ratio, B_ratio, C_ratio, K):
    """Calculate VTI reflection coefficient"""
    term1 = 0.5 * A_ratio
    term2 = -0.5 * K * np.sin(theta)**2 * B_ratio
    term3 = 0.5 * np.tan(theta)**2 * C_ratio
    return term1 + term2 + term3

def aki_richards_reflection_coefficient(theta, vp1, vp2, vs1, vs2, rho1, rho2):
    """Calculate isotropic reflection coefficient"""
    vp_avg = (vp1 + vp2) / 2
    vs_avg = (vs1 + vs2) / 2
    rho_avg = (rho1 + rho2) / 2
    
    dvp = vp2 - vp1
    dvs = vs2 - vs1
    drho = rho2 - rho1
    
    term1 = 0.5 * (dvp/vp_avg + drho/rho_avg)
    term2 = (0.5 * dvp/vp_avg - 2 * (vs_avg**2/vp_avg**2) * (dvs/vs_avg + drho/rho_avg)) * np.sin(theta)**2
    term3 = 0.5 * dvp/vp_avg * (np.tan(theta)**2 - np.sin(theta)**2)
    
    return term1 + term2 + term3

def avo_classification(vp1, vs1, rho1, vp2, vs2, rho2):
    """Classify AVO response"""
    imp1 = vp1 * rho1
    imp2 = vp2 * rho2
    
    vp_vs1 = vp1 / vs1
    vp_vs2 = vp2 / vs2
    
    if imp2 > imp1:
        return "Class I" if vp_vs2 > vp_vs1 else "Class II"
    else:
        return "Class IV" if vp_vs2 > vp_vs1 else "Class III"

# ==================== MAIN PROCESSING FUNCTION ====================
def main_processing(df, vp_col='VP', vs_col='VS', rho_col='RHOB', vclay_col='VCLAY', phi_col='PHIT', method='default'):
    """Main function to process well logs"""
    result_df = df.copy()
    
    # Extract data
    vp = df[vp_col].values
    vs = df[vs_col].values
    rho = df[rho_col].values * 1000  # Convert to kg/m³
    
    # Get clay volume and porosity
    vclay = df[vclay_col].values if vclay_col in df.columns else np.zeros_like(vp)
    porosity = df[phi_col].values if phi_col in df.columns else np.zeros_like(vp)
    
    # Estimate Thomsen parameters
    epsilon, gamma, delta = estimate_thomsen_from_logs(vp, vs, vclay, porosity, method=method)
    
    result_df['EPSILON'] = epsilon
    result_df['GAMMA'] = gamma
    result_df['DELTA'] = delta
    
    # Calculate elastic constants
    constants = calculate_elastic_constants(vp, vs, rho, epsilon, gamma, delta)
    
    for key, value in constants.items():
        result_df[key] = value
    
    # Calculate A, B, C attributes
    A = rho * vp
    B = rho * vs**2 * np.exp(((vp/vs)**2 * (epsilon - delta))/4)
    C = vp * np.exp(epsilon)
    
    result_df['A'] = A
    result_df['B'] = B
    result_df['C'] = C
    
    # Calculate attribute ratios
    A_ratio = np.log(A[1:] / A[:-1])
    B_ratio = np.log(B[1:] / B[:-1])
    C_ratio = np.log(C[1:] / C[:-1])
    
    # Add ratios to result dataframe
    result_df['A_ratio'] = np.nan
    result_df['B_ratio'] = np.nan
    result_df['C_ratio'] = np.nan
    result_df.loc[result_df.index[:-1], 'A_ratio'] = A_ratio
    result_df.loc[result_df.index[:-1], 'B_ratio'] = B_ratio
    result_df.loc[result_df.index[:-1], 'C_ratio'] = C_ratio
    
    return result_df

# ==================== PLOTLY VISUALIZATION FUNCTIONS ====================
def plot_angle_gather(angle_axis, depth_axis, synthetic_gather, title, colormap='RdBu'):
    """Plot angle gather synthetic seismic section"""
    fig = go.Figure(data=go.Heatmap(
        z=synthetic_gather,
        x=angle_axis,
        y=depth_axis,
        colorscale=colormap,
        hoverongaps=False,
        colorbar=dict(title="Amplitude"),
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title='Incidence Angle (degrees)',
        yaxis_title='Depth (m)',
        yaxis=dict(autorange="reversed"),
        width=800,
        height=600
    )
    return fig

def plot_avo_response(angles_deg, rc_vti, rc_iso, avo_class, interface_depth, show_confidence=True):
    """Create interactive AVO response plot"""
    fig = go.Figure()
    
    # Calculate confidence intervals if requested
    if show_confidence:
        # Simple uncertainty estimation based on angle
        uncertainty = 0.05 + 0.001 * angles_deg**2
        upper_vti = rc_vti + uncertainty
        lower_vti = rc_vti - uncertainty
        
        fig.add_trace(go.Scatter(
            x=np.concatenate([angles_deg, angles_deg[::-1]]),
            y=np.concatenate([upper_vti, lower_vti[::-1]]),
            fill='toself',
            fillcolor='rgba(0, 100, 255, 0.2)',
            line=dict(color='rgba(255, 255, 255, 0)'),
            name='VTI Uncertainty',
            showlegend=True
        ))
    
    fig.add_trace(go.Scatter(
        x=angles_deg, y=rc_vti,
        mode='lines+markers',
        name='VTI RC (3-parameter)',
        line=dict(color='blue', width=2)
    ))
    
    fig.add_trace(go.Scatter(
        x=angles_deg, y=rc_iso,
        mode='lines+markers',
        name='Isotropic RC (Aki-Richards)',
        line=dict(color='red', width=2, dash='dash')
    ))
    
    # Add annotations for AVO class
    fig.add_annotation(
        x=0.05, y=0.95,
        xref="paper", yref="paper",
        text=f"AVO Class: {avo_class}",
        showarrow=False,
        font=dict(size=14, color="black"),
        bgcolor="white",
        bordercolor="black",
        borderwidth=1,
        borderpad=4
    )
    
    fig.update_layout(
        title=f'AVO Response at Depth: {interface_depth:.1f} m',
        xaxis_title='Incidence Angle (degrees)',
        yaxis_title='Reflection Coefficient',
        width=800,
        height=500,
        hovermode='x unified'
    )
    return fig

def plot_well_logs(df, depth_col, vp_col, vs_col, rho_col, gr_col=None, selected_depth=None):
    """Create interactive well log visualization with highlighted depth"""
    # Determine number of tracks based on available data
    tracks = 3
    if gr_col and gr_col in df.columns:
        tracks = 4
    
    # Create subplots
    fig = make_subplots(
        rows=1, 
        cols=tracks, 
        subplot_titles=("VP (m/s)", "VS (m/s)", "Density (g/cc)", "Gamma Ray") if tracks == 4 else ("VP (m/s)", "VS (m/s)", "Density (g/cc)"),
        shared_yaxes=True,
        horizontal_spacing=0.05
    )
    
    # Add VP log
    fig.add_trace(
        go.Scatter(
            x=df[vp_col], 
            y=df[depth_col], 
            mode='lines',
            name='VP',
            line=dict(color='blue', width=1)
        ),
        row=1, col=1
    )
    
    # Add VS log
    fig.add_trace(
        go.Scatter(
            x=df[vs_col], 
            y=df[depth_col], 
            mode='lines',
            name='VS',
            line=dict(color='green', width=1)
        ),
        row=1, col=2
    )
    
    # Add Density log
    fig.add_trace(
        go.Scatter(
            x=df[rho_col], 
            y=df[depth_col], 
            mode='lines',
            name='Density',
            line=dict(color='red', width=1)
        ),
        row=1, col=3
    )
    
    # Add Gamma Ray log if available
    if tracks == 4:
        fig.add_trace(
            go.Scatter(
                x=df[gr_col], 
                y=df[depth_col], 
                mode='lines',
                name='GR',
                line=dict(color='purple', width=1)
            ),
            row=1, col=4
        )
    
    # Add highlighted depth line if selected
    if selected_depth is not None:
        # Find the closest depth in the data
        depth_idx = (np.abs(df[depth_col] - selected_depth)).argmin()
        actual_depth = df[depth_col].iloc[depth_idx]
        
        # Add horizontal line across all subplots
        for col in range(1, tracks+1):
            fig.add_hline(
                y=actual_depth, 
                line=dict(color="orange", width=3, dash="dash"),
                row=1, 
                col=col
            )
    
    # Update layout
    fig.update_layout(
        title="Well Log Visualization",
        height=600,
        showlegend=False,
        yaxis=dict(autorange="reversed", title="Depth (m)"),
    )
    
    # Update x-axis titles
    fig.update_xaxes(title_text="VP (m/s)", row=1, col=1)
    fig.update_xaxes(title_text="VS (m/s)", row=1, col=2)
    fig.update_xaxes(title_text="Density (g/cc)", row=1, col=3)
    if tracks == 4:
        fig.update_xaxes(title_text="Gamma Ray (API)", row=1, col=4)
    
    return fig

def plot_thomsen_parameters(depth, epsilon, gamma, delta):
    """Plot Thomsen parameters with depth"""
    fig = make_subplots(rows=1, cols=3, 
                       subplot_titles=('Epsilon (ε)', 'Gamma (γ)', 'Delta (δ)'))
    
    fig.add_trace(go.Scatter(x=epsilon, y=depth, mode='lines', name='ε', 
                            line=dict(color='blue')), row=1, col=1)
    fig.add_trace(go.Scatter(x=gamma, y=depth, mode='lines', name='γ', 
                            line=dict(color='green')), row=1, col=2)
    fig.add_trace(go.Scatter(x=delta, y=depth, mode='lines', name='δ', 
                            line=dict(color='red')), row=1, col=3)
    
    fig.update_yaxes(title_text="Depth (m)", autorange="reversed", row=1, col=1)
    fig.update_xaxes(title_text="ε", row=1, col=1)
    fig.update_xaxes(title_text="γ", row=1, col=2)
    fig.update_xaxes(title_text="δ", row=1, col=3)
    
    fig.update_layout(
        title="Thomsen Anisotropy Parameters",
        height=500,
        showlegend=False
    )
    
    return fig

# ==================== GUIDE AND THEORY CONTENT ====================
def show_guide_and_theory():
    """Display user guide and theoretical background"""
    st.header("📖 User Guide & Theoretical Background")
    
    tab1, tab2, tab3, tab4 = st.tabs(["User Guide", "Theory", "References", "FAQ & Troubleshooting"])
    
    with tab1:
        st.subheader("User Guide")
        st.markdown("""
        ### How to Use This App
        
        1. **Upload Data**: Use the sidebar to upload a CSV file with well log data
        2. **Configure Parameters**: Adjust wavelet settings, angle range, and display options
        3. **Map Columns**: If uploading data, map the correct columns for VP, VS, RHOB, etc.
        4. **Select Interface**: Use the slider to choose the depth interface to analyze
        5. **Analyze Results**: View AVO responses and synthetic seismic gathers
        6. **Download Results**: Export analysis results and summary reports
        
        ### Required Data Format
        - CSV file with columns: DEPTH, VP, VS, RHOB
        - Optional columns: GR, PHIT, SW, RT, VCLAY
        
        ### Default Values
        - If no file is uploaded, synthetic data will be generated
        - Missing optional parameters will be estimated from available data
        """)
    
    with tab2:
        st.subheader("Theoretical Background")
        st.markdown("""
        ### VTI Anisotropy Theory
        
        **Transverse Isotropy with Vertical Axis (VTI)** media are characterized by:
        - Rotational symmetry around the vertical axis
        - Different velocities in horizontal vs vertical directions
        - Five independent elastic constants: c₁₁, c₁₃, c₃₃, c₄₄, c₆₆
        
        ### Thomsen Parameters
        Thomsen (1986) introduced three dimensionless parameters to describe weak anisotropy:
        
        - **ε (Epsilon)**: P-wave anisotropy parameter
          $$ε = \\frac{c_{11} - c_{33}}{2c_{33}}$$
        
        - **γ (Gamma)**: S-wave anisotropy parameter  
          $$γ = \\frac{c_{66} - c_{44}}{2c_{44}}$$
        
        - **δ (Delta)**: Near-vertical anisotropy parameter
          $$δ = \\frac{(c_{13} + c_{44})^2 - (c_{33} - c_{44})^2}{2c_{33}(c_{33} - c_{44})}$$
        
        ### Reflection Coefficient Formulation
        The VTI reflection coefficient is given by:
        
        $$R_{VTI}(θ) = \\frac{1}{2} \\frac{Δ(ρV_{P0})}{ρV_{P0}} - \\frac{1}{2} K \\sin^2θ \\left[\\frac{Δ(ρV_{S0}^2 e^{σ/4})}{ρV_{S0}^2 e^{σ/4}}\\right] + \\frac{1}{2} \\tan^2θ \\frac{Δ(V_{P0} e^ε)}{V_{P0} e^ε}$$
        
        Where $K = (2V_{S0}/V_{P0})^2$ and $σ = (V_{P0}/V_{S0})^2(ε - δ)$
        
        ### AVO Classification
        - **Class I**: High impedance contrast, positive intercept
        - **Class II**: Near-zero impedance contrast  
        - **Class III**: Low impedance contrast, negative intercept
        - **Class IV**: Very low impedance contrast, negative gradient
        """)
    
    with tab3:
        st.subheader("References")
        st.markdown("""
        ### Key References
        
        1. **Thomsen, L. (1986)**
           *"Weak elastic anisotropy"*
           Geophysics, 51(10), 1954-1966
        
        2. **Rüger, A. (1997)**
           *"P-wave reflection coefficients for transversely isotropic models with vertical and horizontal axis of symmetry"*
           Geophysics, 62(3), 713-722
        
        3. **Aki, K., and Richards, P.G. (1980)**
           *"Quantitative Seismology: Theory and Methods"*
           W.H. Freeman and Company
        
        4. **Zhang, F., Zhang, T., and Li, X.Y. (2013)**
           *"A new approximation for PP-wave reflection coefficient in VTI media"*
           Geophysical Prospecting, 61(2), 237-248
        
        5. **Tsvankin, I. (2012)**
           *"Seismic Signatures and Analysis of Reflection Data in Anisotropic Media"*
           Society of Exploration Geophysicists
        
        ### Software Implementation
        - This app uses Python with Streamlit for the web interface
        - Scientific computing with NumPy, SciPy, and Pandas
        - Visualization with Plotly for interactive graphs
        - Wavelet generation using Ricker and bandpass filters
        """)
        
    with tab4:
        st.subheader("FAQ & Troubleshooting")
        st.markdown("""
        ### Frequently Asked Questions
        
        **Q: What file format should I use for my data?**
        A: The app accepts CSV files with headers. Make sure your data has columns for DEPTH, VP, VS, and RHOB.
        
        **Q: Why are my Thomsen parameters showing unexpected values?**
        A: Thomsen parameters are estimated from available logs. If you have VCLAY or GR data, the estimation will be more accurate.
        
        **Q: How do I interpret the AVO classification?**
        A: The AVO class provides information about the impedance contrast and fluid content at the interface.
        
        **Q: What's the difference between the isotropic and VTI reflection coefficients?**
        A: The isotropic model assumes no anisotropy, while the VTI model accounts for directional velocity variations.
        
        ### Troubleshooting
        
        **Problem: File upload fails**
        - Solution: Check that your file is a valid CSV with proper headers
        
        **Problem: No data displayed after upload**
        - Solution: Verify that you've correctly mapped the column names in the sidebar
        
        **Problem: Synthetic seismic looks noisy or unrealistic**
        - Solution: Adjust the wavelet parameters (frequency, length) and check your input data quality
        """)

# ==================== STREAMLIT APP MAIN ====================
def main():
    # Create tabs for different sections
    tab1, tab2, tab3 = st.tabs(["Analysis", "Data Visualization", "Guide & Theory"])
    
    with tab1:
        st.title("🎯 VTI Anisotropy Analysis with Synthetic Seismic")
        
        # Check if file is uploaded
        if uploaded_file is None:
            st.info("""
            **Welcome to VTI Anisotropy Analysis!**
            
            Please upload a CSV file using the sidebar to begin analysis.
            You can also download the sample CSV file from the sidebar to test the application.
            
            Required columns: DEPTH, VP, VS, RHOB
            Optional columns: GR, PHIT, SW, RT, VCLAY
            """)
            return
            
        # Load data from uploaded file
        try:
            # Reset file pointer to beginning
            uploaded_file.seek(0)
            
            # Try different encodings and delimiters
            try:
                # First try with standard parameters
                df = pd.read_csv(uploaded_file)
            except Exception as e:
                st.warning(f"First read attempt failed: {str(e)}. Trying alternative methods...")
                
                # Reset file pointer
                uploaded_file.seek(0)
                
                # Try different encodings
                encodings = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252']
                for encoding in encodings:
                    try:
                        uploaded_file.seek(0)
                        df = pd.read_csv(uploaded_file, encoding=encoding)
                        st.success(f"Successfully read with {encoding} encoding")
                        break
                    except:
                        continue
                else:
                    # If all encodings failed, try with engine parameter
                    try:
                        uploaded_file.seek(0)
                        df = pd.read_csv(uploaded_file, engine='python')
                        st.success("Successfully read with python engine")
                    except:
                        # Last resort: try with error handling
                        uploaded_file.seek(0)
                        df = pd.read_csv(uploaded_file, on_bad_lines='skip', encoding='latin-1')
                        st.warning("Read with error skipping - some rows may be missing")
            
            # Check if DataFrame is empty
            if df.empty:
                st.error("Uploaded CSV file is empty or could not be parsed. Please upload a valid CSV file.")
                st.info("""
                **Tips for valid CSV files:**
                - Ensure your file has column headers in the first row
                - Use comma separation (not semicolon or tab)
                - Check that the file is not corrupted
                - Try opening the file in a text editor to verify the format
                """)
                return
                
            st.success(f"CSV file loaded successfully! Shape: {df.shape}")
            st.write(f"Columns found: {list(df.columns)}")
            
        except Exception as e:
            st.error(f"Error reading CSV file: {str(e)}")
            st.info("""
            **Common solutions:**
            1. Check that your file is a valid CSV format
            2. Ensure the file has column headers
            3. Try saving your file with a different encoding (UTF-8 recommended)
            4. Open the file in a text editor to verify the format
            5. If using Excel, use 'Save As' and choose CSV format
            """)
            
            # Provide more detailed error information
            try:
                uploaded_file.seek(0)
                sample_content = uploaded_file.read(200).decode('utf-8', errors='ignore')
                st.text_area("First 200 characters of file:", sample_content, height=100)
            except:
                st.write("Could not preview file content")
            
            return
        
        # Preprocess logs
        processed_df, vclay_col_used, phi_col_used = preprocess_logs(
            df, 
            vp_col=vp_col, 
            vs_col=vs_col, 
            rho_col=rho_col,
            gr_col=gr_col,
            vclay_col=vclay_col,
            phi_col=phi_col
        )
        
        # Process the data
        result_df = main_processing(
            processed_df, 
            vp_col=vp_col, 
            vs_col=vs_col, 
            rho_col=rho_col, 
            vclay_col=vclay_col_used, 
            phi_col=phi_col_used,
            method=anisotropy_method
        )
        
        # Generate wavelet
        wavelet = generate_wavelet(
            wavelet_type=wavelet_type,
            frequency=wavelet_frequency,
            freq_low=freq_low,
            freq_high=freq_high,
            length=wavelet_length,
            dt=dt
        )
        
        # Display wavelet info
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Wavelet Information")
            st.write(f"**Type:** {wavelet_type}")
            st.write(f"**Frequency:** {wavelet_frequency} Hz")
            st.write(f"**Length:** {len(wavelet)} samples")
            st.write(f"**Max Amplitude:** {np.max(np.abs(wavelet)):.3f}")
            st.write(f"**Sample Rate:** {dt*1000:.1f} ms")
        
        with col2:
            fig_wavelet = go.Figure()
            fig_wavelet.add_trace(go.Scatter(
                y=wavelet,
                mode='lines',
                name='Wavelet',
                line=dict(color='blue', width=2)
            ))
            fig_wavelet.update_layout(
                title="Generated Wavelet",
                xaxis_title="Samples",
                yaxis_title="Amplitude",
                height=300
            )
            st.plotly_chart(fig_wavelet, use_container_width=True)
        
        # Well log visualization
        st.subheader("Well Log Visualization")
        well_log_fig = plot_well_logs(
            result_df, 
            depth_col, 
            vp_col, 
            vs_col, 
            rho_col,
            gr_col=gr_col
        )
        st.plotly_chart(well_log_fig, use_container_width=True)
        
        # Interface selection
        st.subheader("Interface Analysis")
        
        # Ensure we have valid depth data
        if depth_col not in result_df.columns:
            st.error(f"Depth column '{depth_col}' not found in data")
            return
        
        interface_depth = st.slider(
            "Select Interface Depth", 
            min_value=float(result_df[depth_col].min()),
            max_value=float(result_df[depth_col].max()),
            value=float(result_df[depth_col].iloc[min(100, len(result_df)-2)]),
            step=0.5,
            help="Select the depth where you want to analyze the interface between two layers"
        )
        
        # Update well log visualization with selected depth
        well_log_fig_selected = plot_well_logs(
            result_df, 
            depth_col, 
            vp_col, 
            vs_col, 
            rho_col,
            gr_col=gr_col,
            selected_depth=interface_depth
        )
        st.plotly_chart(well_log_fig_selected, use_container_width=True)
        
        # Find closest depth index
        interface_idx = (np.abs(result_df[depth_col] - interface_depth)).argmin()
        
        if interface_idx < len(result_df) - 1:
            # Get properties for the interface
            vp1 = result_df[vp_col].iloc[interface_idx]
            vs1 = result_df[vs_col].iloc[interface_idx]
            rho1 = result_df[rho_col].iloc[interface_idx] * 1000
            
            vp2 = result_df[vp_col].iloc[interface_idx + 1]
            vs2 = result_df[vs_col].iloc[interface_idx + 1]
            rho2 = result_df[rho_col].iloc[interface_idx + 1] * 1000
            
            # Calculate K value
            vp_avg = (vp1 + vp2) / 2
            vs_avg = (vs1 + vs2) / 2
            K = (2 * vs_avg / vp_avg)**2
            
            # Get attribute ratios
            A_ratio = result_df['A_ratio'].iloc[interface_idx]
            B_ratio = result_df['B_ratio'].iloc[interface_idx]
            C_ratio = result_df['C_ratio'].iloc[interface_idx]
            
            # Generate reflection coefficients
            angles_deg = np.arange(angle_range_min, angle_range_max + angle_sampling, angle_sampling)
            angles_rad = np.radians(angles_deg)
            
            rc_vti = vti_reflection_coefficient(angles_rad, A_ratio, B_ratio, C_ratio, K)
            rc_iso = aki_richards_reflection_coefficient(angles_rad, vp1, vp2, vs1, vs2, rho1, rho2)
            
            # AVO classification
            avo_class = avo_classification(vp1, vs1, rho1/1000, vp2, vs2, rho2/1000)
            
            # Display results
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("AVO Class", avo_class)
            with col2:
                st.metric("Interface Depth", f"{interface_depth:.1f} m")
            with col3:
                st.metric("K Value", f"{K:.4f}")
            with col4:
                st.metric("Impedance Contrast", f"{(vp2*rho2/1000)/(vp1*rho1/1000):.3f}")
            
            # Create plots
            st.subheader("AVO Response Comparison")
            avo_fig = plot_avo_response(angles_deg, rc_vti, rc_iso, avo_class, interface_depth, show_confidence_intervals)
            st.plotly_chart(avo_fig, use_container_width=True)
            
            # Create synthetic seismic gathers
            st.subheader("Synthetic Seismic Angle Gathers")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # VTI angle gather
                angle_axis_vti, depth_axis_vti, synthetic_gather_vti = create_angle_gather_synthetic(
                    rc_vti, angles_deg, wavelet, interface_depth, num_traces, max_offset
                )
                gather_fig_vti = plot_angle_gather(
                    angle_axis_vti, depth_axis_vti, synthetic_gather_vti,
                    f'VTI Model ({wavelet_type} wavelet)',
                    colormap
                )
                st.plotly_chart(gather_fig_vti, use_container_width=True)
            
            with col2:
                # Isotropic angle gather
                angle_axis_iso, depth_axis_iso, synthetic_gather_iso = create_angle_gather_synthetic(
                    rc_iso, angles_deg, wavelet, interface_depth, num_traces, max_offset
                )
                gather_fig_iso = plot_angle_gather(
                    angle_axis_iso, depth_axis_iso, synthetic_gather_iso,
                    f'Isotropic Model ({wavelet_type} wavelet)',
                    colormap
                )
                st.plotly_chart(gather_fig_iso, use_container_width=True)
            
            # Display attribute ratios
            st.subheader("Attribute Ratios")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("A Ratio", f"{A_ratio:.4f}", 
                         help="Relative change in acoustic impedance (ρVp)")
            with col2:
                st.metric("B Ratio", f"{B_ratio:.4f}", 
                         help="Relative change in shear modulus term (ρVs²exp(σ/4))")
            with col3:
                st.metric("C Ratio", f"{C_ratio:.4f}", 
                         help="Relative change in anisotropic P-wave term (Vpe^ε)")
        
        # Results download section
        st.subheader("Results Download")
        
        # Convert DataFrame to CSV
        csv = result_df.to_csv(index=False)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.download_button(
                label="📥 Download Results as CSV",
                data=csv,
                file_name="vti_analysis_results.csv",
                mime="text/csv",
                help="Download the complete analysis results as a CSV file"
            )
        
        with col2:
            # Create summary report
            summary_report = f"""
            VTI Anisotropy Analysis Report
            ==============================
            
            Analysis Parameters:
            - Wavelet Type: {wavelet_type}
            - Wavelet Frequency: {wavelet_frequency} Hz
            - Angle Range: {angle_range_min}-{angle_range_max} degrees
            - Colormap: {colormap}
            - Anisotropy Estimation Method: {anisotropy_method}
            
            Interface Analysis:
            - Depth: {interface_depth:.1f} m
            - AVO Class: {avo_class}
            - K Value: {K:.4f}
            - Attribute Ratios: A={A_ratio:.4f}, B={B_ratio:.4f}, C={C_ratio:.4f}
            - Impedance Contrast: {(vp2*rho2/1000)/(vp1*rho1/1000):.3f}
            
            Data Source: Uploaded CSV
            """
            
            st.download_button(
                label="📄 Download Summary Report",
                data=summary_report,
                file_name="vti_analysis_summary.txt",
                mime="text/plain",
                help="Download a summary report of the analysis"
            )
    
    with tab2:
        st.header("Data Visualization")
        
        if 'result_df' in locals():
            # Display Thomsen parameters
            st.subheader("Thomsen Anisotropy Parameters")
            thomsen_fig = plot_thomsen_parameters(
                result_df[depth_col].values,
                result_df['EPSILON'].values,
                result_df['GAMMA'].values,
                result_df['DELTA'].values
            )
            st.plotly_chart(thomsen_fig, use_container_width=True)
            
            # Display log curves
            st.subheader("Well Log Curves")
            
            log_fig = make_subplots(rows=1, cols=4, 
                                   subplot_titles=('VP and VS', 'Density', 'Anisotropy Parameters', 'Other Logs'),
                                   shared_yaxes=True)
            
            # VP and VS
            log_fig.add_trace(go.Scatter(x=result_df[vp_col], y=result_df[depth_col], 
                                        name='VP', line=dict(color='red')), row=1, col=1)
            log_fig.add_trace(go.Scatter(x=result_df[vs_col], y=result_df[depth_col], 
                                        name='VS', line=dict(color='blue')), row=1, col=1)
            
            # Density
            log_fig.add_trace(go.Scatter(x=result_df[rho_col], y=result_df[depth_col], 
                                        name='Density', line=dict(color='green')), row=1, col=2)
            
            # Anisotropy parameters
            log_fig.add_trace(go.Scatter(x=result_df['EPSILON'], y=result_df[depth_col], 
                                        name='Epsilon', line=dict(color='orange')), row=1, col=3)
            log_fig.add_trace(go.Scatter(x=result_df['GAMMA'], y=result_df[depth_col], 
                                        name='Gamma', line=dict(color='purple')), row=1, col=3)
            log_fig.add_trace(go.Scatter(x=result_df['DELTA'], y=result_df[depth_col], 
                                        name='Delta', line=dict(color='brown')), row=1, col=3)
            
            # Other logs if available
            if gr_col and gr_col in result_df.columns:
                log_fig.add_trace(go.Scatter(x=result_df[gr_col], y=result_df[depth_col], 
                                            name='GR', line=dict(color='black')), row=1, col=4)
            
            # Update axes
            log_fig.update_yaxes(title_text="Depth (m)", autorange="reversed", row=1, col=1)
            log_fig.update_xaxes(title_text="Velocity (m/s)", row=1, col=1)
            log_fig.update_xaxes(title_text="Density (g/cc)", row=1, col=2)
            log_fig.update_xaxes(title_text="Anisotropy Parameters", row=1, col=3)
            log_fig.update_xaxes(title_text="Other Logs", row=1, col=4)
            
            log_fig.update_layout(
                height=600,
                showlegend=True
            )
            
            st.plotly_chart(log_fig, use_container_width=True)
            
            # Show data preview
            st.subheader("Data Preview")
            st.dataframe(result_df.head(), use_container_width=True)
            
            # Show statistics
            st.subheader("Statistics")
            numeric_cols = result_df.select_dtypes(include=[np.number]).columns.tolist()
            st.dataframe(result_df[numeric_cols].describe(), use_container_width=True)
        else:
            st.info("Process data in the Analysis tab to visualize results here.")
    
    with tab3:
        show_guide_and_theory()

if __name__ == "__main__":
    main()

