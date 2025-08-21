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
uploaded_file = st.sidebar.file_uploader("Upload CSV file", type=["csv"])

# Wavelet parameters
st.sidebar.header("Wavelet Settings")
wavelet_type = st.sidebar.selectbox("Wavelet Type", ["ricker", "bandpass"], index=0)
wavelet_frequency = st.sidebar.slider("Wavelet Frequency (Hz)", 10, 100, 30)
freq_low = st.sidebar.slider("Low Frequency (Hz)", 5, 50, 10)
freq_high = st.sidebar.slider("High Frequency (Hz)", 20, 100, 50)
wavelet_length = st.sidebar.slider("Wavelet Length", 50, 200, 100)
dt = st.sidebar.slider("Time Sampling (s)", 0.001, 0.005, 0.002, 0.001)

# Angle parameters
st.sidebar.header("Angle Settings")
angle_range_min = st.sidebar.slider("Minimum Angle (degrees)", 0, 30, 0)
angle_range_max = st.sidebar.slider("Maximum Angle (degrees)", 30, 60, 50)
angle_sampling = st.sidebar.slider("Angle Sampling (degrees)", 0.1, 2.0, 0.5, 0.1)
num_traces = st.sidebar.slider("Number of Traces", 20, 100, 50)
max_offset = st.sidebar.slider("Depth Offset (m)", 10, 50, 30)

# Display parameters
st.sidebar.header("Display Settings")
colormap = st.sidebar.selectbox("Colormap", 
                               ["RdBu", "Viridis", "Plasma", "Inferno", "Magma", "Coolwarm", "Spectral"],
                               index=0)

# Manual column selection
st.sidebar.header("Column Mapping")
if uploaded_file is not None:
    try:
        df_preview = pd.read_csv(uploaded_file)
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
        
    except Exception as e:
        st.sidebar.error(f"Error reading file: {e}")

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
    
    return epsilon, gamma, delta

def calculate_elastic_constants(vp, vs, rho, epsilon, gamma, delta):
    """Calculate elastic constants for VTI media"""
    c33 = rho * vp**2
    c44 = rho * vs**2
    
    c11 = c33 * (1 + 2 * epsilon)
    c66 = c44 * (1 + 2 * gamma)
    c13 = np.sqrt(2 * delta * c33 * (c33 - c44)) + (c33 - 2 * c44)
    
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
def main_processing(df, vp_col='VP', vs_col='VS', rho_col='RHOB', vclay_col='VCLAY', phi_col='PHIT'):
    """Main function to process well logs"""
    result_df = df.copy()
    
    # Extract data
    vp = df[vp_col].values
    vs = df[vs_col].values
    rho = df[rho_col].values * 1000
    
    # Get clay volume and porosity
    vclay = df[vclay_col].values if vclay_col in df.columns else np.zeros_like(vp)
    porosity = df[phi_col].values if phi_col in df.columns else np.zeros_like(vp)
    
    # Estimate Thomsen parameters
    epsilon, gamma, delta = estimate_thomsen_from_logs(vp, vs, vclay, porosity)
    
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

def plot_avo_response(angles_deg, rc_vti, rc_iso, avo_class, interface_depth):
    """Create interactive AVO response plot"""
    fig = go.Figure()
    
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
    
    fig.update_layout(
        title=f'AVO Response: {avo_class} at Depth: {interface_depth:.1f} m',
        xaxis_title='Incidence Angle (degrees)',
        yaxis_title='Reflection Coefficient',
        width=800,
        height=500,
    )
    return fig

# ==================== STREAMLIT APP MAIN ====================
def main():
    st.title("🎯 VTI Anisotropy Analysis with Synthetic Seismic")
    
    # Generate or load data
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.success("CSV file loaded successfully!")
        except Exception as e:
            st.error(f"Error reading CSV file: {e}")
            return
    else:
        # Generate synthetic data
        st.info("Using synthetic data for demonstration")
        depth = np.arange(1000, 3000, 0.5)
        vp = 2000 + 1.5 * (depth - 1000) / 10 + np.random.normal(0, 50, len(depth))
        vs = vp / 1.7 + np.random.normal(0, 30, len(depth))
        rho = 2.1 + 0.0004 * (depth - 1000) + np.random.normal(0, 0.05, len(depth))
        
        df = pd.DataFrame({
            'DEPTH': depth,
            'VP': vp,
            'VS': vs,
            'RHOB': rho,
        })
    
    # Preprocess logs
    processed_df, vclay_col_used, phi_col_used = preprocess_logs(
        df, 
        vp_col=vp_col, 
        vs_col=vs_col, 
        rho_col=rho_col,
        gr_col=gr_col,
        phi_col=phi_col
    )
    
    # Process the data
    result_df = main_processing(
        processed_df, 
        vp_col=vp_col, 
        vs_col=vs_col, 
        rho_col=rho_col, 
        vclay_col=vclay_col_used, 
        phi_col=phi_col_used
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
    
    # Interface selection
    st.subheader("Interface Analysis")
    interface_depth = st.slider(
        "Select Interface Depth", 
        min_value=float(result_df['DEPTH'].min()),
        max_value=float(result_df['DEPTH'].max()),
        value=float(result_df['DEPTH'].iloc[1000]),
        step=0.5
    )
    
    # Find closest depth index
    interface_idx = (np.abs(result_df['DEPTH'] - interface_depth)).argmin()
    
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
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("AVO Class", avo_class)
        with col2:
            st.metric("Interface Depth", f"{interface_depth:.1f} m")
        with col3:
            st.metric("K Value", f"{K:.4f}")
        
        # Create plots
        col1, col2 = st.columns(2)
        
        with col1:
            avo_fig = plot_avo_response(angles_deg, rc_vti, rc_iso, avo_class, interface_depth)
            st.plotly_chart(avo_fig, use_container_width=True)
        
        with col2:
            # Create angle gather synthetic seismic
            angle_axis_vti, depth_axis_vti, synthetic_gather_vti = create_angle_gather_synthetic(
                rc_vti, angles_deg, wavelet, interface_depth, num_traces, max_offset
            )
            gather_fig_vti = plot_angle_gather(
                angle_axis_vti, depth_axis_vti, synthetic_gather_vti,
                f'VTI Angle Gather ({wavelet_type} wavelet)',
                colormap
            )
            st.plotly_chart(gather_fig_vti, use_container_width=True)
        
        # Display attribute ratios
        st.subheader("Attribute Ratios")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("A Ratio", f"{A_ratio:.4f}")
        with col2:
            st.metric("B Ratio", f"{B_ratio:.4f}")
        with col3:
            st.metric("C Ratio", f"{C_ratio:.4f}")
    
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
        
        Interface Analysis:
        - Depth: {interface_depth:.1f} m
        - AVO Class: {avo_class}
        - K Value: {K:.4f}
        - Attribute Ratios: A={A_ratio:.4f}, B={B_ratio:.4f}, C={C_ratio:.4f}
        """
        
        st.download_button(
            label="📄 Download Summary Report",
            data=summary_report,
            file_name="vti_analysis_summary.txt",
            mime="text/plain",
            help="Download a summary report of the analysis"
        )
    
    # Show data preview
    st.subheader("Data Preview")
    st.dataframe(result_df.head(), use_container_width=True)
    
    # Show statistics
    st.subheader("Statistics")
    numeric_cols = result_df.select_dtypes(include=[np.number]).columns.tolist()
    st.dataframe(result_df[numeric_cols].describe(), use_container_width=True)

if __name__ == "__main__":
    main()
