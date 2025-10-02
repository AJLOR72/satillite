import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from datetime import datetime
from PIL import Image
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io

# Set page configuration
st.set_page_config(
    page_title="Satellite Image Analysis Dashboard",
    page_icon="🛰️",
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding-left: 20px;
        padding-right: 20px;
    }
    </style>
""", unsafe_allow_html=True)

def load_image(uploaded_file):
    """Load and convert uploaded image to RGB numpy array."""
    img = Image.open(uploaded_file)
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img_rgb = np.array(img)
    return img_rgb

def extract_average_rgb(img_rgb):
    """Extract average RGB values across the entire image."""
    avg_r = np.mean(img_rgb[:, :, 0])
    avg_g = np.mean(img_rgb[:, :, 1])
    avg_b = np.mean(img_rgb[:, :, 2])
    return avg_r, avg_g, avg_b

def calculate_pseudo_ndvi(img_rgb):
    """Calculate pseudo-NDVI using Red and Green channels."""
    red = img_rgb[:, :, 0].astype(float)
    green = img_rgb[:, :, 1].astype(float)
    
    denominator = green + red
    denominator[denominator == 0] = 1
    
    pseudo_ndvi = (green - red) / denominator
    avg_pseudo_ndvi = np.mean(pseudo_ndvi)
    
    return pseudo_ndvi, avg_pseudo_ndvi

def perform_kmeans_segmentation(img_rgb, n_clusters=4):
    """Apply K-means clustering for land cover classification."""
    pixels = img_rgb.reshape(-1, 3)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(pixels)
    centers = kmeans.cluster_centers_.astype(int)
    segmented_img = centers[labels].reshape(img_rgb.shape)
    
    unique, counts = np.unique(labels, return_counts=True)
    total_pixels = len(labels)
    distribution = {i: (counts[i] / total_pixels) * 100 for i in unique}
    
    return segmented_img, centers, distribution, labels

def create_powerbi_data(img_rgb, avg_rgb, pseudo_ndvi_value, centers, distribution, 
                        spatial_res, lat, lon):
    """Generate data structure for visualizations."""
    acq_date = datetime.now().strftime('%Y-%m-%d')
    
    # Main metadata table
    metadata_df = pd.DataFrame({
        'Metric': ['Acquisition Date', 'Spatial Resolution (m)', 'Latitude', 'Longitude', 
                   'Image Width (px)', 'Image Height (px)', 'Radiometric Resolution',
                   'Average Red', 'Average Green', 'Average Blue', 'Pseudo-NDVI'],
        'Value': [acq_date, spatial_res, lat, lon, 
                  img_rgb.shape[1], img_rgb.shape[0], '8-bit',
                  f"{avg_rgb[0]:.2f}", f"{avg_rgb[1]:.2f}", f"{avg_rgb[2]:.2f}", 
                  f"{pseudo_ndvi_value:.4f}"]
    })
    
    # Cluster analysis table
    cluster_df = pd.DataFrame({
        'Cluster_ID': [f"Cluster {i+1}" for i in range(len(centers))],
        'Red_Value': centers[:, 0],
        'Green_Value': centers[:, 1],
        'Blue_Value': centers[:, 2],
        'Percentage': [distribution[i] for i in range(len(centers))],
        'Pixel_Count': [int((distribution[i]/100) * img_rgb.shape[0] * img_rgb.shape[1]) 
                        for i in range(len(centers))],
        'RGB_Hex': [f'rgb({centers[i][0]},{centers[i][1]},{centers[i][2]})' for i in range(len(centers))]
    })
    
    # RGB statistics table
    rgb_stats_df = pd.DataFrame({
        'Channel': ['Red', 'Green', 'Blue'],
        'Mean': [np.mean(img_rgb[:, :, i]) for i in range(3)],
        'Median': [np.median(img_rgb[:, :, i]) for i in range(3)],
        'Std_Dev': [np.std(img_rgb[:, :, i]) for i in range(3)],
        'Min': [np.min(img_rgb[:, :, i]) for i in range(3)],
        'Max': [np.max(img_rgb[:, :, i]) for i in range(3)]
    })
    
    return metadata_df, cluster_df, rgb_stats_df

def create_interactive_histogram(img_rgb):
    """Create interactive RGB histogram using Plotly."""
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=('Red Channel', 'Green Channel', 'Blue Channel')
    )
    
    colors = ['red', 'green', 'blue']
    for i, color in enumerate(colors):
        channel_data = img_rgb[:, :, i].flatten()
        fig.add_trace(
            go.Histogram(x=channel_data, nbinsx=256, marker_color=color, 
                        name=f'{color.capitalize()} Channel', opacity=0.7),
            row=1, col=i+1
        )
    
    fig.update_xaxes(title_text="Pixel Value", range=[0, 255])
    fig.update_yaxes(title_text="Frequency")
    fig.update_layout(height=400, showlegend=False, title_text="RGB Channel Histograms")
    
    return fig

def create_cluster_pie_chart(cluster_df):
    """Create interactive pie chart for cluster distribution."""
    fig = px.pie(
        cluster_df, 
        values='Percentage', 
        names='Cluster_ID',
        title='Land Cover Distribution',
        color='Cluster_ID',
        color_discrete_sequence=cluster_df['RGB_Hex'].tolist()
    )
    fig.update_traces(textposition='inside', textinfo='percent+label')
    fig.update_layout(height=500)
    return fig

def create_cluster_bar_chart(cluster_df):
    """Create interactive bar chart for cluster analysis."""
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=cluster_df['Cluster_ID'],
        y=cluster_df['Percentage'],
        marker_color=cluster_df['RGB_Hex'],
        text=cluster_df['Percentage'].round(2),
        texttemplate='%{text}%',
        textposition='outside'
    ))
    
    fig.update_layout(
        title='Cluster Distribution (Percentage)',
        xaxis_title='Cluster',
        yaxis_title='Percentage (%)',
        height=400
    )
    
    return fig

def create_rgb_comparison_chart(cluster_df):
    """Create grouped bar chart comparing RGB values across clusters."""
    fig = go.Figure()
    
    fig.add_trace(go.Bar(name='Red', x=cluster_df['Cluster_ID'], y=cluster_df['Red_Value'], marker_color='red'))
    fig.add_trace(go.Bar(name='Green', x=cluster_df['Cluster_ID'], y=cluster_df['Green_Value'], marker_color='green'))
    fig.add_trace(go.Bar(name='Blue', x=cluster_df['Cluster_ID'], y=cluster_df['Blue_Value'], marker_color='blue'))
    
    fig.update_layout(
        title='RGB Values per Cluster',
        xaxis_title='Cluster',
        yaxis_title='Pixel Value',
        barmode='group',
        height=400
    )
    
    return fig

def create_rgb_stats_chart(rgb_stats_df):
    """Create interactive chart for RGB statistics."""
    fig = go.Figure()
    
    metrics = ['Mean', 'Median', 'Std_Dev']
    colors = ['red', 'green', 'blue']
    
    for metric in metrics:
        fig.add_trace(go.Bar(
            name=metric,
            x=rgb_stats_df['Channel'],
            y=rgb_stats_df[metric],
            text=rgb_stats_df[metric].round(2),
            textposition='outside'
        ))
    
    fig.update_layout(
        title='RGB Channel Statistics Comparison',
        xaxis_title='Channel',
        yaxis_title='Value',
        barmode='group',
        height=400
    )
    
    return fig

def create_ndvi_gauge(ndvi_value):
    """Create gauge chart for NDVI value."""
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=ndvi_value,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Pseudo-NDVI", 'font': {'size': 24}},
        delta={'reference': 0.0},
        gauge={
            'axis': {'range': [-1, 1], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': "darkblue"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [-1, -0.2], 'color': '#d62728'},
                {'range': [-0.2, 0], 'color': '#ff7f0e'},
                {'range': [0, 0.2], 'color': '#bcbd22'},
                {'range': [0.2, 1], 'color': '#2ca02c'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': ndvi_value
            }
        }
    ))
    
    fig.update_layout(height=300)
    return fig

def create_kpi_cards(avg_rgb, ndvi_value, img_shape, n_clusters):
    """Create KPI metrics display."""
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="🔴 Average Red",
            value=f"{avg_rgb[0]:.2f}",
            delta=f"{avg_rgb[0] - 127.5:.1f} from mid"
        )
    
    with col2:
        st.metric(
            label="🟢 Average Green",
            value=f"{avg_rgb[1]:.2f}",
            delta=f"{avg_rgb[1] - 127.5:.1f} from mid"
        )
    
    with col3:
        st.metric(
            label="🔵 Average Blue",
            value=f"{avg_rgb[2]:.2f}",
            delta=f"{avg_rgb[2] - 127.5:.1f} from mid"
        )
    
    with col4:
        vegetation_status = "High" if ndvi_value > 0.2 else "Medium" if ndvi_value > 0 else "Low"
        st.metric(
            label="🌿 Pseudo-NDVI",
            value=f"{ndvi_value:.4f}",
            delta=vegetation_status
        )

# Streamlit App
def main():
    st.title("🛰️ Satellite Image Analysis Dashboard")
    st.markdown("Upload a satellite image (JPG) to perform comprehensive Earth Observation analysis with interactive visualizations")
    
    # Sidebar for parameters
    st.sidebar.header("⚙️ Analysis Parameters")
    n_clusters = st.sidebar.slider("Number of Clusters", 3, 7, 4)
    spatial_res = st.sidebar.number_input("Spatial Resolution (meters)", value=10.0, min_value=0.1)
    lat = st.sidebar.number_input("Latitude", value=0.0, format="%.6f")
    lon = st.sidebar.number_input("Longitude", value=0.0, format="%.6f")
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📊 Dashboard Features")
    st.sidebar.markdown("""
    - 📈 Interactive Charts
    - 🎨 RGB Analysis
    - 🗺️ Land Cover Classification
    - 📉 Statistical Insights
    - 💾 Data Export Options
    """)
    
    # File uploader
    uploaded_file = st.file_uploader("Choose a satellite image...", type=['jpg', 'jpeg', 'png'])
    
    if uploaded_file is not None:
        # Load image
        img_rgb = load_image(uploaded_file)
        
        # Extract RGB values
        avg_rgb = extract_average_rgb(img_rgb)
        
        # Calculate pseudo-NDVI
        pseudo_ndvi, avg_pseudo_ndvi = calculate_pseudo_ndvi(img_rgb)
        
        # Perform K-means segmentation
        with st.spinner('Performing analysis...'):
            segmented_img, centers, distribution, labels = perform_kmeans_segmentation(img_rgb, n_clusters)
        
        # Generate data tables
        metadata_df, cluster_df, rgb_stats_df = create_powerbi_data(
            img_rgb, avg_rgb, avg_pseudo_ndvi, centers, distribution, 
            spatial_res, lat, lon
        )
        
        # Create tabs for different sections
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📊 Overview", 
            "🖼️ Image Analysis", 
            "📈 Interactive Charts",
            "📋 Data Tables",
            "💾 Export Data"
        ])
        
        # TAB 1: Overview
        with tab1:
            st.header("Dashboard Overview")
            
            # KPI Cards
            create_kpi_cards(avg_rgb, avg_pseudo_ndvi, img_rgb.shape, n_clusters)
            
            st.markdown("---")
            
            # Two column layout
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📍 Image Metadata")
                st.dataframe(metadata_df, use_container_width=True, hide_index=True)
            
            with col2:
                st.subheader("🌿 Vegetation Index Gauge")
                fig_gauge = create_ndvi_gauge(avg_pseudo_ndvi)
                st.plotly_chart(fig_gauge, use_container_width=True)
            
            st.markdown("---")
            
            # Cluster summary
            st.subheader("🗺️ Land Cover Summary")
            col1, col2 = st.columns([1, 1])
            
            with col1:
                fig_pie = create_cluster_pie_chart(cluster_df)
                st.plotly_chart(fig_pie, use_container_width=True)
            
            with col2:
                fig_bar = create_cluster_bar_chart(cluster_df)
                st.plotly_chart(fig_bar, use_container_width=True)
        
        # TAB 2: Image Analysis
        with tab2:
            st.header("Image Analysis")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Original Satellite Image")
                st.image(img_rgb, use_container_width=True)
            
            with col2:
                st.subheader(f"Segmented Image ({n_clusters} clusters)")
                st.image(segmented_img.astype(np.uint8), use_container_width=True)
            
            st.markdown("---")
            
            st.subheader("Pseudo-NDVI Heatmap")
            fig_ndvi, ax = plt.subplots(figsize=(12, 8))
            im = ax.imshow(pseudo_ndvi, cmap='RdYlGn')
            plt.colorbar(im, ax=ax, label='Pseudo-NDVI')
            ax.set_title('Pseudo-NDVI (Green-Red)/(Green+Red)')
            ax.axis('off')
            st.pyplot(fig_ndvi)
            
            # Interpretation
            st.info("""
            **NDVI Interpretation:**
            - Values > 0.2: High vegetation density
            - Values 0 to 0.2: Medium vegetation
            - Values < 0: Low vegetation / Built-up areas / Water bodies
            """)
        
        # TAB 3: Interactive Charts
        with tab3:
            st.header("Interactive Analytics Dashboard")
            
            # RGB Histograms
            st.subheader("📊 RGB Channel Distribution")
            fig_hist = create_interactive_histogram(img_rgb)
            st.plotly_chart(fig_hist, use_container_width=True)
            
            st.markdown("---")
            
            # RGB Statistics
            st.subheader("📈 RGB Channel Statistics")
            fig_stats = create_rgb_stats_chart(rgb_stats_df)
            st.plotly_chart(fig_stats, use_container_width=True)
            
            st.markdown("---")
            
            # Cluster RGB Comparison
            st.subheader("🎨 RGB Composition by Cluster")
            fig_rgb_comp = create_rgb_comparison_chart(cluster_df)
            st.plotly_chart(fig_rgb_comp, use_container_width=True)
            
            st.markdown("---")
            
            # Additional insights
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📌 Dominant Cluster")
                dominant = cluster_df.loc[cluster_df['Percentage'].idxmax()]
                st.markdown(f"""
                **{dominant['Cluster_ID']}**
                - Coverage: {dominant['Percentage']:.2f}%
                - RGB: ({dominant['Red_Value']}, {dominant['Green_Value']}, {dominant['Blue_Value']})
                - Pixels: {dominant['Pixel_Count']:,}
                """)
            
            with col2:
                st.subheader("📌 Spectral Diversity")
                diversity_score = rgb_stats_df['Std_Dev'].mean()
                st.metric("Average Std Deviation", f"{diversity_score:.2f}")
                st.markdown(f"""
                Spectral diversity indicates the variety of surface types in the image.
                - Score: {diversity_score:.2f}
                - Interpretation: {"High diversity" if diversity_score > 50 else "Low diversity"}
                """)
        
        # TAB 4: Data Tables
        with tab4:
            st.header("Detailed Data Tables")
            
            st.subheader("🗂️ Cluster Analysis Details")
            st.dataframe(cluster_df, use_container_width=True, hide_index=True)
            
            st.markdown("---")
            
            st.subheader("📊 RGB Statistics")
            st.dataframe(rgb_stats_df, use_container_width=True, hide_index=True)
            
            st.markdown("---")
            
            st.subheader("📋 Complete Metadata")
            st.dataframe(metadata_df, use_container_width=True, hide_index=True)
        
        # TAB 5: Export Data
        with tab5:
            st.header("Export Analysis Data")
            
            st.markdown("Download your analysis results in various formats for further processing or reporting.")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.subheader("📄 CSV Files")
                
                csv_metadata = metadata_df.to_csv(index=False)
                st.download_button(
                    label="Download Metadata",
                    data=csv_metadata,
                    file_name="satellite_metadata.csv",
                    mime="text/csv"
                )
                
                csv_cluster = cluster_df.to_csv(index=False)
                st.download_button(
                    label="Download Clusters",
                    data=csv_cluster,
                    file_name="cluster_analysis.csv",
                    mime="text/csv"
                )
                
                csv_rgb = rgb_stats_df.to_csv(index=False)
                st.download_button(
                    label="Download RGB Stats",
                    data=csv_rgb,
                    file_name="rgb_statistics.csv",
                    mime="text/csv"
                )
            
            with col2:
                st.subheader("📊 Excel Workbook")
                
                output = io.BytesIO()
                with pd.ExcelWriter(output, engine='openpyxl') as writer:
                    metadata_df.to_excel(writer, sheet_name='Metadata', index=False)
                    cluster_df.to_excel(writer, sheet_name='Clusters', index=False)
                    rgb_stats_df.to_excel(writer, sheet_name='RGB_Statistics', index=False)
                
                excel_data = output.getvalue()
                st.download_button(
                    label="📥 Download Complete Dataset",
                    data=excel_data,
                    file_name="satellite_analysis_complete.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
            
            with col3:
                st.subheader("📝 Text Report")
                
                acq_date = datetime.now().strftime('%Y-%m-%d')
                report = f"""SATELLITE IMAGE ANALYSIS REPORT
Generated: {acq_date}

=== IMAGE METADATA ===
Spatial Resolution: {spatial_res} meters
Geolocation: {lat}, {lon}
Dimensions: {img_rgb.shape[1]} x {img_rgb.shape[0]} pixels

=== RADIOMETRIC ANALYSIS ===
Average RGB: R={avg_rgb[0]:.2f}, G={avg_rgb[1]:.2f}, B={avg_rgb[2]:.2f}
Pseudo-NDVI: {avg_pseudo_ndvi:.4f}

=== LAND COVER CLASSIFICATION ===
Number of Clusters: {len(centers)}
{cluster_df.to_string(index=False)}

=== RGB STATISTICS ===
{rgb_stats_df.to_string(index=False)}
"""
                
                st.download_button(
                    label="Download Text Report",
                    data=report,
                    file_name="analysis_report.txt",
                    mime="text/plain"
                )
    
    else:
        st.info("👆 Please upload a satellite image to begin analysis")
        
        # Show example insights
        st.markdown("---")
        st.subheader("What you'll get:")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("### 📊 Interactive Charts")
            st.markdown("- RGB histograms")
            st.markdown("- Cluster distribution")
            st.markdown("- Statistical comparisons")
        
        with col2:
            st.markdown("### 🗺️ Land Analysis")
            st.markdown("- K-means segmentation")
            st.markdown("- Pseudo-NDVI mapping")
            st.markdown("- Vegetation assessment")
        
        with col3:
            st.markdown("### 💾 Data Export")
            st.markdown("- CSV/Excel downloads")
            st.markdown("- Detailed reports")
            st.markdown("- Power BI ready data")

if __name__ == "__main__":
    main()