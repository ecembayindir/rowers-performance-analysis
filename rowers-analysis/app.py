import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Load your processed data
df = pd.read_csv('rowers-analysis/merged_series_data.csv')

# Sidebar for tab selection
st.sidebar.title("🏁 Navigation")
tab_labels = [
    "About",
    "General Insights",
    "2000m Completion Time",
    "Average Speed (2000m)",
    "500m Split Speed",
    "2000m Stroke Length",
    "500m Stroke Length",
    "Heatmap"
]
tab_selection = st.sidebar.radio("", tab_labels)

# Sidebar for filtering
if tab_selection not in ["About", "Overview"]:  # Only show filters if not on the About or Overview page
    st.sidebar.markdown("### 👤 Select Participant")
    participants = ["All"] + list(df['participant'].unique())  # Add "All" to the participants list
    selected_participant = st.sidebar.selectbox("Participant", participants)

    # Filter the data based on the selected participant
    if selected_participant != "All":
        df = df[df['participant'] == selected_participant]

# Function to set the custom styles based on tab selection
def set_custom_styles():
    st.markdown("""
        <style>
        .reportview-container {
            font-family: 'Arial', sans-serif;
        }
        .sidebar .sidebar-content {
            background: linear-gradient(to bottom, #f0f0f0, #d9e4dd);
            color: #333;
        }
        h2, h3 {
            color: #333;
            font-family: 'Arial', sans-serif;
        }
        .stSidebar > div {
            font-family: 'Arial', sans-serif;
        }
        </style>
        """, unsafe_allow_html=True)

# Apply custom styles globally to maintain consistency
set_custom_styles()

# Main dashboard title
st.title("🚣‍♂️ Rowing Competition Dashboard")

# Function to display the About page
def show_about_page():
    st.header("About This Project")
    st.markdown("""
        Welcome to the **Rowing Competition Dashboard**! 🏆 This dashboard provides a comprehensive analysis of the rowing competition performance metrics.

        Here’s what you'll find:
        - **Completion Time Analysis**: Compare participants' 2000m completion times.
        - **Speed and Stroke Metrics**: Examine average speeds, stroke rates, and stroke lengths.
        - **Visual Analysis**: Heatmaps and split-by-split comparisons to help you understand the efficiency of each participant.

        The dashboard is interactive — explore the data, select specific participants, and gain insights into how each rower performed!

        **Use the tabs on the left to navigate through the different analyses.** 📊
    """)

# Function to display the Overview page
def show_overview_page():
    st.header("General Insights")
    st.markdown("""
        - The maximum speed achieved was **19.54 km/h**.
        - The stroke rate at maximum speed was **34.00 strokes/minute**.
        - The average stroke length at maximum speed was **9.62 meters**.

        The overview presents the highest-level insights into the competition's best performance, helping identify efficient rowing techniques.
    """)

    # Plot the Average Speed vs. Average Stroke Length with color-coded stroke rates
    fig = px.scatter(df, x='avg_speed_2000', y='avg_stroke_length_2000m',
                     color='avg_spm_2000m',
                     labels={'x': 'Average Speed (km/h)', 'y': 'Average Stroke Length (meters)', 'color': 'Stroke Rate (SPM)'},
                     title='Average Speed vs. Average Stroke Length for 500m Sections',
                     template='plotly_white')
    fig.update_traces(marker=dict(size=10, opacity=0.7, line=dict(width=1, color='DarkSlateGrey')))
    fig.add_trace(go.Scatter(x=[19.54], y=[9.62], mode='markers+text',
                             marker=dict(color='red', size=15),
                             text=['Max Speed Point'],
                             textposition='top center'))
    st.plotly_chart(fig, use_container_width=True)

# Function to plot 2000m completion time
def plot_2000m_completion_time():
    fig = px.bar(df, x='participant', y='total_time_2000m',
                 title='2000m Completion Time by Participant',
                 labels={'total_time_2000m': 'Completion Time (seconds)', 'participant': 'Participants'},
                 color='total_time_2000m',
                 color_continuous_scale=px.colors.sequential.Plasma,
                 template='plotly_dark')
    fig.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig, use_container_width=True)

# Function to plot average speed over 2000m with stroke rate
def plot_avg_speed_2000m():
    fig = go.Figure()
    fig.add_trace(go.Bar(x=df['participant'], y=df['avg_speed_2000'],
                         name='Avg Speed (km/h)',
                         marker=dict(color='#00b4d8')))
    fig.add_trace(go.Scatter(x=df['participant'], y=df['avg_spm_2000m'],
                             mode='lines+markers',
                             name='Stroke Rate (SPM)',
                             line=dict(color='#e63946')))
    fig.update_layout(title='Average Speed and Stroke Rate (2000m)',
                      yaxis_title='Speed (km/h) / Stroke Rate (SPM)',
                      xaxis=dict(title='Participants', tickangle=-45),
                      legend=dict(x=0.01, y=0.99),
                      template='ggplot2')
    st.plotly_chart(fig, use_container_width=True)

# Function to plot split-by-split average speed for each series
def plot_split_by_split_speed():
    fig = go.Figure()

    for series in df['series'].unique():
        df_series = df[df['series'] == series]
        speeds = [
            df_series["avg_speed_500m_1"].values[0],
            df_series["avg_speed_500m_2"].values[0],
            df_series["avg_speed_500m_3"].values[0],
            df_series["avg_speed_500m_4"].values[0]
        ]
        fig.add_trace(go.Scatter(x=[1, 2, 3, 4], y=speeds,
                                 mode='lines+markers',
                                 name=f'Series {series}',
                                 line=dict(shape='spline')))

    fig.update_layout(title='500m Split-by-Split Average Speed by Series',
                      xaxis_title='500m Split',
                      yaxis_title='Speed (km/h)',
                      xaxis=dict(tickmode='array', tickvals=[1, 2, 3, 4], ticktext=['500m 1', '500m 2', '500m 3', '500m 4']),
                      legend=dict(x=0.01, y=0.99),
                      template='simple_white')
    st.plotly_chart(fig, use_container_width=True)

# Function to plot 2000m average stroke length for each participant
def plot_2000m_stroke_length():
    fig = px.bar(df, x='participant', y='avg_stroke_length_2000m',
                 title='2000m Stroke Length by Participant',
                 labels={'avg_stroke_length_2000m': 'Stroke Length (meters)', 'participant': 'Participants'},
                 color='avg_stroke_length_2000m',
                 color_continuous_scale=px.colors.sequential.Viridis,
                 template='plotly_white')
    fig.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig, use_container_width=True)

# Function to plot 500m split average stroke length for each participant
def plot_500m_split_stroke_length():
    fig = go.Figure()

    for i in range(len(df)):
        avg_500m_stroke_length = [
            df["avg_stroke_length_500m_1"].iloc[i],
            df["avg_stroke_length_500m_2"].iloc[i],
            df["avg_stroke_length_500m_3"].iloc[i],
            df["avg_stroke_length_500m_4"].iloc[i]
        ]
        fig.add_trace(go.Scatter(x=[1, 2, 3, 4], y=avg_500m_stroke_length,
                                 mode='lines+markers',
                                 name=f'{df["participant"].iloc[i]} (500m Splits)',
                                 line=dict(shape='linear', width=2),
                                 marker=dict(size=10, line=dict(width=2))))

    fig.update_layout(title='500m Split Average Stroke Length by Participant',
                      xaxis_title='500m Split',
                      yaxis_title='Stroke Length (meters)',
                      xaxis=dict(tickmode='array', tickvals=[1, 2, 3, 4], ticktext=['500m 1', '500m 2', '500m 3', '500m 4']),
                      legend=dict(x=0.01, y=0.99),
                      template='plotly_dark')
    st.plotly_chart(fig, use_container_width=True)

# Function to create a heatmap for stroke rate vs speed
def plot_stroke_rate_vs_speed_heatmap():
    heatmap_df = pd.DataFrame({
        'stroke_rate': pd.concat(
            [df['split_spm_500m_1'], df['split_spm_500m_2'], df['split_spm_500m_3'], df['split_spm_500m_4']]),
        'speed': pd.concat(
            [df['avg_speed_500m_1'], df['avg_speed_500m_2'], df['avg_speed_500m_3'], df['avg_speed_500m_4']])
    })

    fig = px.density_heatmap(heatmap_df, x='stroke_rate', y='speed', nbinsx=20, nbinsy=20,
                             title='Heatmap: Stroke Rate vs Speed',
                             color_continuous_scale=px.colors.sequential.Inferno,
                             template='seaborn')
    fig.update_layout(xaxis_title='Stroke Rate (SPM)', yaxis_title='Speed (km/h)')
    st.plotly_chart(fig, use_container_width=True)

# Display the appropriate plot based on the selected tab
if tab_selection == "About":
    show_about_page()

elif tab_selection == "General Insights":
    show_overview_page()

elif tab_selection == "2000m Completion Time":
    plot_2000m_completion_time()

elif tab_selection == "Average Speed (2000m)":
    plot_avg_speed_2000m()

elif tab_selection == "500m Split Speed":
    plot_split_by_split_speed()

elif tab_selection == "2000m Stroke Length":
    plot_2000m_stroke_length()

elif tab_selection == "500m Stroke Length":
    plot_500m_split_stroke_length()

elif tab_selection == "Heatmap":
    plot_stroke_rate_vs_speed_heatmap()
