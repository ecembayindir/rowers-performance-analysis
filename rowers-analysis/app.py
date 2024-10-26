import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import statsmodels.api as sm
from statsmodels.formula.api import ols

st.set_page_config(page_title="Sorbonne Python Project", page_icon="🤖")


# Load your processed data
df = pd.read_csv('rowers-analysis/merged_series_data.csv')
anova_results_df = pd.read_csv('rowers-analysis/anova_results.csv')
coef_df = pd.read_csv('rowers-analysis/ols_coefficients.csv')
model_summary_df = pd.read_csv('rowers-analysis/ols_summary.csv')
noc = pd.read_csv('rowers-analysis/data_externes_aviron.csv')

# Sidebar for tab selection
st.sidebar.title("🏁 Navigation")
tab_labels = [
    "About",
    "General Insights",
    "2000m Completion Time",
    "Average Speed (2000m)",
    "Average 500m Split Speed",
    "2000m Stroke Length",
    "500m Stroke Length",
    "Heatmap",
    "Correlation Analysis",
    "Gender & Competitor Analysis",
    "Machine Learning Prediction"
]
tab_selection = st.sidebar.radio("", tab_labels)

# Sidebar for filtering
if tab_selection not in ["About", "Correlation Analysis", "Gender & Competitor Analysis",
                         "Machine Learning Prediction"]:
    st.sidebar.markdown("### 👤 Select Participant")
    participants = ["All"] + list(df['participant'].unique())  # Add "All" to the participants list
    selected_participant = st.sidebar.selectbox("Participant", participants)
    if selected_participant != "All":
        df = df[df['participant'] == selected_participant]


# Function to set custom styles
def set_custom_styles():
    st.markdown("""
        <style>
        .reportview-container { font-family: 'Arial', sans-serif; }
        .sidebar .sidebar-content { background: linear-gradient(to bottom, #f0f0f0, #d9e4dd); color: #333; }
        h2, h3 { color: #333; font-family: 'Arial', sans-serif; }
        .stSidebar > div { font-family: 'Arial', sans-serif; }
        </style>
        """, unsafe_allow_html=True)

set_custom_styles()

# Main dashboard title
st.title("🚣‍♂️ Rowing Competition Dashboard")

# About page
def show_about_page():
    st.header("About This Project")
    st.markdown("""
        Welcome to the **Rowing Competition Dashboard**! 🏆 This dashboard provides a comprehensive analysis of the rowing competition performance metrics.

        - **Completion Time Analysis**: Compare participants' 2000m completion times.
        - **Speed and Stroke Metrics**: Examine average speeds, stroke rates, and stroke lengths.
        - **Visual Analysis**: Heatmaps and split-by-split comparisons to help you understand the efficiency of each participant.

        **Use the tabs on the left to navigate through the different analyses.** 📊
    """)

# General insights
def show_overview_page():
    st.header("General Insights")
    st.markdown("""
        - The maximum speed achieved was **19.54 km/h**.
        - The cadence at maximum speed was **34.00 strokes/minute**.
        - The average stroke length at maximum speed was **9.62 meters**.

        The overview presents the highest-level insights into the competition's best performance, helping identify efficient rowing techniques.
    """)

    # Step 1: Prepare the data for plotting
    # X-axis: Average cadence for each 500m section
    x_values = pd.concat([df["split_spm_500m_1"],
                          df["split_spm_500m_2"],
                          df["split_spm_500m_3"],
                          df["split_spm_500m_4"]]).reset_index(drop=True)

    # Y-axis: Average stroke length for each 500m section
    y_values = pd.concat([df["avg_stroke_length_500m_1"],
                          df["avg_stroke_length_500m_2"],
                          df["avg_stroke_length_500m_3"],
                          df["avg_stroke_length_500m_4"]]).reset_index(drop=True)

    # Speed values for determining the maximum speed point
    speed_values = pd.concat([df["avg_speed_500m_1"],
                              df["avg_speed_500m_2"],
                              df["avg_speed_500m_3"],
                              df["avg_speed_500m_4"]]).reset_index(drop=True)

    # Create a DataFrame for plotting
    plot_df = pd.DataFrame({
        'Average Cadence (SPM)': x_values,
        'Average Stroke Length (meters)': y_values,
        'Speed (km/h)': speed_values
    })

    # Step 2: Find the point of maximum speed
    max_speed_index = speed_values.idxmax()
    max_speed = speed_values.loc[max_speed_index]
    max_cadence = x_values.loc[max_speed_index]
    max_stroke_length = y_values.loc[max_speed_index]

    # Step 3: Create an interactive scatter plot using Plotly
    fig = px.scatter(
        plot_df,
        x='Average Cadence (SPM)',
        y='Average Stroke Length (meters)',
        color='Speed (km/h)',
        title='Average Cadence vs. Average Stroke Length for 500m Sections',
        labels={
            'Average Cadence (SPM)': 'Average Cadence (strokes/minute)',
            'Average Stroke Length (meters)': 'Average Stroke Length (meters)',
            'Speed (km/h)': 'Speed (km/h)'
        },
        template='plotly_white',
        color_continuous_scale='Viridis'
    )

    # Highlight the point of maximum speed with a star and correct tooltip
    fig.add_trace(
        go.Scatter(
            x=[max_cadence],
            y=[max_stroke_length],
            mode='markers',
            marker=dict(color='red', size=15, symbol='star'),
            name='Max Speed Point',
            hovertemplate=f'Cadence: {max_cadence:.2f} strokes/minute<br>'
                          f'Stroke Length: {max_stroke_length:.2f} meters<br>'
                          f'Speed: {max_speed:.2f} km/h'
        )
    )

    # Improve the layout for better user experience
    fig.update_layout(
        xaxis_title='Average Cadence (strokes/minute)',
        yaxis_title='Average Stroke Length (meters)',
        xaxis=dict(showgrid=True, zeroline=True, gridcolor='LightGrey'),
        yaxis=dict(showgrid=True, zeroline=True, gridcolor='LightGrey'),
        hovermode='closest',
        height=600
    )

    # Customize marker size and opacity
    fig.update_traces(
        marker=dict(
            size=10,
            opacity=0.7,
            line=dict(width=1, color='DarkSlateGrey')
        ),
        selector=dict(mode='markers')
    )

    # Step 4: Display the plot
    st.plotly_chart(fig, use_container_width=True)

# Completion time analysis
def plot_2000m_completion_time():
    st.header("2000m Completion Time Analysis")
    st.markdown("This chart compares each participant's total completion time for the 2000m race.")
    fig = px.bar(df, x='participant', y='total_time_2000m',
                 title='2000m Completion Time by Participant',
                 labels={'total_time_2000m': 'Completion Time (seconds)', 'participant': 'Participants'},
                 color='total_time_2000m',
                 color_continuous_scale=px.colors.sequential.Plasma,
                 template='plotly_dark')
    st.plotly_chart(fig, use_container_width=True)

# Average speed and stroke rate analysis
def plot_avg_speed_2000m():
    st.header("Average Speed and Stroke Rate Analysis (2000m)")
    st.markdown("This graph displays participants' average speed along with stroke rate for the entire 2000m race.")
    fig = go.Figure()
    fig.add_trace(go.Bar(x=df['participant'], y=df['avg_speed_2000'], name='Avg Speed (km/h)', marker=dict(color='#00b4d8')))
    fig.add_trace(go.Scatter(x=df['participant'], y=df['avg_spm_2000m'], mode='lines+markers', name='Stroke Rate (SPM)', line=dict(color='#e63946')))
    st.plotly_chart(fig, use_container_width=True)


# Function to plot 500m split average stroke length for each participant
def plot_500m_split_stroke_length():
    st.header("500m Split Stroke Length by Participant")
    st.markdown("This line chart shows the average stroke length across each 500m split for individual participants.")

    fig = go.Figure()

    # Error handling in case any columns are missing or contain NaN values
    try:
        for i in range(len(df)):
            avg_500m_stroke_length = [
                df["avg_stroke_length_500m_1"].iloc[i] if "avg_stroke_length_500m_1" in df else None,
                df["avg_stroke_length_500m_2"].iloc[i] if "avg_stroke_length_500m_2" in df else None,
                df["avg_stroke_length_500m_3"].iloc[i] if "avg_stroke_length_500m_3" in df else None,
                df["avg_stroke_length_500m_4"].iloc[i] if "avg_stroke_length_500m_4" in df else None
            ]

            # Skip participant if any of their split values are missing
            if any(pd.isna(avg_500m_stroke_length)):
                continue

            fig.add_trace(go.Scatter(
                x=[1, 2, 3, 4],
                y=avg_500m_stroke_length,
                mode='lines+markers',
                name=f'{df["participant"].iloc[i]} (500m Splits)',
                line=dict(shape='linear', width=2),
                marker=dict(size=10, line=dict(width=2))
            ))

        fig.update_layout(
            title='500m Split Average Stroke Length by Participant',
            xaxis_title='500m Split',
            yaxis_title='Stroke Length (meters)',
            xaxis=dict(tickmode='array', tickvals=[1, 2, 3, 4], ticktext=['500m 1', '500m 2', '500m 3', '500m 4']),
            legend=dict(x=0.01, y=0.99),
            template='plotly_dark'
        )
        st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"Error displaying 500m split stroke length: {e}")

# Split-by-split average speed
def plot_split_by_split_speed():
    st.header("500m Split-by-Split Average Speed")
    st.markdown("This chart shows the average speed achieved in each 500m section of the race by series.")
    fig = go.Figure()

    for series in df['series'].unique():
        df_series = df[df['series'] == series]
        speeds = [df_series[f"avg_speed_500m_{i+1}"].values[0] for i in range(4)]
        fig.add_trace(go.Scatter(x=[1, 2, 3, 4], y=speeds, mode='lines+markers', name=f'Series {series}', line=dict(shape='spline')))

    fig.update_layout(xaxis_title='500m Split', yaxis_title='Speed (km/h)', xaxis=dict(tickvals=[1, 2, 3, 4]), template='simple_white')
    st.plotly_chart(fig, use_container_width=True)

# Stroke length analysis
def plot_2000m_stroke_length():
    st.header("2000m Stroke Length Analysis")
    st.markdown("This chart shows the average stroke length for each participant over the entire 2000m race.")
    fig = px.bar(df, x='participant', y='avg_stroke_length_2000m',
                 title='2000m Stroke Length by Participant',
                 labels={'avg_stroke_length_2000m': 'Stroke Length (meters)', 'participant': 'Participants'},
                 color='avg_stroke_length_2000m',
                 color_continuous_scale=px.colors.sequential.Viridis,
                 template='plotly_white')
    st.plotly_chart(fig, use_container_width=True)

# Heatmap for stroke rate vs speed
def plot_stroke_rate_vs_speed_heatmap():
    st.header("Heatmap: Stroke Rate vs Speed")
    st.markdown("This heatmap visualizes the relationship between stroke rate and speed across 500m splits.")
    heatmap_df = pd.DataFrame({
        'stroke_rate': pd.concat([df[f'split_spm_500m_{i+1}'] for i in range(4)]),
        'speed': pd.concat([df[f'avg_speed_500m_{i+1}'] for i in range(4)])
    })

    fig = px.density_heatmap(heatmap_df, x='stroke_rate', y='speed', nbinsx=20, nbinsy=20, color_continuous_scale=px.colors.sequential.Inferno, template='seaborn')
    st.plotly_chart(fig, use_container_width=True)

# Correlation analysis
def show_correlation_analysis():
    st.header("Correlation Analysis")
    st.markdown("This section provides insights into the performance correlation with calories spent.")
    numeric_columns = ['calories', 'total_time_2000m_seconds', 'avg_speed_2000', 'avg_speed_500m_1', 'avg_speed_500m_2', 'avg_speed_500m_3', 'avg_speed_500m_4']
    df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric, errors='coerce')
    correlation_matrix = df[numeric_columns].corr()

    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='RdBu', linewidths=0.5, ax=ax)
    st.pyplot(fig)

# Function to show gender and competitor analysis with tabs
def show_gender_competitor_analysis():
    st.header("Gender & Competitor Analysis")

    # Create tabs for each part of the analysis
    tab1, tab2, tab3 = st.tabs(["Distribution of Average Speed by Gender", "Comparison by Participant Type and Gender",
                                "ANOVA and OLS Results"])

    # Tab 1: Distribution of Average Speed by Gender
    with tab1:
        st.subheader("Distribution of Average Speed by Gender")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.histplot(data=df, x='avg_speed_2000', hue='sex', kde=True, bins=10, element='step', stat='density',
                     common_norm=False, palette={'M': 'skyblue', 'F': 'lightpink'}, ax=ax)
        ax.set_title('Histogram of Average Speed by Gender')
        st.pyplot(fig)
        st.markdown("""
        **Comment:**  
        This histogram shows the distribution of average speeds by gender:

        - Each bar represents the number of participants in a specific speed range.
        - Bars are colored blue for men and pink for women.
        - Density indicates the concentration of participants in each speed range.
        """)

    # Tab 2: Comparison by Participant Type and Gender
    with tab2:
        st.subheader("Comparison by Participant Type and Gender")
        df['type_participant'] = 'Amateur'
        noc['type_participant'] = 'Competitor'
        combined_df = pd.concat([df[['participant', 'avg_speed_2000', 'sex', 'type_participant']], noc])

        fig, ax = plt.subplots(figsize=(12, 8))
        sns.boxplot(x='type_participant', y='avg_speed_2000', hue='sex', data=combined_df,
                    palette={'M': 'skyblue', 'F': 'lightpink'}, ax=ax)
        ax.set_title("Speed by Gender and Participant Type")
        ax.set_xlabel("Type of Participant")
        ax.set_ylabel("Average Speed (km/h)")
        st.pyplot(fig)
        st.markdown("""
        **Comment:**  
        For amateur participants, men generally achieve faster speeds than women, as indicated by higher median and quartile values. 
        Women also tend to take longer to complete the 2000m race.

        To further analyze amateur performances and compare them with Olympic athletes, we referenced the 2000m rowing events from 
        the [Skiff - Femmes Report on the official Olympics website for Paris 2024](https://olympics.com/fr/paris-2024/rapports/aviron/skiff---femmes).

        We focused on data such as average speed over 2000m, strokes per minute (spm), completion time, and gender. These details are compiled in the attached CSV file for further analysis.
        """)

    # Tab 3: ANOVA and OLS Results
    with tab3:
        st.subheader("ANOVA Results")
        st.dataframe(anova_results_df)

        st.subheader("OLS Regression Coefficients")
        st.dataframe(coef_df)
        st.markdown("""
        **Comment:**  
        - Amateur participants have a lower speed compared to real competitors.
        - At a 5% significance level, speed is influenced by the type of participant.
        - Amateur participants have a speed that is 2.27 km/h lower than real competitors.
        - There is no interaction effect between gender and type of participant.
        """)


# Machine Learning Prediction
def show_machine_learning_prediction():
    st.header("Machine Learning Prediction")

    X = df[['avg_spm_2000m', 'total_time_2000m_seconds']]
    y = df['avg_speed_2000']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression().fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)

    st.write("Model Results:")
    st.write("Coefficients:", model.coef_)
    st.write("Intercept:", model.intercept_)
    st.write("Mean Squared Error:", mse)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(y_test, y_pred, color='dodgerblue', edgecolor='k', alpha=0.7)
    ax.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--')
    ax.set_xlabel("Actual Average Speed (km/h)")
    ax.set_ylabel("Predicted Average Speed (km/h)")
    ax.set_title("Actual vs Predicted Average Speed")
    st.pyplot(fig)
    st.subheader("Model Performance")
    # Show the Mean Squared Error (MSE) or other evaluation metrics if available
    st.markdown("**Mean Squared Error (MSE):** 0.0534")

    st.markdown("""
        **Comment:**  
        The linear model is used to predict the average speed based on the average stroke rate (SPM) and total time for 2000m.

        - **Stroke Rate (SPM) Coefficient:** 0.0347  
          This indicates that, on average, for each unit increase in stroke rate, the average speed increases by 0.0347 km/h.

        - **Total Time for 2000m Coefficient:** -0.0284  
          This indicates that, on average, for each additional second taken to complete the 2000m, the average speed decreases by 0.0284 km/h.

        - **Intercept:** 27.83  
          This represents the estimated average speed when both predictors (stroke rate and total time) are zero.

        - **Model Performance (MSE):**  
          The Mean Squared Error (MSE) was calculated to be 0.0534, representing the average squared difference between actual and predicted average speeds. A lower MSE indicates better performance of the model.
        """)

# Display plots based on tab selection
if tab_selection == "About":
    show_about_page()
elif tab_selection == "General Insights":
    show_overview_page()
elif tab_selection == "2000m Completion Time":
    plot_2000m_completion_time()
elif tab_selection == "Average Speed (2000m)":
    plot_avg_speed_2000m()
elif tab_selection == "Average 500m Split Speed":
    plot_split_by_split_speed()
elif tab_selection == "2000m Stroke Length":
    plot_2000m_stroke_length()
elif tab_selection == "500m Stroke Length":
    plot_500m_split_stroke_length()
elif tab_selection == "Heatmap":
    plot_stroke_rate_vs_speed_heatmap()
elif tab_selection == "Correlation Analysis":
    show_correlation_analysis()
elif tab_selection == "Gender & Competitor Analysis":
    show_gender_competitor_analysis()
elif tab_selection == "Machine Learning Prediction":
    show_machine_learning_prediction()
