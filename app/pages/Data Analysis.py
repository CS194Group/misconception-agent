import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objs as go
import seaborn as sns
import matplotlib.pyplot as plt

# Set page configuration
st.set_page_config(page_title='AI Model Performance Dashboard', layout='wide')

# Create sample performance data
np.random.seed(42)
models = ['Claude 3 Opus', 'GPT-4', 'Gemini Pro', 'Claude Haiku', 'PaLM 2']
tasks = ['Coding', 'Math', 'Creative Writing', 'Reasoning', 'Translation']

# Generate synthetic performance data
performance_data = pd.DataFrame({
    'Model': np.repeat(models, len(tasks)),
    'Task': tasks * len(models),
    'Performance Score': np.random.uniform(70, 95, len(models) * len(tasks)),
    'Inference Speed (tokens/sec)': np.random.uniform(20, 100, len(models) * len(tasks)),
    'Cost per 1M Tokens': np.random.uniform(5, 50, len(models) * len(tasks))
})

# Title
st.title('🤖 AI Model Performance Comparison Dashboard')

# Layout with three columns
col1, col2, col3 = st.columns(3)

# Line Chart: Performance Scores Across Tasks
with col1:
    st.header('Performance Scores')
    fig_line = px.line(performance_data, x='Task', y='Performance Score', color='Model', 
                       title='Model Performance by Task')
    st.plotly_chart(fig_line, use_container_width=True)

# Pie Chart: Overall Performance Distribution
with col2:
    st.header('Performance Distribution')
    model_avg_performance = performance_data.groupby('Model')['Performance Score'].mean()
    fig_pie = px.pie(values=model_avg_performance, names=model_avg_performance.index, 
                     title='Average Performance Share')
    st.plotly_chart(fig_pie, use_container_width=True)

# Box Plot: Performance Variability
with col3:
    st.header('Performance Variability')
    fig_box = go.Figure(data=[
        go.Box(name=model, y=performance_data[performance_data['Model'] == model]['Performance Score'])
        for model in models
    ])
    fig_box.update_layout(title='Performance Score Distribution')
    st.plotly_chart(fig_box, use_container_width=True)

# Heatmap: Performance and Speed Correlation
st.header('Performance vs Inference Speed Heatmap')
performance_speed = performance_data.pivot_table(
    index='Model', 
    columns='Task', 
    values='Inference Speed (tokens/sec)'
)

fig, ax = plt.subplots(figsize=(10, 6))
sns.heatmap(performance_speed, annot=True, cmap='YlGnBu', ax=ax)
st.pyplot(fig)

# Scatter Plot: Performance vs Cost
st.header('Performance vs Cost Analysis')
fig_scatter = px.scatter(performance_data, 
    x='Cost per 1M Tokens', 
    y='Performance Score', 
    color='Model',
    size='Inference Speed (tokens/sec)',
    title='Performance, Cost, and Speed Relationship')
st.plotly_chart(fig_scatter, use_container_width=True)

# Radar Chart: Multidimensional Performance
st.header('Multidimensional Performance Radar')
tasks_radar = ['Coding', 'Math', 'Creative Writing', 'Reasoning', 'Translation']
claude_scores = performance_data[performance_data['Model'] == 'Claude 3 Opus'].groupby('Task')['Performance Score'].mean()

fig_radar = go.Figure(data=go.Scatterpolar(
    r=claude_scores.values,
    theta=tasks_radar,
    fill='toself'
))
fig_radar.update_layout(title='Claude 3 Opus Performance Across Tasks')
st.plotly_chart(fig_radar, use_container_width=True)

# Footer
st.markdown('*Note: Data is synthetically generated for illustrative purposes*')