import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

st.title("Canteen Rush Predictor (Full Live Demo)")

# ---------------- Load Dataset ----------------
url = "https://raw.githubusercontent.com/bloomingcodeby-iffath/MU_Canteen_Rush_Predictor/refs/heads/main/canteen_rush_data.csv"
df = pd.read_csv(url)

# ---------------- Load Model ----------------
rf = pickle.load(open("rf_model.pkl", "rb"))

# ---------------- Input Section ----------------
st.sidebar.header("Input Data")

time = st.sidebar.slider("Time", 9, 17, 12)
lunch = st.sidebar.selectbox("Lunch Time", [0,1])
day = st.sidebar.selectbox("Day", df['Day'].unique())
weather = st.sidebar.selectbox("Weather", df['Weather'].unique())

# Encode
day_map = {d:i for i,d in enumerate(df['Day'].unique())}
weather_map = {w:i for i,w in enumerate(df['Weather'].unique())}

X = pd.DataFrame([[time, lunch, day_map[day], weather_map[weather]]],
                 columns=['Time','Lunch_Time','Day_Encoded','Weather_Encoded'])

# ---------------- Prediction ----------------
pred = rf.predict(X)[0]

def get_rush_level(students):
    if students < 20:
        return 'Low'
    elif students < 40:
        return 'Medium'
    else:
        return 'High'

rush = get_rush_level(pred)

st.subheader(f"Predicted Students: {int(pred)}")

color_map = {'Low':'green','Medium':'orange','High':'red'}
st.markdown(f"### Rush Level: <span style='color:{color_map[rush]}'>{rush}</span>", unsafe_allow_html=True)

# ---------------- Graph Section ----------------
st.header("Visualizations")

# 1. Rush Level vs Students
st.subheader("Rush Level vs Students (Bar Chart)")
fig1, ax1 = plt.subplots()
sns.barplot(x='Rush_Level', y='Students', data=df, palette=color_map, ax=ax1)
st.pyplot(fig1)

# 2. Rush Level Distribution
st.subheader("Rush Level Distribution (Pie Chart)")
fig2, ax2 = plt.subplots()
rush_counts = df['Rush_Level'].value_counts()
ax2.pie(rush_counts, labels=rush_counts.index, autopct='%1.1f%%', colors=['yellow','orange','red'])
st.pyplot(fig2)

# 3. Time vs Students
st.subheader("Time vs Students (Line Plot)")
fig3, ax3 = plt.subplots()
ax3.plot(df['Time'], df['Students'], marker='o')
ax3.set_title("Time vs Students")
st.pyplot(fig3)

# 4. Scatter Plot
st.subheader("Time vs Students (Scatter Plot)")
fig4, ax4 = plt.subplots()
ax4.scatter(df['Time'], df['Students'])
ax4.set_title("Scatter Plot")
st.pyplot(fig4)

# 5. Weather Impact
st.subheader("Weather vs Students (Box Plot)")
fig5, ax5 = plt.subplots()
sns.boxplot(x='Weather', y='Students', data=df, ax=ax5)
st.pyplot(fig5)

# 6. Lunch Impact
st.subheader("Lunch Time vs Students (Bar Plot)")
fig6, ax6 = plt.subplots()
sns.barplot(x='Lunch_Time', y='Students', data=df, ax=ax6)
st.pyplot(fig6)

# 7. Correlation Heatmap
st.subheader("Correlation Heatmap")
df['Rush_Level_Encoded'] = df['Rush_Level'].map({'Low':0,'Medium':1,'High':2})
corr = df[['Time','Lunch_Time','Students','Rush_Level_Encoded']].corr()

fig7, ax7 = plt.subplots()
sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax7)
st.pyplot(fig7)

# 8. Pairplot
st.subheader("Feature Relationships (Pairplot)")
fig8 = sns.pairplot(df, hue='Rush_Level')
st.pyplot(fig8)

# 9. Stacked Bar
st.subheader("Time vs Rush Level (Stacked Bar Chart)")
rush_time = df.groupby(['Time','Rush_Level'])['Students'].sum().unstack()
fig9 = rush_time.plot(kind='bar', stacked=True).figure
st.pyplot(fig9)