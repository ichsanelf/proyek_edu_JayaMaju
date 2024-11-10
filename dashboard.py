import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

sns.set_style("dark")

def buat_rental_harian_df(df):
    buat_rental_harian_df = df.resample(rule='D', on='dteday').agg({
        "casual" : "sum",
        "registered" : "sum",
        "cnt": "sum"
    })
    
    buat_rental_harian_df = buat_rental_harian_df.reset_index()
    buat_rental_harian_df.rename(columns={
        "casual" : "casual_sum",
        "registered" : "register_sum",
        "cnt": "total"
    }, inplace=True)
    
    return buat_rental_harian_df

def buat_rental_jam_df(df):
    buat_rental_jam_df = df.groupby(by="hr").cnt.sum().reset_index()
    buat_rental_jam_df.rename(columns={"cnt" : "total"}, inplace=True)

    return buat_rental_jam_df

def buat_grafik_musim(df):
    buat_grafik_musim = df.groupby(by="season").cnt.count().reset_index()
    buat_grafik_musim.rename(columns={"cnt": "total"}, inplace=True)

    return buat_grafik_musim

def buat_grafik_cuaca(df):
    buat_grafik_cuaca = df.groupby(by="weathersit").cnt.count().reset_index()
    buat_grafik_cuaca.rename(columns={"cnt": "total"}, inplace=True)

    return buat_grafik_cuaca

def buat_persentase(df):
    casual_sum = df['casual'].sum()
    register_sum = df['registered'].sum()
    buat_persentase = pd.DataFrame({ 
        'jenis': ['Casual', 'Registered'], 
        'total_rental': [casual_sum, register_sum]
    })

    return buat_persentase

rental_df = pd.read_csv("hour_df.csv")

datetime_columns = ["dteday"]
rental_df.sort_values(by="dteday", inplace=True)
rental_df.reset_index(inplace=True)
 
for column in datetime_columns:
    rental_df[column] = pd.to_datetime(rental_df[column])

min_date = rental_df["dteday"].min()
max_date = rental_df["dteday"].max()

with st.sidebar:
  
    st.header(":date: Calendar :date:")

    # Mengambil start_date & end_date dari date_input
    start_date, end_date = st.date_input(
            label='Choose Date',min_value=min_date,
            max_value=max_date,
            value=[min_date, max_date]
        )
    
    main_df = rental_df[(rental_df["dteday"] >= str(start_date)) & 
        (rental_df["dteday"] <= str(end_date))]

rental_harian_df = buat_rental_harian_df(main_df)
rental_jam_df = buat_rental_jam_df(main_df)
grafik_musim = buat_grafik_musim(main_df)
grafik_cuaca = buat_grafik_cuaca(main_df)
persetanse_pie = buat_persentase(main_df) 

st.title(":bike: :blue[Analysis] of Bike Rental :bike:")

st.divider()

st.header("Bike Rental Data From :blue[2011] to :blue[2012]", divider='blue')
tab1, tab2 = st.tabs(["Monthly/Daily Data", "Rental Data per Hour"])

with tab1:
    total_rental = rental_harian_df.total.sum()
    st.metric("Total Rental", value=total_rental)

    fig, ax = plt.subplots(figsize=(16, 8))
    sns.lineplot(x="dteday", y="casual_sum", data=rental_harian_df, marker='o', linewidth=2, label="Casual", color='blue')
    sns.lineplot(x="dteday", y="register_sum", data=rental_harian_df, marker='o', linewidth=2, label="Registered", color='green')
    sns.lineplot(x="dteday", y="total", data=rental_harian_df, marker='o', linewidth=2, label="Total", color ='red')
    ax.tick_params(axis='y', labelsize=20)
    ax.tick_params(axis='x', labelsize=15)
    ax.grid(True) 
    st.pyplot(fig)

with tab2:
    total_rental = rental_jam_df.total.sum()
    st.metric("Total Rental", value=total_rental)

    fig, ax = plt.subplots(figsize=(20, 10))

    ax.plot(
       rental_jam_df["hr"],
       rental_jam_df["total"],
       marker='o', 
       linewidth=2,
       color="#90CAF9"
    )

    ax.grid(True) 
    st.pyplot(fig)

st.divider()

st.header("Bike Rental Data by :rainbow[Season]", divider='orange')    

fig, ax = plt.subplots(figsize=(20, 10))

sns.barplot(
    y="season", 
    x="total", 
    hue="total", 
    data=grafik_musim.sort_values(by="total", ascending=False), 
    palette="dark:blue", 
    legend=False, 
    orient="h"
)
ax.set_title("Bike Rental Data by Seasons", fontsize=30)
ax.set_xlabel(None)
ax.set_ylabel(None)
ax.tick_params(axis='y', labelsize=20)
ax.tick_params(axis='x', labelsize=15)
st.pyplot(fig)
st.divider()
st.header(":cloud: Bike Rental Data by :rainbow[Weathers]", divider='violet')

fig, ax = plt.subplots(figsize=(18, 9))

sns.barplot(
    x="weathersit",
    y="total",
    hue="total",
    data=grafik_cuaca.sort_values(by="total", ascending=False)
)

ax.set_title("Bike Rental Data by Weathers", loc="center", fontsize=30)
ax.set_ylabel(None)
ax.set_xlabel(None)
ax.tick_params(axis='x', labelsize=8)
ax.legend(title="Weather Conditions",
           labels=["1. Clear, Few/Partly Cloudy",
                  "2. Mist, Cloudy, Broken Clouds",
                  "3. Light Snow/Rain, Thunderstorm",
                  "4. Heavy Rain/Ice Pallets/Snow/Fog"
                 ]           
          )
st.pyplot(fig)

st.divider()

st.header("Data Perbandingan Data Rental", divider='blue')

fig, ax = plt.subplots(figsize=(8, 6))

ax.pie(persetanse_pie['total_rental'], labels=persetanse_pie['jenis'], autopct='%1.1f%%', colors=['#1f77b4', '#ff7f0e'], explode=(0.1, 0))
ax.set_title('Persentase antara Casual dan Registered')

st.pyplot(fig)

st.divider()

st.caption("Copyright (c) Nothingness :tm:")
