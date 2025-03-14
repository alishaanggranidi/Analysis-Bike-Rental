import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import numpy as np
from sklearn.cluster import KMeans
import requests
from sklearn.preprocessing import StandardScaler

sns.set(style='dark')

def create_hourly_rentals_df(df):
    hourly_rentals_df = df.groupby("hours")["count_cr"].sum().reset_index()
    return hourly_rentals_df

def create_seasonal_rentals_df(df):
    seasonal_rentals_df = df.groupby("season")["count_cr"].sum().reset_index()
    return seasonal_rentals_df

def create_weekday_rentals_df(df):
    weekday_rentals_df = df.groupby("one_of_week")["count_cr"].sum().reset_index()
    weekday_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    weekday_rentals_df['one_of_week'] = pd.Categorical(weekday_rentals_df['one_of_week'], categories=weekday_order)
    weekday_rentals_df = weekday_rentals_df.sort_values('one_of_week')
    return weekday_rentals_df

def create_monthly_rentals_df(df):
    monthly_rentals_df = df.groupby("month")["count_cr"].sum().reset_index()
    month_order = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    monthly_rentals_df['month'] = pd.Categorical(monthly_rentals_df['month'], categories=month_order)
    monthly_rentals_df = monthly_rentals_df.sort_values('month')
    return monthly_rentals_df

def create_weather_cluster_df(df):
    weather_data = df.groupby('weather_situation').agg({
        'count_cr': ['mean', 'sum', 'std'],
        'temp': 'mean',
        'humidity': 'mean',
        'wind_speed': 'mean'
    }).reset_index()
    
    weather_data.columns = ['_'.join(col).strip('_') for col in weather_data.columns.values]
    
    scaler = StandardScaler()
    features = ['count_cr_mean', 'temp_mean', 'humidity_mean', 'wind_speed_mean']
    scaled_features = scaler.fit_transform(weather_data[features])
    
    kmeans = KMeans(n_clusters=2, random_state=42)
    weather_data['cluster'] = kmeans.fit_predict(scaled_features)
    
    return weather_data

def create_weather_situation_df(df):
    weather_unique_counts = df.groupby(by="weather_situation", observed=True)["count_cr"].nunique().sort_values(ascending=False)
    weather_unique_df = weather_unique_counts.reset_index()
    weather_unique_df.columns = ['weather_situation', 'unique_count_values']
    return weather_unique_df

def main():
    st.set_page_config(
        page_title="Bike Rental Analysis Dashboard",
        page_icon="🚲",
        layout="wide"
    )
    
    df_day = pd.read_csv("day.csv")
    df_hour = pd.read_csv("hour.csv")
    
    df_day['dteday'] = pd.to_datetime(df_day['dteday'])
    df_hour['dteday'] = pd.to_datetime(df_hour['dteday'])
    
    df_day.rename(columns={
        'yr':'year',
        'mnth':'month',
        'weekday':'one_of_week', 
        'weathersit':'weather_situation', 
        'windspeed':'wind_speed',
        'cnt':'count_cr',
        'hum':'humidity'
    }, inplace=True)
    
    df_hour.rename(columns={
        'yr':'year',
        'hr':'hours',
        'mnth':'month',
        'weekday':'one_of_week', 
        'weathersit':'weather_situation',
        'windspeed':'wind_speed',
        'cnt':'count_cr',
        'hum':'humidity'
    }, inplace=True)
    
    columns = ['season', 'month', 'holiday', 'one_of_week', 'weather_situation']
    for column in columns:
       df_day[column] = df_day[column].astype("category")
       df_hour[column] = df_hour[column].astype("category")

    df_hour['month'] = df_hour['month'].replace(
        {1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun',
         7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'}
    )
    
    df_day['month'] = df_day['month'].replace(
        {1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun',
         7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'}
    )
    
    df_day['weather_situation'] = df_day['weather_situation'].replace(
        {1: 'Clear', 2: 'Misty', 3: 'Light_rainsnow', 4: 'Heavy_rainsnow'}
    )
    
    df_hour['weather_situation'] = df_hour['weather_situation'].replace(
        {1: 'Clear', 2: 'Misty', 3: 'Light_rainsnow', 4: 'Heavy_rainsnow'}
    )
    
    df_day['one_of_week'] = df_day['one_of_week'].replace(
        {0: 'Sunday', 1: 'Monday', 2: 'Tuesday', 3: 'Wednesday',
         4: 'Thursday', 5: 'Friday', 6: 'Saturday'}
    )
    
    df_hour['one_of_week'] = df_hour['one_of_week'].replace(
        {0: 'Sunday', 1: 'Monday', 2: 'Tuesday', 3: 'Wednesday',
         4: 'Thursday', 5: 'Friday', 6: 'Saturday'}
    )
    
    df_day['season'] = df_day['season'].replace(
        {1: 'Spring', 2: 'Summer', 3: 'Fall', 4: 'Winter'}
    )
    
    df_hour['season'] = df_hour['season'].replace(
        {1: 'Spring', 2: 'Summer', 3: 'Fall', 4: 'Winter'}
    )
    
    df_day['year'] = df_day['year'].replace({0: '2011', 1: '2012'})
    df_hour['year'] = df_hour['year'].replace({0: '2011', 1: '2012'})
    
    min_date = df_day["dteday"].min().date()
    max_date = df_day["dteday"].max().date()
    
    st.title('🚲 Bike Rental Analysis Dashboard')
    
    with st.sidebar:
        file_id = "1kfgGB6DIm7jnRl_Kn-3NX508J68sbtBU"

        url = f"https://drive.google.com/uc?export=view&id={file_id}"
        response = requests.get(url)
        st.image(response.content)
        
        start_date, end_date = st.date_input(
            label='Date Range',
            min_value=min_date,
            max_value=max_date,
            value=[min_date, max_date]
        )
        
        season_options = df_day['season'].unique().tolist()
        selected_seasons = st.multiselect('Select Seasons', season_options, default=season_options)
    
    filtered_day_df = df_day[
        (df_day["dteday"].dt.date >= start_date) & 
        (df_day["dteday"].dt.date <= end_date) &
        (df_day["season"].isin(selected_seasons))
    ]
    
    filtered_hour_df = df_hour[
        (df_hour["dteday"].dt.date >= start_date) & 
        (df_hour["dteday"].dt.date <= end_date) &
        (df_hour["season"].isin(selected_seasons))
    ]
    
    hourly_rentals_df = create_hourly_rentals_df(filtered_hour_df)
    seasonal_rentals_df = create_seasonal_rentals_df(filtered_day_df)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_rentals = filtered_day_df['count_cr'].sum()
        st.metric("Total Rentals", value=f"{total_rentals:,}")
    
    with col2:
        avg_daily_rentals = round(filtered_day_df['count_cr'].mean(), 1)
        st.metric("Avg. Daily Rentals", value=f"{avg_daily_rentals:,}")
    
    with col3:
        peak_hour = hourly_rentals_df.sort_values(by='count_cr', ascending=False).iloc[0]['hours']
        st.metric("Peak Hour", value=f"{int(peak_hour)}:00")
    
    with col4:
        top_season = seasonal_rentals_df.sort_values(by='count_cr', ascending=False).iloc[0]['season']
        st.metric("Top Season", value=top_season)
    
    st.subheader("Jam berapa yang memiliki jumlah penyewaan terbanyak dan paling sedikit?")
    sum_order_items_df = hourly_rentals_df.groupby("hours")["count_cr"].sum().reset_index()

    fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(10, 25))

    most_rentals_df = sum_order_items_df.sort_values(by="count_cr", ascending=False).head(5)
    sns.barplot(
        x="hours", y="count_cr", data=most_rentals_df,
        hue="hours", palette=["#F1E7EB", "#F1E7EB", "#FFADCB", "#F1E7EB", "#F1E7EB"],
        legend=False, ax=ax[0]
    )
    ax[0].set_ylabel(None)
    ax[0].set_xlabel("Hours (PM)", fontsize=20)
    ax[0].set_title("Jam penyewaan terbanyak", loc="center", fontsize=30, pad=10)
    ax[0].tick_params(axis='y', labelsize=20)
    ax[0].tick_params(axis='x', labelsize=20)

    least_hours_rentals = sum_order_items_df[sum_order_items_df["hours"].isin([0, 1, 2, 3, 4])]
    least_rentals_df = least_hours_rentals.sort_values(by="hours", ascending=False)

    palette = ["#F1E7EB", "#F1E7EB", "#F1E7EB", "#F1E7EB", "#FFADCB"]

    sns.barplot(
        x="hours", y="count_cr", data=least_rentals_df,
        hue="hours", palette=palette,
        legend=False, ax=ax[1]
    )
    ax[1].set_ylabel(None)
    ax[1].set_xlabel("Hours (AM)", fontsize=20)
    ax[1].set_title("Jam penyewaan paling sedikit", loc="center", fontsize=30, pad=10)
    ax[1].tick_params(axis='y', labelsize=20)
    ax[1].tick_params(axis='x', labelsize=20)

    plt.tight_layout()
    st.pyplot(fig)    
    st.subheader("Musim dengan jumlah penyewaan terbanyak")

    colors = ["#F1E7EB", "#F1E7EB", "#FFADCB", "#F1E7EB"]
    fig, ax = plt.subplots(figsize=(20, 10))
    sns.barplot(
            y="count_cr", 
            x="season",
            data=seasonal_rentals_df.sort_values(by="season", ascending=False),
            palette=colors,
            ax=ax
        )
    ax.set_title("Grafik berdasarkan Musim", loc="center", fontsize=50)
    ax.set_ylabel(None)
    ax.set_xlabel(None)
    ax.tick_params(axis='x', labelsize=35)
    ax.tick_params(axis='y', labelsize=30)
    st.pyplot(fig)
    
    st.subheader("Distribusi penyewaan berdasarkan musim")

    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = ["#F1E7EB", "#F1E7EB", "#FFADCB", "#F1E7EB"]
    explode = (0, 0, 0.1, 0)
    
    wedges, texts, autotexts = ax.pie(
        seasonal_rentals_df['count_cr'],
        labels=seasonal_rentals_df['season'],
        autopct='%1.1f%%',
        colors=colors,
        explode=explode,
        startangle=90,
        textprops={'fontsize': 12}
    )
    
    ax.set_title('Distribusi penyewaan berdasarkan musim', fontsize=15)
    ax.axis('equal')
    st.pyplot(fig)

    st.subheader("Analisis Situasi Cuaca")

    weather_unique_counts = create_weather_situation_df(filtered_hour_df)
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("### Nilai Unik berdasarkan Situasi Cuaca")
        st.dataframe(
            weather_unique_counts,
            use_container_width=True,
            column_config={
                "weather_situation": "Situasi Cuaca",
                "unique_count_values": "Jumlah Nilai Unik"
            }
        )
    
    with col2:
        fig, ax = plt.subplots(figsize=(10, 6))
        weather_colors = ["#FFADCB", "#F1E7EB", "#F1E7EB", "#F1E7EB"]
        
        sns.barplot(
            x="weather_situation", 
            y="unique_count_values", 
            data=weather_unique_counts,
            palette=weather_colors,
            ax=ax
        )
        
        ax.set_title("Distribusi Nilai Unik berdasarkan Situasi Cuaca", fontsize=15)
        ax.set_xlabel("Situasi Cuaca")
        ax.set_ylabel("Jumlah Nilai Unik")
        plt.xticks(rotation=45)
        
        for i, v in enumerate(weather_unique_counts["unique_count_values"]):
            ax.text(i, v + 5, str(v), ha='center')
            
        st.pyplot(fig)
    
    st.subheader("Korelasi antara Faktor Cuaca dan Penyewaan")

    numeric_cols = ['temp', 'humidity', 'wind_speed', 'count_cr']
    corr_df = filtered_hour_df[numeric_cols].corr()
    
    fig, ax = plt.subplots(figsize=(8, 6))
    mask = np.triu(np.ones_like(corr_df, dtype=bool))
    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    
    sns.heatmap(
        corr_df, 
        mask=mask, 
        cmap=cmap, 
        vmax=1, 
        vmin=-1, 
        center=0,
        annot=True, 
        fmt=".2f",
        square=True, 
        linewidths=.5, 
        cbar_kws={"shrink": .8}
    )
    
    plt.title("Korelasi antara Faktor Cuaca dan Penyewaan", fontsize=15)
    st.pyplot(fig)

    st.caption('Bike Rental Analysis Dashboard © 2025')

if __name__ == '__main__':
    main()