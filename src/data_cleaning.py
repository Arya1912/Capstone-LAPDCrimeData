import pandas as pd
import numpy as np
import os
import json

def generate_quality_report(df):
    report = {
        "Total Records": len(df),
        "Missing Values": df.isnull().sum().to_dict(),
        "Date Range": f"{df['DATE OCC'].min()} to {df['DATE OCC'].max()}"
    }
    return report

def clean_lapd_data(df):
    df = df[(df['LAT'] > 33.3) & (df['LAT'] < 34.8) & 
            (df['LON'] > -119.0) & (df['LON'] < -117.0)]
    df['TIME OCC'] = df['TIME OCC'].astype(str).str.zfill(4)
    df['hour'] = df['TIME OCC'].str[:2].astype(int)
    df['DATE OCC'] = pd.to_datetime(df['DATE OCC'])
    df['time_bucket'] = df['hour'].apply(lambda x: 'Morning' if 5<=x<12 else 'Afternoon' if 12<=x<17 else 'Evening' if 17<=x<21 else 'Night')
    return df

def save_processed_data(df, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)

def export_quality_report(df, report_path):
    """Saves a JSON summary of the data for your final Master's report."""
    report = {
        "Total_Records": int(len(df)),
        "Date_Range": f"{df['DATE OCC'].min()} to {df['DATE OCC'].max()}",
        "Missing_Values": df.isnull().sum().to_dict()
    }
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=4)
    print(f"Quality Report exported to: {report_path}")