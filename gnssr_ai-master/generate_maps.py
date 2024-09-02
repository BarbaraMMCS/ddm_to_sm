import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import seaborn as sns
import os


def plot_q_maps_rf(year, q, starts_date, ends_date, model, folder):
    precision = 9000
    df_1 = pd.read_csv(f'../data/predictions/{model}/{model}_{starts_date}_{ends_date}.csv').eval(
        'abs_diff = abs(soil_moisture - pred_soil_moisture)')
    df_1['pred_soil_moisture_filtered'] = np.where(df_1.soil_moisture.isna(), np.nan, df_1.pred_soil_moisture)
    gdf_1 = (
        gpd.GeoDataFrame(
            df_1, geometry=gpd.points_from_xy(df_1.sp_lon, df_1.sp_lat)
        )
        .set_crs(4326)
        .to_crs(6933)  # Projection
    )
    gdf_1['x_round'] = (gdf_1.geometry.x / precision).round() * precision
    gdf_1['y_round'] = (gdf_1.geometry.y / precision).round() * precision

    gdf_grouped = (
        gdf_1
        .groupby(['x_round', 'y_round'])
        .agg(mean_target=('soil_moisture', 'mean'), mean_pred=('pred_soil_moisture_filtered', 'mean'),
             mean_pred_all=('pred_soil_moisture', 'mean'),
             mean_abs_diff=('abs_diff', 'mean'), mean_cluster=('7', 'mean'),
             mean_vegetation_water_content=('vegetation_water_content', 'mean'),
             mean_reflectivity=('reflectivity', 'mean'), mean_sp_inc_angle=('sp_inc_angle', 'mean'),
             mean_peak_power=('peak_power', 'mean'))
        .reset_index()
    )

    gdf_grouped['x'] = gdf_grouped.x_round / precision
    gdf_grouped['y'] = -gdf_grouped.y_round / precision

    heatmap_data = pd.pivot(gdf_grouped.filter(['y', 'x', 'mean_target']), columns='x', index='y')
    ax = sns.heatmap(heatmap_data, cmap='Spectral', vmax=0.4, vmin=0)
    ax.set_title(f'Q{q} - {year} - Mean Soil Moisture')
    ax.set(xlabel=None)
    ax.set(ylabel=None)
    ax.set(xticklabels=[])
    ax.set(yticklabels=[])
    plt.axis('off')
    plt.savefig(f'../figures/{folder}/maps/mean_soil_moisture_{year}Q{q}.svg')
    plt.close()

    heatmap_data = pd.pivot(gdf_grouped.filter(['y', 'x', 'mean_pred']), columns='x', index='y')
    ax = sns.heatmap(heatmap_data, cmap='Spectral', vmax=0.4, vmin=0)
    ax.set_title(f'Q{q} - {year} - Mean Predicted Soil Moisture')
    ax.set(xlabel=None)
    ax.set(ylabel=None)
    ax.set(xticklabels=[])
    ax.set(yticklabels=[])
    plt.axis('off')
    plt.savefig(f'../figures/{folder}/maps/mean_pred_soil_moisture_{year}Q{q}.svg')
    plt.close()

    heatmap_data = pd.pivot(gdf_grouped.filter(['y', 'x', 'mean_pred_all']), columns='x', index='y')
    ax = sns.heatmap(heatmap_data, cmap='Spectral', vmax=0.4, vmin=0)
    ax.set_title(f'Q{q} - {year} - Mean Predicted Soil Moisture All')
    ax.set(xlabel=None)
    ax.set(ylabel=None)
    ax.set(xticklabels=[])
    ax.set(yticklabels=[])
    plt.axis('off')
    plt.savefig(f'../figures/{folder}/maps/mean_pred_soil_moisture_all_{year}Q{q}.svg')
    plt.close()

    heatmap_data = pd.pivot(gdf_grouped.filter(['y', 'x', 'mean_abs_diff']), columns='x', index='y')
    ax = sns.heatmap(heatmap_data, cmap='Reds')
    ax.set_title(f'Q{q} - {year} - Mean abs diff')
    ax.set(xlabel=None)
    ax.set(ylabel=None)
    ax.set(xticklabels=[])
    ax.set(yticklabels=[])
    plt.axis('off')
    plt.savefig(f'../figures/{folder}/maps/mean_abs_diff_{year}Q{q}.svg')
    plt.close()

    heatmap_data = pd.pivot(gdf_grouped.filter(['y', 'x', 'mean_cluster']), columns='x', index='y')
    ax = sns.heatmap(heatmap_data, cmap='Reds')
    ax.set_title(f'Q{q} - {year} - Mean Cluster')
    ax.set(xlabel=None)
    ax.set(ylabel=None)
    ax.set(xticklabels=[])
    ax.set(yticklabels=[])
    plt.axis('off')
    plt.savefig(f'../figures/{folder}/maps/mean_cluster_{year}Q{q}.svg')
    plt.close()

    heatmap_data = pd.pivot(gdf_grouped.filter(['y', 'x', 'mean_vegetation_water_content']), columns='x', index='y')
    ax = sns.heatmap(heatmap_data, cmap='Spectral')
    ax.set_title(f'Q{q} - {year} - Mean Vegetation Water Content')
    ax.set(xlabel=None)
    ax.set(ylabel=None)
    ax.set(xticklabels=[])
    ax.set(yticklabels=[])
    plt.axis('off')
    plt.savefig(f'../figures/{folder}/maps/mean_vegetation_water_content_{year}Q{q}.svg')
    plt.close()

    heatmap_data = pd.pivot(gdf_grouped.filter(['y', 'x', 'mean_reflectivity']), columns='x', index='y')
    ax = sns.heatmap(heatmap_data, cmap='Reds')
    ax.set_title(f'Q{q} - {year} - Mean Reflectivity')
    ax.set(xlabel=None)
    ax.set(ylabel=None)
    ax.set(xticklabels=[])
    ax.set(yticklabels=[])
    plt.axis('off')
    plt.savefig(f'../figures/{folder}/maps/mean_reflectivity_{year}Q{q}.svg')
    plt.close()

    heatmap_data = pd.pivot(gdf_grouped.filter(['y', 'x', 'mean_peak_power']), columns='x', index='y')
    ax = sns.heatmap(heatmap_data, cmap='Spectral')
    ax.set_title(f'Q{q} - {year} - Mean Peak Power')
    ax.set(xlabel=None)
    ax.set(ylabel=None)
    ax.set(xticklabels=[])
    ax.set(yticklabels=[])
    plt.axis('off')
    plt.savefig(f'../figures/{folder}/maps/mean_peak_power_{year}Q{q}.svg')
    plt.close()
    

def plot_q_maps_xgb(year, q, starts_date, ends_date, model, folder):
    precision = 9000
    df_1 = pd.read_csv(f'../data/predictions/{model}/{model}_{starts_date}_{ends_date}.csv').eval(
        'abs_diff = abs(soil_moisture - pred_soil_moisture)')
    df_1['pred_soil_moisture_filtered'] = np.where(df_1.soil_moisture.isna(), np.nan, df_1.pred_soil_moisture)
    gdf_1 = (
        gpd.GeoDataFrame(
            df_1, geometry=gpd.points_from_xy(df_1.lon, df_1.lat)
        )
        .set_crs(4326)
        .to_crs(6933)  # Projection
    )
    gdf_1['x_round'] = (gdf_1.geometry.x / precision).round() * precision
    gdf_1['y_round'] = (gdf_1.geometry.y / precision).round() * precision

    gdf_grouped = (
        gdf_1
        .groupby(['x_round', 'y_round'])
        .agg(mean_pred=('pred_soil_moisture_filtered', 'mean'),
             mean_pred_all=('pred_soil_moisture', 'mean'),
             mean_abs_diff=('abs_diff', 'mean'))
        .reset_index()
    )

    gdf_grouped['x'] = gdf_grouped.x_round / precision
    gdf_grouped['y'] = -gdf_grouped.y_round / precision

    heatmap_data = pd.pivot(gdf_grouped.filter(['y', 'x', 'mean_pred']), columns='x', index='y')
    ax = sns.heatmap(heatmap_data, cmap='Spectral', vmax=0.4, vmin=0)
    ax.set_title(f'Q{q} - {year} - Mean Predicted Soil Moisture')
    ax.set(xlabel=None)
    ax.set(ylabel=None)
    ax.set(xticklabels=[])
    ax.set(yticklabels=[])
    plt.axis('off')
    plt.savefig(f'../figures/{folder}/maps/mean_pred_soil_moisture_{year}Q{q}.svg')
    plt.close()

    heatmap_data = pd.pivot(gdf_grouped.filter(['y', 'x', 'mean_pred_all']), columns='x', index='y')
    ax = sns.heatmap(heatmap_data, cmap='Spectral', vmax=0.4, vmin=0)
    ax.set_title(f'Q{q} - {year} - Mean Predicted Soil Moisture All')
    ax.set(xlabel=None)
    ax.set(ylabel=None)
    ax.set(xticklabels=[])
    ax.set(yticklabels=[])
    plt.axis('off')
    plt.savefig(f'../figures/{folder}/maps/mean_pred_soil_moisture_all_{year}Q{q}.svg')
    plt.close()

    heatmap_data = pd.pivot(gdf_grouped.filter(['y', 'x', 'mean_abs_diff']), columns='x', index='y')
    ax = sns.heatmap(heatmap_data, cmap='Reds')
    ax.set_title(f'Q{q} - {year} - Mean abs diff')
    ax.set(xlabel=None)
    ax.set(ylabel=None)
    ax.set(xticklabels=[])
    ax.set(yticklabels=[])
    plt.axis('off')
    plt.savefig(f'../figures/{folder}/maps/mean_abs_diff_{year}Q{q}.svg')
    plt.close()


if __name__ == '__main__':
    year_q_start_end =  first_last_days = [
        ('2022', '1', '20220101', '20220331'),
        ('2022', '2', '20220401', '20220630'),
        ('2022', '3', '20220701', '20220930'),
        ('2022', '4', '20221001', '20221231'),
        ('2023', '1', '20230101', '20230331'),
        ('2023', '2', '20230401', '20230630'),
        ('2023', '3', '20230701', '20230930'),
        ('2023', '4', '20231001', '20231231')
    ]

    # for params in year_q_start_end:
    #     plot_q_maps_rf(*params, 'rf', 'random_forest')

    for params in year_q_start_end:
        plot_q_maps_xgb(*params, 'xgb', 'xgb')
    
    