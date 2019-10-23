"""
This module provides various functions for assessing and calculating expsoures to traffic noise.
The functions are useful in calculating noise costs for quiet path route optimization and in comparing exposures to noise
between paths.

"""

from typing import List, Set, Dict, Tuple
import ast
import pandas as pd
import geopandas as gpd
import numpy
import rasterio as rio
from rasterstats import zonal_stats, point_query
from shapely.geometry import LineString
import utils.geometry as geom_utils


def get_aqi_costs() -> Dict[int, float]:
    """Returns a set of AQI-specific air quality cost coefficients. They can be used in calculating the base air quality
     cost for edges. (Alternative air quality costs can be calculated by multiplying the base noise cost with different
     air quality sensitivity from get_aq_sensitivity())

    Returns:
        A dictionary of air quality cost coefficients where the keys are the air quality index classes.
    """
    return {1: 0.1, 2: 0.2, 3: 0.3, 4: 0.4, 5: 0.5}


def get_aq_tolerances() -> List[float]:
    """Returns a set of air quality tolerance coefficients that can be used in adding alternative air quality -based costs
    to edges and subsequently calculating alternative green paths (using different weights for air quality cost in routing).

    Returns:
        A list of air quality tolerance values.
    """
    return [0.1, 0.15, 0.25, 0.5, 1, 1.5, 2, 4, 6, 10, 20, 40]


def add_aq_to_split_lines(aq_polygons: gpd.GeoDataFrame, split_lines: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Performs a spatial join of air quality values (from air quality raster/polygon) to LineString objects based on
    spatial intersection of the center points of the lines and the air quality raster/polygons.

    Note:
        The polygons in the aq_polygons GeoDataFrame should not overlap. However, they sometimes do and hence
        the result of the spatial join needs to be filtered from duplicate noise values for some lines. In removing
        the duplicate noise values, higher noise values are retained.

    Returns:
        The result of the spatial join as a GeoDataFrame with the added columns from air quality raster/polygons.
    """
    split_lines['split_line_index'] = split_lines.index
    split_lines['geom_line'] = split_lines['geometry']
    split_lines['geom_point'] = [geom_utils.get_line_middle_point(geom) for geom in split_lines['geometry']]
    split_lines['geometry'] = split_lines['geom_point']
    line_aq = gpd.sjoin(split_lines, aq_polygons, how='left', op='intersects')
    line_aq['geometry'] = line_aq['geom_line']
    if (len(line_aq.index) > len(split_lines.index)):
        line_aq = line_aq.sort_values('db_lo', ascending=False)
        line_aq = line_aq.drop_duplicates(subset=['split_line_index'], keep='first')
    return line_aq[['geometry', 'length', 'db_lo', 'db_hi', 'index_right']]


def get_aq_exposure_lines(line_geom: LineString, aq_polys: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """TODO: check if this is needed anymore.
    """
    split_lines = geom_utils.get_split_lines_gdf(line_geom, aq_polys)
    if (split_lines.empty):
        return gpd.GeoDataFrame()
    line_aq = add_aq_to_split_lines(noise_polys, split_lines)
    line_aq = line_aq.fillna({'db_lo': 40})
    len_error = abs(line_geom.length - line_aq['length'].sum())
    if (len_error > 0.1):
        print('len_error:', len_error)
        print(' orig geom len:', line_geom.length)
        print(' line noises sum len:', line_aq['length'].sum())
    return line_aq


def get_exposures(line_aq: gpd.GeoDataFrame) -> Dict[int, float]:
    """Aggregates exposures (contaminated distances) to different air quality levels to a dictionary.
    """
    if (len(line_aq.index) == 0):
        return {}
    noise_dict = {}
    noise_groups = line_aq.groupby('db_lo')
    for key, values in noise_aq:
        tot_len = round(values['length'].sum(), 3)
        aq_dict[int(key)] = tot_len
    return aq_dict


def get_th_exposures(aq_dict: dict, ths: List[int]) -> Dict[int, float]:
    """Aggregates exposures to traffic noise levels exceeding the traffic noise levels specified in [ths].
    TODO: Check whether this is needed for AQI
    """
    th_count = len(ths)
    th_lens = [0] * len(ths)
    for th in aq_dict.keys():
        th_len = aq_dict[th]
        for idx in range(th_count):
            if (th >= ths[idx]):
                th_lens[idx] = th_lens[idx] + th_len
    th_aq_dict = {}
    for idx in range(th_count):
        th_aq_dict[ths[idx]] = round(th_lens[idx], 3)
    return th_aq_dict


def get_aq_range(aqi: float) -> int:
    """Returns the lower limit of one of the six pre-defined AQI ranges based on FMI's Air Quality Index.
    TODO: Check the threshold values
    """
    if dB >= 5.0:
        return 5
    elif dB >= 4.0:
        return 4
    elif dB >= 3.0:
        return 3
    elif dB >= 2.0:
        return 2
    elif dB >= 1.0:
        return 1
    else:
        return 1


def get_aq_range_pcts(airquality: dict, total_length: float) -> Dict[int, float]:
    """Calculates percentages of aggregated exposures to different air quality levels of total length.

    Note:
        Air quality levels exceeding 5 are aggregated and as well as levels lower than 2.
    Returns:
        A dictionary containing air quality level values with respective percentages.
        (e.g. { 1: 35, 2: 65 })
    """
    # interpolate AQI 1 distance
    aq_dists = dict(airquality)
    aq_1_len = round(total_length - get_total_aq_len(aq_dists), 2)
    if aq_1_len > 0:
        aq_dists[1] = aq_1_len

    # aggregate air quality exposures to pre-defined dB-ranges
    aq_range_lens = {}
    for aq in aq_dists.keys():
        aq_range = get_aq_range(aq)
        if (aq_range in aq_range_lens.keys()):
            aq_range_lens[aq_range] += aq_dists[aq]
        else:
            aq_range_lens[aq_range] = aq_dists[aq]

    # calculate ratio (%) of each range's length to total length
    range_pcts = {}
    for aq_range in aq_range_lens.keys():
        range_pcts[aq_range] = round(aq_range_lens[aq_range] * 100 / total_length, 1)

    return range_pcts


def get_noise_attrs_to_split_lines(gdf: gpd.GeoDataFrame, aq_raster: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Performs a spatial join of aq values (from aq raster) to LineString objects based on a
    spatial intersection of the center points of the lines [gdf] and the aq raster.
    TODO: Check if raster sampling actually works

    Note:
        Unlike the equivalent function for noisy paths, this function eats a raster file with the split lines gdf.

    Returns:
        The result of sampled air quality raster values in a new column of the geodataframe.
    """
    gdf['split_line_index'] = gdf.index
    gdf['geometry'] = gdf['mid_point']
    # Get mid point coordinates for each line
    coords = numpy.vstack((gdf.mid_point.x, gdf.mid_point.y))
    # Retrieve sample per mid point coordinate
    aq_samples = aq_raster.sample(coords)
    # Join sampled air quality values to split lines
    split_line_aq = gdf.copy()
    split_line_aq['air_quality'] = aq_samples
    # drop duplicate lines if there are any
    if (len(split_line_aq.index) > len(gdf.index)):
        split_line_aq = split_line_aq.sort_values('air_quality', ascending=False)
        split_line_aq = split_line_aq.drop_duplicates(subset=['split_line_index'], keep='first')
    return split_line_aq


def aggregate_line_aq(split_line_aq: gpd.GeoDataFrame, uniq_id: str) -> pd.DataFrame:
    """Aggregates air quality exposures (contaminated distances) from lines' air quality exposures by unique id.
    """
    row_accumulator = []
    grouped = split_line_aq.groupby(uniq_id)
    for key, values in grouped:
        row_d = {uniq_id: key}
        row_d['noises'] = get_exposures(values)
        row_accumulator.append(row_d)
    return pd.DataFrame(row_accumulator)


def aggregate_exposures(exp_list: List[dict]) -> Dict[int, float]:
    """Aggregates noise exposures (contaminated distances) from a list of noise exposures.
    """
    exps = {}
    for exp_d_value in exp_list:
        exp_d = ast.literal_eval(exp_d_value) if type(exp_d_value) == str else exp_d_value
        for aqi in exp_d.keys():
            if aqi in exps.keys():
                exps[aqi] += exp_d[aqi]
            else:
                exps[aqi] = exp_d[aqi]
    for aqi in exps.keys():
        exps[aqi] = round(exps[aqi], 2)
    return exps


def get_aq_diff(s_aq: Dict[int, float], q_aq: Dict[int, float], full_aq_range=True) -> Dict[int, float]:
    """Calculates the differences in exposures (contaminated distances) to different air quality levels between two
    exposure dictionaries.
    """
    aqi = [1, 2, 3, 4, 5]
    diff_dict = {}
    for aq in aqi:
        if (full_aq_range == False):
            if ((aq not in s_aq.keys()) and (aq not in q_aq.keys())):
                continue
        s_aq = s_aq[aq] if aq in s_aq.keys() else 0
        q_aq = q_aq[aq] if aq in q_aq.keys() else 0
        aq_diff = q_aq - s_aq
        diff_dict[aq] = round(aq_diff, 2)
    return diff_dict


def get_total_aq_len(air_quality: Dict[int, float]) -> float:
    """Returns a total length of exposures to all noise levels.
    """
    totlen = 0
    for key in air_quality.keys():
        totlen += air_quality[key]
    return round(totlen, 3)


def get_mean_aq_level(air_quality: dict, length: float) -> float:
    """Returns a mean air quality level based on noise exposures weighted by the contaminated distances to different air
     quality levels.
    """
    sum_aqi = 0
    # estimate mean dB of 5 dB range to be min dB + 2.5 dB
    for aqi in air_quality.keys():
        sum_aqi += (int(aqi)) * air_quality[aqi]
    # extrapolate noise level range 40-45dB (42.5dB) for all noises less than lowest noise range in the noise data 45-50dB
    sum_aqi_len = get_total_aq_len(air_quality)
    aqlen = length - sum_aqi_len
    sum_aqi += 42.5 * aqlen
    mean_aqi = sum_aqi / length
    return round(mean_aqi, 1)


def get_aq_cost(aqi: Dict[int, float] = {}, aq_costs: Dict[int, float] = {}, nt: float = 1) -> float:
    """Returns a total air quality cost based on contaminated distances to different air quality levels, aq_costs and aq tolerance.
    """
    aq_cost = 0
    for aq in aqi:
        if (aq in aq_costs):
            aq_cost += aqi[aq] * aq_costs[aq] * nt
    return round(aq_cost, 2)


def interpolate_link_aq(link_geom: LineString, edge_geom: LineString, edge_aq: Dict[int, float]) -> Dict[
    int, float]:
    """Interpolates air quality exposures for a split edge by multiplying each contaminated distance with a proportion
    between the edge length to the length of the original edge.
    """
    link_air_qualities = {}
    link_len_ratio = link_geom.length / edge_geom.length
    for aq in edge_aq.keys():
        link_air_qualities[aq] = round(edge_aq[aq] * link_len_ratio, 3)
    return link_air_qualities


def get_link_edge_aq_cost_estimates(nts, aq_costs, edge_dict=None, link_geom=None) -> dict:
    """Estimates air quality exposures and air quality costs for a split edge based on exposures of the original edge
    (from which the edge was split).
    """
    cost_attrs = {}
    # estimate link air qualities based on link length - edge length -ratio and edge air qualities
    cost_attrs['air_quality'] = interpolate_link_aq(link_geom, edge_dict['geometry'], edge_dict['air_quality'])
    # calculate noise tolerance specific noise costs
    for nt in nts:
        aq_cost = get_aq_cost(aqi=cost_attrs['air_quality'], aq_costs=aq_costs, nt=nt)
        cost_attrs['aqc_' + str(nt)] = round(aq_cost + link_geom.length, 2)
    aq_sum_len = get_total_aq_len(cost_attrs['air_quality'])
    if ((aq_sum_len - link_geom.length) > 0.1):
        print('link lengths do not match:', aq_sum_len, link_geom.length)
    return cost_attrs


def compare_lens_aq_lens(edge_gdf) -> gpd.GeoDataFrame:
    """Adds new columns to a GeoDataFrame of edges so that the aggregated contaminated distances can be validated against edge lengths.
    """
    gdf = edge_gdf.copy()
    gdf['uvkey_str'] = [str(uvkey[0]) + '_' + str(uvkey[1]) for uvkey in gdf['uvkey']]
    gdf['node_from'] = [uvkey[0] for uvkey in gdf['uvkey']]
    gdf['length'] = [geom.length for geom in gdf['geometry']]
    gdf['len_from_aq'] = [get_total_noises_len(noises) for noises in gdf['air_quality']]
    gdf['len_aq_error'] = gdf.apply(lambda row: row['length'] - row['len_from_aq'], axis=1)
    return gdf
