import numpy as np
import pandas as pd

import geopandas as gpd
import pyproj

import folium
from folium import Map

from branca.element import Template, MacroElement

from shapely.geometry import Polygon


def read_land_use_df(data_path, crs2convert='epsg:4326'):
    landuse_path = data_path + '/SAD69-96_SHP_tpclusopredominante/SAD69-96_SHP_tpclusopredominante.shp'

    landuse_df = gpd.read_file(landuse_path)
    landuse_df.set_crs(epsg=5533, inplace=True)

    if crs2convert is not None:
        landuse_df.to_crs(crs2convert, inplace=True)

    return landuse_df

def bearing_from_path(df, path_col='img_path'):
    bearing = df[path_col].transform(lambda x: x[x.rfind('_') + 1:x.rfind('.')])
    return bearing.transform(lambda x: float(x) if x != 'nan' else 0)

def geo_buffer(df, buffer_size, metric_crs=3857):
    geo_crs = df.crs
    buffer_df = df.copy().to_crs(crs=metric_crs) 
    buffer_df['geometry'] = buffer_df['geometry'].buffer(buffer_size)
    return buffer_df.to_crs(crs=geo_crs) 

def geo_centroid(df, metric_crs=3857):
    geo_crs = df.crs
    return df.to_crs(crs=metric_crs).centroid.to_crs(crs=geo_crs)

def iter_buffer_inter(df2buffer, df, start_buffer_size=10, idx_col='u_idx'):
    all_idxs = set(list(df2buffer[idx_col]))
    
    
    buffered_df = geo_buffer(df2buffer, start_buffer_size)
    inter_df = gpd.overlay(buffered_df, df, how='intersection')
    inter_dfs = [inter_df]
    nr_no_inter = len(all_idxs - set(inter_df[idx_col]))

    i = 1
    while nr_no_inter > 0:
        print(i, nr_no_inter, df2buffer.shape[0])
        buffered_df = buffered_df[~buffered_df[idx_col].isin(inter_df[idx_col])]
        buffered_df = geo_buffer(buffered_df, start_buffer_size + i*10)
        inter_df = gpd.overlay(buffered_df, df, how='intersection')
        nr_no_inter -= inter_df[idx_col].unique().shape[0]
        if inter_df.shape[0] != 0:
            inter_dfs += [inter_df]
        i += 1

    inter_df = gpd.GeoDataFrame(pd.concat(inter_dfs, ignore_index=True), crs=inter_dfs[0].crs)
 
    inter_df['bearing'] = bearing_from_path(inter_df)
    inter_df['centroid_point'] = geo_centroid(inter_df)

    geodesic = pyproj.Geod(ellps='WGS84')

    inter_df['other_bearing'] = inter_df.apply(lambda row: geodesic.inv(row.img_lon, row.img_lat, 
                                                                        row.centroid_point.x, row.centroid_point.y)[0], 
                                               axis=1)
    inter_df.other_bearing = inter_df.other_bearing.transform(lambda x: (360 + x) % 360)

    inter_df['bearing_diff'] = ((360 + (inter_df.bearing - inter_df.other_bearing) + 180) % 360 - 180).abs()

    inter_df = inter_df.loc[inter_df.groupby(idx_col).bearing_diff.idxmin()].reset_index(drop=True)
    
    return inter_df

def land_type_id_to_name(series):
    landuse_type = {0: "Sem informação", 1: "Residencial horizontal baixo padrão", 2: "Residencial horizontal médio/alto padrão",
                    3: "Residencial vertical baixo padrão", 4: "Residencial vertical médio/alto padrão", 5: "Comércio e serviços",
                    6: "Indústria e armazéns", 7: "Residencial e Comércio/serviços", 8: "Residencial e Indústria/armazéns",
                    9: "Comércio/serviços e Indústria/armazéns", 10: "Garagens", 11: "Equipamentos públicos",
                    12: "Escolas", 13: "Terrenos vagos", 14: "Outros", 15: "Sem predominâncias"} # encontrado na planilha de metadados do geosampa


    return series.transform(lambda x: landuse_type[int(x)] if np.isfinite(x) and int(x) in landuse_type else landuse_type[0])


def group_land_types(series):
    residential_group = ['Residencial horizontal médio/alto padrão', 
                         'Residencial vertical médio/alto padrão', 
                         'Residencial vertical baixo padrão', 
                         'Residencial horizontal baixo padrão']

    commercial_group = ['Comércio e serviços', 'Residencial e Comércio/serviços', 
                        'Comércio/serviços e Indústria/armazéns']

    industry_group = ['Indústria e armazéns',]

    school_group = ['Escolas']

    others_group = ['Sem predominâncias', 'Equipamentos públicos', 'Sem informação', 'Escolas', 
                    'Terrenos vagos', 'Outros', 'Garagens']

    def group_func(land_type):
        if land_type in residential_group:
            return 'residencial'
        elif land_type in commercial_group:
            return 'comercial'
        elif land_type in industry_group:
            return 'industrial'
        elif land_type in school_group:
            return 'escolar'
        return 'outros'

    return series.transform(group_func)


def text_df_to_gdf(df, cols2include=('img_lat', 'img_lon', 'img_path'), crs='epsg:4326'):
    df['u_idx'] = [i for i in range(df.shape[0])]
    det_rec_geo_df = df[['u_idx'] + list(cols2include)].copy()
    det_rec_geo_df = gpd.GeoDataFrame(det_rec_geo_df, geometry=gpd.points_from_xy(det_rec_geo_df.img_lon, det_rec_geo_df.img_lat))
    det_rec_geo_df.set_crs(crs, inplace=True)

    return det_rec_geo_df


def include_landuse_in_df(df, data_path, cols2include=('img_lat', 'img_lon', 'img_path')):
    landuse_df = read_land_use_df(data_path)

    det_rec_geo_df = text_df_to_gdf(df, cols2include=cols2include)

    inter_df = iter_buffer_inter(det_rec_geo_df, landuse_df, start_buffer_size=10, idx_col='u_idx')

    inter_df['land_type'] = land_type_id_to_name(inter_df.tp_pred_15)

    idx2land_type = pd.Series(inter_df.land_type.values, index=inter_df.u_idx).to_dict()

    df['land_type'] = df.u_idx.transform(lambda x: idx2land_type[x])
    df['land_type_grouped'] = group_land_types(df.land_type)

    df.drop(['u_idx'], axis=1, inplace=True)

    return df


def restrict_by_bounds(df, minx, miny, maxx, maxy):
    border = [(minx, maxy), (minx, miny), (maxx, miny), (maxx, maxy)]
    rect = gpd.GeoDataFrame({'geometry':[Polygon(shell=border)]}, crs=df.crs)

    return gpd.overlay(df, rect, how='intersection')


def land_type_map_vis(data_path, foilum_map=None, df2restrict=None, opacity=1.0):
    landuse_df = read_land_use_df(data_path)
    landuse_df['land_type'] = land_type_id_to_name(landuse_df.tp_pred_15)
    landuse_df['land_type_grouped'] = group_land_types(landuse_df.land_type)

    landtype_eng = {'residencial': 'residential', 'comercial':'comercial', 'escolar':'school', 
                'outros':'others', 'industrial':'industrial'}
    
    landuse_df["land_type_grouped"] = landuse_df.land_type_grouped.transform(lambda x: landtype_eng[x])

    fill_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', 
                   '#bcbd22', '#17becf']
    fill_colors = {land_type:color for land_type, color in zip(landuse_df.land_type_grouped.unique(), fill_colors)}
    landuse_df['color'] = landuse_df.land_type_grouped.transform(lambda x: fill_colors[x])

    if foilum_map is None:
        centroids = landuse_df.geometry.centroid
        mid = (centroids.y.mean(), centroids.x.mean())

        foilum_map = Map(location=mid, zoom_start=11)

    if df2restrict is not None:
        minx, miny = df2restrict.img_lon.min(), df2restrict.img_lat.min()
        maxx, maxy = df2restrict.img_lon.max(), df2restrict.img_lat.max()

        landuse_df = restrict_by_bounds(landuse_df, minx, miny, maxx, maxy)
    
    landuse_df['geometry'] = gpd.GeoSeries(landuse_df['geometry']).simplify(tolerance=0.0001)
    
    geo_j = folium.GeoJson(data=landuse_df.to_json(),
                           style_function=lambda x: {'fillColor': x['properties']['color'], 
                                                     'color': x['properties']['color'], 'fillOpacity': opacity, 
                                                     'opacity': opacity})

    geo_j.add_to(foilum_map)

    return add_legend(foilum_map, fill_colors)


def add_legend(folium_map, landtype2color):

    template_start = """
    {% macro html(this, kwargs) %}

    <!doctype html>
    <html lang="en">
    <head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>jQuery UI Draggable - Default functionality</title>
    <link rel="stylesheet" href="//code.jquery.com/ui/1.12.1/themes/base/jquery-ui.css">

    <script src="https://code.jquery.com/jquery-1.12.4.js"></script>
    <script src="https://code.jquery.com/ui/1.12.1/jquery-ui.js"></script>
    
    <script>
    $( function() {
        $( "#maplegend" ).draggable({
                        start: function (event, ui) {
                            $(this).css({
                                right: "auto",
                                top: "auto",
                                bottom: "auto"
                            });
                        }
                    });
    });

    </script>
    </head>
    <body>

    
    <div id='maplegend' class='maplegend' 
        style='position: absolute; z-index:9999; border:2px solid grey; background-color:rgba(255, 255, 255, 0.8);
        border-radius:6px; padding: 10px; font-size:14px; right: 20px; bottom: 20px;'>
        
    <div class='legend-title'>Legend</div>
    <div class='legend-scale'>
    <ul class='legend-labels'>"""
    
    legend_items = []
    for land_type, color in landtype2color.items():
        legend_items += [f"<li><span style='background:{color};opacity:0.7;'></span>{land_type}</li>\n"]
        
    template_end = """

    </ul>
    </div>
    </div>
    
    </body>
    </html>

    <style type='text/css'>
    .maplegend .legend-title {
        text-align: left;
        margin-bottom: 5px;
        font-weight: bold;
        font-size: 90%;
        }
    .maplegend .legend-scale ul {
        margin: 0;
        margin-bottom: 5px;
        padding: 0;
        float: left;
        list-style: none;
        }
    .maplegend .legend-scale ul li {
        font-size: 80%;
        list-style: none;
        margin-left: 0;
        line-height: 18px;
        margin-bottom: 2px;
        }
    .maplegend ul.legend-labels li span {
        display: block;
        float: left;
        height: 16px;
        width: 30px;
        margin-right: 5px;
        margin-left: 0;
        border: 1px solid #999;
        }
    .maplegend .legend-source {
        font-size: 80%;
        color: #777;
        clear: both;
        }
    .maplegend a {
        color: #777;
        }
    </style>
    {% endmacro %}"""

    macro = MacroElement()
    macro._template = Template(template_start + "".join(legend_items) + template_end)

    folium_map.get_root().add_child(macro)

    return folium_map
