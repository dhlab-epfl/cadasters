# -*- coding: utf-8 -*-
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output

import geopandas as gpd
import json
import numpy as np
from glob import glob

app = dash.Dash()

data_files = glob('./filt/*filt*.geojson')

data_list = list()
raw_data_list = list()
for file in data_files:
    with open(file, 'r') as f:
        raw_data_list.append(json.load(f))
    # print('Nb points: {}'.format(sum([len(f['geometry']['coordinates'][0]) for f in raw_data_list[-1]['features']])))

    data_list.append(gpd.read_file(file))


data_graph = [
    dict(
        type='scattermapbox',
        lon=data.centroid.x,
        lat=data.centroid.y,
        text=["{}<br>{}".format(s, t) for s, t in zip(data.score, data.best_transcription)],
        customdata=data.best_transcription,
        hoverinfo='text',
        marker=dict(
            size=4,
            opacity=0.6,
            color=list('rgb({},{},{})'.format(i, i, i) for i in range(len(data)))
        )
    )
    for data in data_list
]

layers = [
    {
        "below": "water",
        "color": "rgb({}, {}, {})".format(*(255*np.random.random(3))),
        "opacity": 0.7,
        "sourcetype": "geojson",
        "type": "fill",
        "source": raw_data,
    }
    for raw_data in raw_data_list
]


app.layout = html.Div(children=[
    html.H1(children='Hello Dash'),
    html.Div(id='my-div'),
    html.Div(id='my-div-2'),


    html.Div(children='''
        Dash: A web application framework for Python.
    '''),

    dcc.Graph(
        id='example-graph',
        figure={
            'data': data_graph,
            'layout': {
                "autosize": True,
                "font": {"family": "Balto"},
                "height": 800,
                # "width": 1200,
                "hovermode": "closest",
                "mapbox": {
                    # "accesstoken": "pk.eyJ1IjoiZW1wZXQiLCJhIjoiY2l4OXdlYXh4MDAzNDJvbWdwcGdlemhkdyJ9.hPC39hOpk1pO09UHoEGNIw",
                    "accesstoken": "pk.eyJ1Ijoic29saXZyIiwiYSI6ImNqOW9keGxlaTR2Y3gzMHQ0NTZxbzQ3dDIifQ.VTeBL8GaxZFPP048VsTjfA",
                    "style": "mapbox://styles/solivr/cj9pnyf895hu72sova6afy3pa",
                    "bearing": 0,
                    "center": {
                        "lat": 45.436,
                        "lon": 12.330
                    },
                    "zoom": 13,
                    "layers": layers,
                }
            }
        })
])


@app.callback(
    Output(component_id='my-div', component_property='children'),
    [Input(component_id='example-graph', component_property='hoverData')]
)
def update_output_div(input_value):
    return 'Hovered data "{}"'.format(input_value)


@app.callback(
    Output(component_id='my-div-2', component_property='children'),
    [Input(component_id='example-graph', component_property='clickData')]
)
def update_output_div(input_value):
    return 'Clicked data "{}"'.format(input_value)


if __name__ == '__main__':
    app.run_server(debug=True,
                   port=8051)