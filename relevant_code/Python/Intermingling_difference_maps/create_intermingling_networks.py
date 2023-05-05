import dash
from dash import Dash, dcc, html, Input, Output, ctx
import dash_cytoscape as cyto
import dash_html_components as html
import dash_bootstrap_components as dbc
import json
import plotly


# import json file with the network
TF = 'CEBPB'
f = open('cytoscape_'+TF+'.json', "r")
network = json.loads(f.read())

cyto.load_extra_layouts()

# define colors and style
my_stylesheet = [
    # Group selectors
    {
        'selector': 'node',
        'style': {
            'content': 'data(label)',
            'background-color': 'data(color)',
            'font-size': '20px'
        }
    },
    {
        'selector': 'edge',
        'style': {'width': 1, 
                 'opacity': 0.4}
    },

    # Class selectors
    {
        'selector': 'node[node_type= "Bridge TF"]',
        'style': {
            'shape': 'triangle',
            'color': 'data(color)',
            'font-size': '30px'
        }
    }, 
    {
        'selector': '.young',
        'style': {
            'background-color': 'blue',
            'line-color': 'blue', 
            'width': 3,
            'opacity': 0.5
        }
    }, 
    {
        'selector': '.old',
        'style': {
            'background-color': 'magenta',
            'line-color': 'magenta',
            'width': 3,
            'opacity': 0.6
        }
    }
]

styles = {
    'output': {
        'overflow-y': 'scroll',
        'overflow-wrap': 'break-word',
        'height': '500px',
        'width': '730px',
        'border': 'thin lightgrey solid'
    },
    'tab': {'height': 'calc(98vh - 115px)'}
}

app = dash.Dash(__name__, title = "Intermingling network for DE genes of " + TF)
server = app.server 

# building the dashboard layout
body = html.Div([html.H1("Intermingling network for DE genes of " + TF)
                 , dbc.Row(dbc.Col(html.Div([
                     cyto.Cytoscape(id='net',
                                    elements=network['elements']['nodes'] + network['elements']['edges'],
                                    layout={'name': 'preset'}, 
                                    style={'width': '730px', 'height': '500px'},
                                    stylesheet=my_stylesheet)])))

                , dbc.Row([dbc.Col(html.H5('Click on node for details:'))])
                , dbc.Row([dbc.Col(html.Div(html.Pre(id='node_tap')))])
              
   ])

# callback to show tap info for node
@app.callback(dash.dependencies.Output('node_tap', 'children'),
              [dash.dependencies.Input('net', 'tapNodeData')])
def displayTapNodeData(data):
    d_node_info_display_json = json.loads(json.dumps(data))
    l_node_info_display =[]
    try:
        l_node_info_display = 'Name: ' + d_node_info_display_json['name'] 
        l_node_info_display = l_node_info_display + '\n Chromosome: ' + d_node_info_display_json['chromosome']
        
    except:
        l_node_info_display = "No node clicked yet"
    return l_node_info_display

           
app.layout = html.Div([body])


if __name__ == '__main__':
    app.run_server(debug=True)