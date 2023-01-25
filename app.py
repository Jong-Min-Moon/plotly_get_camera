import plotly.graph_objects as go
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output

fig = go.Figure(
    go.Surface(
        x = [1,2,3,4,5],
        y = [1,2,3,4,5],
        z = [[0, 1, 0, 1, 0],
             [1, 0, 1, 0, 1],
             [0, 1, 0, 1, 0],
             [1, 0, 1, 0, 1],
             [0, 1, 0, 1, 0]]
    ))

app = dash.Dash()
app.layout = html.Div([
    html.Div(id="output"),        # use to print current relayout values
    dcc.Graph(id="fig", figure=fig)
])




@app.callback(
    Output("output", "children"),
    Input("fig", "relayoutData")
)
def show_data(data):
    # show camera settings like eye upon change
    return [str(data)]

if __name__ == "__main__":
    app.run()
#app.run_server(debug=False, use_reloader=False)