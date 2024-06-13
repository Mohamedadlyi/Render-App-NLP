import dash
import dash_bootstrap_components as dbc
from dash import html, Input, Output, State, dcc
import pandas as pd
import joblib
import plotly.express as px
import plotly.graph_objects as go

import re
import nltk
from nltk.corpus import stopwords
from sklearn.base import BaseEstimator, TransformerMixin

nltk.download('stopwords')

# Text Preprocessing
class TextCleaner(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = pd.Series(X)
        return X.apply(self.clean_text)

    @staticmethod
    def clean_text(text):
        text = TextCleaner.remove_urls(text)
        text = TextCleaner.remove_mentions(text)
        text = TextCleaner.remove_english_words(text)
        text = TextCleaner.remove_unicode_bmp(text)
        text = TextCleaner.remove_emoji_shortcodes(text)
        text = TextCleaner.remove_specific_punctuation(text)
        text = TextCleaner.remove_complex_patterns(text)
        text = TextCleaner.remove_various_punctuation(text)
        text = TextCleaner.remove_numbers(text)
        text = TextCleaner.remove_extra_spaces(text)
        return text

    @staticmethod
    def remove_urls(text):
        return re.sub(r'http[s]?://\S+', ' ', text)

    @staticmethod
    def remove_mentions(text):
        return re.sub(r'@\w+', ' ', text)

    @staticmethod
    def remove_english_words(text):
        return re.sub(r'\b[a-zA-Z]+\b', ' ', text)

    @staticmethod
    def remove_unicode_bmp(text):
        return re.sub(r'[\U00010000-\U0010ffff]', ' ', text)

    @staticmethod
    def remove_emoji_shortcodes(text):
        return re.sub(r':[a-z_]+:', ' ', text)

    @staticmethod
    def remove_specific_punctuation(text):
        return re.sub(r'[*!?#@]', ' ', text)

    @staticmethod
    def remove_complex_patterns(text):
        return re.sub(r'\|\|+\\s*\d+%\s*\|\|+?[_\-\.\?]+', ' ', text)

    @staticmethod
    def remove_various_punctuation(text):
        return re.sub(r'[_\-\.\"\:\;\,\'\،\♡\\\)/(\&\؟]', ' ', text)

    @staticmethod
    def remove_numbers(text):
        return re.sub(r'\d+', ' ', text)

    @staticmethod
    def remove_extra_spaces(text):
        return ' '.join(text.split())

# Model 
class TextClassificationModel:
    def __init__(self, model_type='logistic'):
        self.model_type = model_type
        self.pipeline = None

    def build_pipeline(self):
        if self.model_type == 'logistic':
            self.pipeline = Pipeline([
                ('cleaner', TextCleaner()),
                ('vectorizer', CountVectorizer()),
                ('classifier', LogisticRegression())
            ])
        elif self.model_type == 'naive_bayes':
            self.pipeline = Pipeline([
                ('cleaner', TextCleaner()),
                ('vectorizer', CountVectorizer()),
                ('classifier', MultinomialNB())
            ])

    def train(self, X_train, y_train):
        self.build_pipeline()
        self.pipeline.fit(X_train, y_train)

    def evaluate(self, X_test, y_test):
        y_pred = self.pipeline.predict(X_test)
        return accuracy_score(y_test, y_pred)

    def save(self, filename):
        joblib.dump(self.pipeline, filename)

    def load(self, filename):
        self.pipeline = joblib.load(filename)

    def predict(self, text):
        return self.pipeline.predict([text])



# Load the model
import __main__
__main__.TextCleaner = TextCleaner
__main__.TextClassificationModel = TextClassificationModel

nb_model = TextClassificationModel(model_type='naive_bayes')
nb_model.load('src/nb_model.pkl')
txt_cleaner =TextCleaner()
# Initialize Dash app
app = dash.Dash(external_stylesheets=[dbc.themes.SLATE])
server = app.server

# Define the layout of the app
app.layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.H1("تخمين اللهجات العربية",  style={'textAlign': 'center', 'marginTop': '50px', 'marginBottom': '20px', 'color': 'gold', 'fontWeight': 'bold'})
        ])
    ]),
    dbc.Row([
        html.Div(style={"height": "20px"}), 
        dbc.Col([
            html.H2("(مصري - ليبي - مغربي - سوداني- لبناني)", style={'textAlign': 'center', 'marginBottom': '30px', 'color': 'gold', 'fontWeight': 'bold'})
        ])
    ]),
    dbc.Row([
        dbc.Col([
            html.Div(style={"height": "50px"}),
            dcc.Textarea( 
                id="text-input",
                placeholder="رجاء إدخال نص عربي",
                className='dark-textarea', 
            ),
            dbc.Row([
                html.Div(style={"height": "10px"}),   
                dbc.Col(dbc.Button(" تخمين", id="predict-button", color="light", className=' btn-primary ms-auto', style={"width":  "190px"} )),
                dbc.Col(html.Button("مسح", id="reset-button", n_clicks=0, className='btn btn-danger'))]),
            dbc.Row([ 
                html.Div(style={"height": "10px"}),   
                dbc.Col(html.Div(id="prediction-alert"), width=20)
                ])
        ], width="12")
    ]),
    dbc.Row([dcc.Graph(id="probability-graph")] 
        )
    ])


def empty_fig(fig= None, alert = None, text = ''):
    fig = go.Figure()
    fig.update_layout(
        xaxis=dict(showgrid=False, visible=False, showticklabels=False),
        yaxis=dict(showgrid=False, visible=False, showticklabels=False),
        plot_bgcolor='rgba(0,0,0,0)',  # Transparent plot background
        paper_bgcolor='rgba(0,0,0,0)'
    )
    return fig, alert, text


@app.callback(
    [Output("probability-graph", "figure"),
     Output("prediction-alert", "children"),
     Output("text-input", "value")],
    [Input("predict-button", "n_clicks"),
     Input("reset-button", "n_clicks")],
    [State("text-input", "value")]
)
def update_output(predict_clicks, reset_clicks, text):
    ctx = dash.callback_context
    if not ctx.triggered:
        button_id = None
    else:
        button_id = ctx.triggered[0]["prop_id"].split(".")[0]

    if button_id == "predict-button" and (predict_clicks is None or text is None or text.strip() == ""):
        return dash.no_update, None, text
    elif button_id == "predict-button":
        if txt_cleaner.clean_text(text) == '':
            alert = dbc.Alert("نص غير عربي", color="cyan", className="d-grid gap-2 col-6 mx-auto", style={'textAlign': 'center', 'marginTop': '50px','marginBottom': '30px', 'color': 'black', 'fontWeight': 'bold', 'fontSize': '20px'})
            print('me')
            return empty_fig(alert, 'نص غير عربي , رجاء إدخال نص عربي')
        probs = nb_model.pipeline.predict_proba(text)[0]
        target_names = ['مصري', 'لبناني', 'ليبي', 'مغربي', 'سوداني']
   
        fig = px.bar(x=target_names, y=probs, labels={'x': 'الدولة', 'y': 'الاحتمال'}, title='نسبة التأكد من التخمين')

        fig.update_traces(marker_color=['cyan' if label == target_names[probs.argmax()] else 'beige' for label in target_names],
                          text=probs, texttemplate='%{text:.2f}', textposition='outside', textfont_size=14, textfont_family='Arial', textfont_color='white')
        
        fig.update_layout(height=400, width=600, margin=dict(l=40, r=40, t=40, b=40), 
                          title={'x':0.5, 'xanchor': 'center', 'font': {'size': 20, 'family': 'Arial', 'color': 'white'}},
                          plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                          xaxis={'title': {'text': 'الدولة', 'font': {'size': 28, 'family': 'Arial', 'color': 'white'}}, 
                                 'tickfont': {'size': 20, 'family': 'Arial', 'color': 'white'}},
                          yaxis={'title': {'text': 'الاحتمال', 'font': {'size': 24, 'family': 'Arial', 'color': 'white'}}, 
                                 'tickfont': {'size': 14, 'family': 'Arial', 'color': 'white'}})

        alert = dbc.Alert(f"التنبؤ: {target_names[probs.argmax()]}", color="cyan", className="d-grid gap-2 col-6 mx-auto", style={'textAlign': 'center', 'marginTop': '50px','marginBottom': '30px', 'color': 'black', 'fontWeight': 'bold', 'fontSize': '20px'})
        return fig, alert, text
    elif button_id == "reset-button":
        return empty_fig()
    else:
        return dash.no_update, None, text

if __name__ == '__main__':
    app.run_server(debug=True)
