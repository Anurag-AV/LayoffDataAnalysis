from flask import Flask, render_template, request, jsonify
from wordcloud import WordCloud
import joblib
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from io import BytesIO
import base64

app = Flask(__name__)

df1 = pd.read_csv('FinalData 1.csv')


df_first_10 = df1.head(10)

df_html = df_first_10.to_html()

# Load models assuming they are in the same directory and named accordingly
linear_regression_model = joblib.load('linear_regression_model.pkl')
gradient_boosting_model = joblib.load('gradient_boost_model.pkl')

def get_country_by_code(country_dict, code):
    for country, encoded_code in country_dict.items():
        if encoded_code[0] == code:
            return country
    return "Code not found"

def get_stage_by_code(stage_dict, code):
    for stage, encoded_code in stage_dict.items():
        if encoded_code[0] == code:
            return stage
    return "Code not found"

@app.route('/')
def index():
    return render_template('index.html',table = df_html)


@app.route('/predict-linear-regression', methods=['POST'])
def predict_linear_regression():
    try:
        # Extract features from form data for prediction
        feature1 = int(request.form['Year'])
        feature2 = int(request.form['Month'])
        feature3 = float(request.form['FundsRaised'])
        feature4 = int(request.form['Stage'])
        feature5 = int(request.form['Country'])

        country_dict = {'United States': [38],
         'Australia': [1],
         'United Kingdom': [37],
         'Sweden': [33],
         'India': [15],
         'Norway': [25],
         'Canada': [4],
         'Cayman Islands': [5],
         'Israel': [18],
         'Czech Republic': [8],
         'Germany': [13],
         'Singapore': [30],
         'France': [12],
         'Netherlands': [22],
         'Estonia': [10],
         'Kenya': [19],
         'Brazil': [3],
         'New Zealand': [23],
         'Ireland': [17],
         'Chile': [6],
         'Spain': [32],
         'South Korea': [31],
         'Indonesia': [16],
         'China': [7],
         'Argentina': [0],
         'Nigeria': [24],
         'Denmark': [9],
         'Thailand': [35],
         'Senegal': [28],
         'Hong Kong': [14],
         'United Arab Emirates': [36],
         'Austria': [2],
         'Finland': [11],
         'Malaysia': [20],
         'Mexico': [21],
         'Russia': [27],
         'Seychelles': [29],
         'Switzerland': [34],
         'Portugal': [26]}

        stage_dict = {'Post-IPO': [1],
         'Series C': [6],
         'Unknown': [15],
         'Series A': [4],
         'Acquired': [0],
         'Series D': [7],
         'Series B': [5],
         'Series F': [9],
         'Series E': [8],
         'Private Equity': [2],
         'Series H': [11],
         'Series G': [10],
         'Seed': [3],
         'Subsidiary': [14],
         'Series I': [12],
         'Series J': [13]}


        features = [feature1, feature2, feature3, feature4]

        # Predict using the linear regression model
        prediction = linear_regression_model.predict([features])[0]
        prediction = prediction * 100
        prediction = round(prediction, 2)

        #country = "United States"
        # Now create the plot
        country = get_country_by_code(country_dict, feature5)
        dat = df1[df1['Stage'].isin(['Post-IPO', 'Seed', 'Series A', 'Series B', 'Series C'])]
        dat = dat[dat['Country'] == country]
        g = sns.catplot(x="Year", col="Stage", col_wrap=3,
                        data=dat[dat.Laid_Off_Count.notnull()],
                        kind="count", height=2.5, aspect=.8, hue="Year", palette="Set1")

        g.set(xticklabels=[])  # remove the tick labels
        g.set(ylabel="Count in Thousands")
        g.tick_params(bottom=False)

        img1 = BytesIO()
        g.savefig(img1, format='png')
        plt.close(g.fig)
        img1.seek(0)
        plot_url1 = base64.b64encode(img1.getvalue()).decode('utf-8')

        # Plot 2: The new line plot
        stage = get_stage_by_code(stage_dict, feature4)
        df_2022 = df1[df1['Stage'] == stage][['Laid_Off_Count', 'Month']]
        average_values22 = df_2022.groupby('Month').mean()
        month_order = ['January', 'February', 'March', 'April', 'May', 'June',
                       'July', 'August', 'September', 'October', 'November', 'December']
        average_values22 = average_values22.reindex(month_order)

        plt.figure(figsize=(10, 5))  # Set the figure size
        plt.plot(average_values22.index, average_values22['Laid_Off_Count'], marker='o', linestyle='-', color='orange')
        plt.xlabel('Month')
        plt.ylabel('Average Lay Off Count')
        plt.legend(['Lay Off Trend'], loc="upper left")
        plt.title('Lay off percent trend Over months')
        plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
        plt.grid(True)

        # Save plot 2 to a BytesIO buffer
        img2 = BytesIO()
        plt.savefig(img2, format='png')
        plt.close()
        img2.seek(0)
        plot_url2 = base64.b64encode(img2.getvalue()).decode('utf-8')

        series = get_stage_by_code(stage_dict, feature4)
        data_wordcloud = df1[df1['Stage'] == series][["Company", "Laid_Off_Count"]]
        data_wordcloud = data_wordcloud.groupby('Company').sum()
        data_wordcloud = data_wordcloud.sort_values(by="Laid_Off_Count")
        data_wordcloud = data_wordcloud.tail(15)
        data_wordcloud = data_wordcloud.squeeze().to_dict()
        wordcloud = WordCloud(width=800, height=400, background_color='white')
        wordcloud.generate_from_frequencies(data_wordcloud)

        # Save wordcloud to a BytesIO buffer
        img3 = BytesIO()
        wordcloud.to_image().save(img3, format='PNG')
        plt.close()
        img3.seek(0)
        plot_url3 = base64.b64encode(img3.getvalue()).decode('utf-8')

        # Plot 4: Bar Chart
        stage = get_stage_by_code(stage_dict, feature4)
        category_counts = df1[df1['Stage'] == stage]['Reason'].value_counts()
        sorted_counts = category_counts.sort_values(ascending=False)

        fig, ax = plt.subplots()
        ax.bar(sorted_counts.index, sorted_counts.values, color='lightgreen')
        ax.set(title='Reason of layoff', xlabel='Reason', ylabel='Counts')
        plt.xticks(rotation=45, ha="right")  # Rotate x-axis labels for better readability

        # Save bar chart to a BytesIO buffer
        img4 = BytesIO()
        plt.tight_layout()
        plt.savefig(img4, format='png')
        plt.close()
        img4.seek(0)
        plot_url4 = base64.b64encode(img4.getvalue()).decode('utf-8')

        # Pass the plot data and the prediction to the template
        return render_template('result.html', model_name='Linear Regression', prediction=prediction,
                               plot_url1=plot_url1, plot_url2=plot_url2, plot_url3=plot_url3, plot_url4=plot_url4)

    except Exception as e:
        # Render an error template if something goes wrong
        return render_template('error.html', error=str(e))

@app.route('/predict-gradient-boosting', methods=['POST'])
def predict_gradient_boosting():
    try:
        # Extract features from form data for prediction
        feature1 = float(request.form['Month'])
        feature2 = float(request.form['Industry'])
        feature3 = float(request.form['Stage'])
        feature4 = float(request.form['Country'])
        features = [feature1, feature2, feature3, feature4]

        country_dict = {'United States': [38],
                        'Australia': [1],
                        'United Kingdom': [37],
                        'Sweden': [33],
                        'India': [15],
                        'Norway': [25],
                        'Canada': [4],
                        'Cayman Islands': [5],
                        'Israel': [18],
                        'Czech Republic': [8],
                        'Germany': [13],
                        'Singapore': [30],
                        'France': [12],
                        'Netherlands': [22],
                        'Estonia': [10],
                        'Kenya': [19],
                        'Brazil': [3],
                        'New Zealand': [23],
                        'Ireland': [17],
                        'Chile': [6],
                        'Spain': [32],
                        'South Korea': [31],
                        'Indonesia': [16],
                        'China': [7],
                        'Argentina': [0],
                        'Nigeria': [24],
                        'Denmark': [9],
                        'Thailand': [35],
                        'Senegal': [28],
                        'Hong Kong': [14],
                        'United Arab Emirates': [36],
                        'Austria': [2],
                        'Finland': [11],
                        'Malaysia': [20],
                        'Mexico': [21],
                        'Russia': [27],
                        'Seychelles': [29],
                        'Switzerland': [34],
                        'Portugal': [26]}

        stage_dict = {'Post-IPO': [1],
                      'Series C': [6],
                      'Unknown': [15],
                      'Series A': [4],
                      'Acquired': [0],
                      'Series D': [7],
                      'Series B': [5],
                      'Series F': [9],
                      'Series E': [8],
                      'Private Equity': [2],
                      'Series H': [11],
                      'Series G': [10],
                      'Seed': [3],
                      'Subsidiary': [14],
                      'Series I': [12],
                      'Series J': [13]}

        # Predict using the gradient boosting model
        prediction = gradient_boosting_model.predict([features])[0]
        prediction = prediction * 100
        prediction = round(prediction, 2)

        # Now create the plot
        country = get_country_by_code(country_dict, feature4)
        dat = df1[df1['Stage'].isin(['Post-IPO', 'Seed', 'Series A', 'Series B', 'Series C'])]
        dat = dat[dat['Country'] == country]
        g = sns.catplot(x="Year", col="Stage", col_wrap=3,
                        data=dat[dat.Laid_Off_Count.notnull()],
                        kind="count", height=2.5, aspect=.8, hue="Year", palette="Set1")

        g.set(xticklabels=[])  # remove the tick labels
        g.set(ylabel="Count in Thousands")
        g.tick_params(bottom=False)

        img1 = BytesIO()
        g.savefig(img1, format='png')
        plt.close(g.fig)
        img1.seek(0)
        plot_url1 = base64.b64encode(img1.getvalue()).decode('utf-8')

        # Plot 2: The new line plot
        stage = get_stage_by_code(stage_dict, feature3)
        df_2022 = df1[df1['Stage'] == stage][['Laid_Off_Count', 'Month']]
        average_values22 = df_2022.groupby('Month').mean()
        month_order = ['January', 'February', 'March', 'April', 'May', 'June',
                       'July', 'August', 'September', 'October', 'November', 'December']
        average_values22 = average_values22.reindex(month_order)

        plt.figure(figsize=(10, 5))  # Set the figure size
        plt.plot(average_values22.index, average_values22['Laid_Off_Count'], marker='o', linestyle='-', color='orange')
        plt.xlabel('Month')
        plt.ylabel('Average Lay Off Count')
        plt.legend(['Lay Off Trend'], loc="upper left")
        plt.title('Lay off percent trend Over months')
        plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
        plt.grid(True)

        # Save plot 2 to a BytesIO buffer
        img2 = BytesIO()
        plt.savefig(img2, format='png')
        plt.close()
        img2.seek(0)
        plot_url2 = base64.b64encode(img2.getvalue()).decode('utf-8')

        series = get_stage_by_code(stage_dict, feature3)
        data_wordcloud = df1[df1['Stage'] == series][["Company", "Laid_Off_Count"]]
        data_wordcloud = data_wordcloud.groupby('Company').sum()
        data_wordcloud = data_wordcloud.sort_values(by="Laid_Off_Count")
        data_wordcloud = data_wordcloud.tail(15)
        data_wordcloud = data_wordcloud.squeeze().to_dict()
        wordcloud = WordCloud(width=800, height=400, background_color='white')
        wordcloud.generate_from_frequencies(data_wordcloud)

        # Save wordcloud to a BytesIO buffer
        img3 = BytesIO()
        wordcloud.to_image().save(img3, format='PNG')
        plt.close()
        img3.seek(0)
        plot_url3 = base64.b64encode(img3.getvalue()).decode('utf-8')

        # Plot 4: Bar Chart
        stage = get_stage_by_code(stage_dict, feature3)
        category_counts = df1[df1['Stage'] == stage]['Reason'].value_counts()
        sorted_counts = category_counts.sort_values(ascending=False)

        fig, ax = plt.subplots()
        ax.bar(sorted_counts.index, sorted_counts.values, color='lightgreen')
        ax.set(title='Reason of layoff', xlabel='Reason', ylabel='Counts')
        plt.xticks(rotation=45, ha="right")  # Rotate x-axis labels for better readability

        # Save bar chart to a BytesIO buffer
        img4 = BytesIO()
        plt.tight_layout()
        plt.savefig(img4, format='png')
        plt.close()
        img4.seek(0)
        plot_url4 = base64.b64encode(img4.getvalue()).decode('utf-8')

        return render_template('result.html', model_name='Gradient Boosting', prediction=prediction,
                               plot_url1=plot_url1, plot_url2=plot_url2, plot_url3=plot_url3, plot_url4=plot_url4)

    except Exception as e:
        # Render an error template if something goes wrong
        return render_template('error.html', error=str(e))
if __name__ == '__main__':
    app.run(debug=True)