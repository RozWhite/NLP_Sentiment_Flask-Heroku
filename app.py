from flask import Flask,render_template,url_for,request
import nltk
nltk.download('vader_lexicon')
from nltk.sentiment.vader import SentimentIntensityAnalyzer

app = Flask(__name__)

@app.route('/')
def home():
	return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predict():


    
    #convert to lowercase
    message = request.form['message']

    sa = SentimentIntensityAnalyzer()
    dd = sa.polarity_scores(message)
    #compound = round((1 + dd['compound'])/2, 2)
    compound = dd['compound']
    print(" result:" , dd['compound'] ,dd['pos'] )

    return render_template('result.html',prediction=compound ,text4=dd['compound'], text1=dd['pos'], text2=dd['neu'] ,text3=dd['neg']  )

if __name__ == '__main__':
	app.run(debug=True)





