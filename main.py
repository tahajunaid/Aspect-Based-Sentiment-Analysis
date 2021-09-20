from flask import Flask, request, jsonify
import predict_polarity
app = Flask(__name__)

@app.route('/API', methods=['GET', 'POST'])
def API():
    text = request.args.get('text',default="", type = str)
    aspect = request.args.get('aspect',default="", type = str)
    if text=="" or aspect=="":
        response = "you missed a mandatory entry"
    else:
        result=predict_polarity.predict(text,aspect)
        response = jsonify({'polarity': result})
        response.headers.add('Access-Control-Allow-Origin', '*')
    
    return response


@app.route('/')
def err():
    return "Unauthorized Access"

if __name__ == "__main__":
    print("Starting Python Flask Server For ASBA")
    app.run(host='0.0.0.0',port=8080)

#Using the API
###http://ec2-3-109-199-145.ap-south-1.compute.amazonaws.com:8080/API?text=<text>&aspect=<aspect>
##replace <text> with your text and <aspect> with the aspect.
##sentence can be entered with spaces or %20 in place of spaces.