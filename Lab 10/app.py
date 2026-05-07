from flask import Flask, render_template, request

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    msg = request.form["msg"].lower()

    # Library Assistant Bot logic
    if "book" in msg:
        reply = "You can search books in the library catalog or ask librarian for help."
    elif "due" in msg:
        reply = "Books must be returned within 14 days from issue date."
    elif "timing" in msg or "hours" in msg or "open" in msg:
        reply = "Library opening hours are 9:00 AM to 5:00 PM (Monday to Friday)."
    elif "hello" in msg or "hi" in msg:
        reply = "Hello! I am Library Assistant Bot. How can I help you?"
    elif "help" in msg:
        reply = "I can help you with books, due dates, and library timings."
    else:
        reply = "Sorry, I can only help with library related queries."

    return reply

if __name__ == "__main__":
    app.run(debug=True)