"""Web app."""
import flask
from flask import Flask, render_template, request, redirect, url_for
import runit

app = flask.Flask(__name__)


@app.route("/")
def index() -> str:
    """Base page."""
    return flask.render_template("layout.html")


@app.route("/about")
def about() -> str:
    """Route to about page."""
    return flask.render_template("about.html")


@app.route("/projects")
def projects() -> str:
    """Route to projects page."""
    return flask.render_template("projects.html")


@app.route("/communication")
def communication() -> str:
    """Route to communication page."""
    return flask.render_template(("communication.html"))

@app.route('/text', methods=['GET', 'POST'])
def text(comments=[]):
    if request.method == "GET":
        return render_template("about.html", comments=comments)
    inputer=request.form["text_input"]
    answer="didn't work"
    answer=runit.compit(inputer)
    comments.append("Political Bias: "+str(answer))
    return redirect(url_for('text'))

if __name__ == "__main__":
    app.run(debug=True)
