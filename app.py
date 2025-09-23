from flask import Flask, flash, redirect, render_template, request, session, jsonify
from flask_session import Session
import numpy as np
import sqlalchemy as db

from kalman import KalmanFilter, KalmanModel
from generate_measurements import Measurements_2D, Systems_2D, Measurements_1D, Systems_1D

# Configure application
app = Flask(__name__)

# Configure session to use filesystem (instead of signed cookies)
app.config["SESSION_PERMANENT"] = False
app.config["SESSION_TYPE"] = "filesystem"
Session(app)


@app.after_request
def after_request(response):
    """Ensure responses aren't cached"""
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    response.headers["Expires"] = 0
    response.headers["Pragma"] = "no-cache"
    return response


@app.route("/")
def index():
    """
    Show starting page
    """

    return render_template("index.html")


@app.route("/kalman_1d")
def kalman_1d():
    models = [x.name for x in KalmanModel]
    systems = [x.name for x in Systems_1D]
    return render_template("kalman_1d.html", models=models, systems=systems)


@app.route("/measurements_1d")
def measurements_1d():
        system = request.args.get("system", "LINEAR")
        dt = request.args.get("dt", 0.01, type=float)
        measurement_variance = request.args.get("measurement_variance", 0.01, type=float)

        process_variance = request.args.get("process_variance", 0.01, type=float)
        model = request.args.get("model", "CONSTANT_VELOCITY")

        zs = Measurements_1D(dt, measurement_variance).gen_measurements(Systems_1D[system])
        filtered = KalmanFilter(dt, process_variance, measurement_variance, dimensions=1, model=KalmanModel[model]).filter_batch(zs)

        # retransform filtered data if system is exponential linearized
        if system == Systems_1D.EXPONENTIAL_LINEARIZED.name:
            zs[:,1] = np.exp(zs[:, 1])
            filtered = np.exp(filtered)
        return jsonify(zs=zs.tolist() , filtered=filtered.tolist())


@app.route("/kalman_2d")
def kalman_2d():
    models = [x.name for x in KalmanModel]
    systems = [x.name for x in Systems_2D]
    return render_template("kalman_2d.html", models=models, systems=systems)


@app.route("/measurements_2d")
def measurements_2d():
        system = request.args.get("system", "CIRCLE")
        dt = request.args.get("dt", 0.01, type=float)
        measurement_variance = request.args.get("measurement_variance", 0.01, type=float)

        process_variance = request.args.get("process_variance", 0.01, type=float)
        model = request.args.get("model", "CONSTANT_VELOCITY")

        zs = Measurements_2D(dt, measurement_variance).gen_measurements(Systems_2D[system])
        filtered = KalmanFilter(dt, process_variance, measurement_variance, dimensions=2, model=KalmanModel[model]).filter_batch(zs)
        return jsonify(zs=zs.tolist() , filtered=filtered.tolist())


@app.route("/feedback", methods=["GET", "POST"])
def about():    
    if request.method == "POST":
        feedback = request.form.get("feedback")

        if feedback is None or feedback.strip() == "":
            flash("Feedback cannot be empty", 'error')
            return redirect("/feedback")
        
        # Create database and table if not exists
        engine = db.create_engine('sqlite:///feedback.db')
        metadata = db.MetaData()
        feedback_table = db.Table('feedback', metadata,
            db.Column('id', db.Integer, primary_key=True, autoincrement=True),
            db.Column('timestamp', db.DateTime, server_default=db.func.now()),
            db.Column('user_ip', db.String),
            db.Column('feedback', db.String),
        )
        metadata.create_all(engine)  # Create table if it doesn't exist

        # Insert feedback into table
        with engine.begin() as conn:
            query = db.insert(feedback_table).values(feedback=feedback, user_ip=request.remote_addr)
            res = conn.execute(query)
            if res.rowcount != 1:
                flash("Failed to submit feedback. Please try again.", 'error')
                return redirect("/feedback")

        flash("Thank you for your feedback!")
        return redirect("/feedback")
    return render_template("feedback.html")
