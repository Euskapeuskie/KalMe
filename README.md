# KalMe
#### Video Demo: https://youtu.be/94jzSP0c-t4
#### Description:
##### General overview:
The purpose of this project is to serve as an educational tool regarding Kalman filters.\
The project has been implemented using Flask as a backend with data generation and kalman filtering happening on the server side. On the client side, necessary values for the Kalman filter are selected like e.g. the pattern of the data, the step size (so to say the sampling interval dt), measurement variance, process variance and the model of the kalman filter. The plots are then generated using JavaScript.
So far only linear Kalman filters have been implemented in 1 and 2 dimensions, but the goal of this website is to extend the functionality further, to provide a bigger overview of state-of-the-art methods when it comes to signal filtering.
- Sensor fusion (e.g. barometer and accelerometer for estimating the height of a drone)
- Extended Kalman filters (to tackle the cases that have been shown to not properly work with linear Kalman filters, like circular motion or sinusoids)
- Unscented Kalman filters
- Monto-Carlo simulation

##### Data - server side:
The page is served as a Flask application. As the client sends a request to show a certain Kalman filter, the following things are executed on the server:
1. **app.py** provides the web page a list of possible values for the system to simulate data for (e.g. linear system, quadratic, ...) as well as a list of possible values which kalman filter systems are implemented (currently constant velocity and constant acceleration). These lists are implemented as enumerators in Python via the Enum package.
2. **generate_measurements.py** generates pseudo-measurements for the selected system, step-size and measurement variance and returns these measurements as numpy arrays. In the 1D-case as [[t0, x0], [t1, x1], ...] or in the 2D-case as [[x0, y0], [x1, y1], [x2, y2], ...]
    - One can select between linear and nonlinear systems to show the limitations of the linear Kalman filter
    - Measurement variance is implemented as a normal distribution which is the ideal case for a Kalman filter. In the exponential 1D-case the growth of the measurement variance is also modelled to be exponential
3. **kalman.py** applies the selected kalman filter batchwise on all the generated measurements and returns the filtered data as numpy arrays. In the 1D-case as [[x0], [x1], [x2], ...] in the 2D-case as [[x0, y0], [x1, y1], [x2, y2]].
4. **app.py** routes the generated measurement data to either /measurements_1d or /measurements_2d in the JSON format containing both the measurement values as well as the filtered values
5. **admin.py** allows viewing the entries in the feedback database but is generally just used as an offline tool and prints to the console.

##### Data - client side:
The client side is implemented as .html pages with additional JavaScript functionalities:
1. The user can select important characteristics regarding measurement data and kalman filter characteristics as input values
2. An event listener on the web page checks for changed values which indicate that new data is available on either /measurements_1d or /measurements_2d as an async function.
3. The data is then fetched and plotted. The used plotting tool is charts.js whereby the plot is implemented as a scatter plot (measurement data points, kalman filtered data lines)
It is also possible to submit feedback via the feedback page. This feedback is parsed into a SQL database using SQLAlchemy containing (ID, Timestamp, User_IP, Feedback)

##### Design choices:
- Implementation of Systems and Kalman Filter Type as Enumerators. This choice has been made on purpose since it allows for concise pattern matching and exhaustive handling of all cases.
- Kalman filter as a class that dynamically handles the number of dimensions (only x or x,y) as well as type (constant velocity or constant acceleration). This prevents rewriting a lot of code for the total of 4 cases. In the future it probably makes sense to imlement new Kalman filter classes once it comes to e.g. sensor fusion, extended kalman filters etc. just to avoid bloating that class.
- Generating measurements as a class for 1-dimensional and 2-dimensional measurements to make things more readable and easy to access in the app.py program.

##### Style:
Default Bootstrap has been used as styling of the pages.