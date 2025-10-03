import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from pathlib import Path
from datetime import datetime
from db import save_comment
from utils import lin_reg
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

ROOT = Path(__file__).resolve().parent.parent
HOUSING_DATA_FILE = ROOT / "data" / "Housing.csv"

def highlight(text, color="#00cc6633"):
    """Return text string wrapped in HTML span with background highlight."""
    return f'<span style="background-color:{color}; padding:2px 4px; border-radius:3px"><b>{text}</b></span>'

housing = pd.read_csv(HOUSING_DATA_FILE)


def quadratic(a,b,c,x):
    y = a*x**2+b*x+c
    return y
def deg_nine_poly(a0, a1,a2,a3,a4,a5,a6,a7,a8,a9, x):
    poly = a9*x**9 + a8*x**8 + a7*x**7 + a6*x**6
def generate_employee_data(n_total=10, seed_choice=1):
    if seed_choice == 1:
        seed_1 = 42
        rng = np.random.default_rng(seed_1)
        training_hours = rng.normal(30, 20, n_total).clip(0,50)
        production = 1000*(1-np.exp(-0.35*(training_hours+3)))+ 50 + rng.normal(0,75,n_total)

        df = pd.DataFrame({
            "Training Hours": training_hours,
            "Production": production
        })

    else:
        seed_2 = 43
        rng = np.random.default_rng(seed_2)
        training_hours = rng.normal(30, 20, n_total).clip(0,50)
        production = 1000*(1-np.exp(-0.35*(training_hours+3)))+ 50 + rng.normal(0,75,n_total)

        df = pd.DataFrame({
            "Training Hours": training_hours,
            "Production": production 
        })

    return df


def quiz_card(question_title, question_text, options, correct_answer, key, correct_feedback, incorrect_feedback):
    st.markdown(
        """
        <style>
        div[data-testid="stExpander"] {
            border: 2px solid var(--secondary-background-color);
            border-radius: 12px;
            box-shadow: 2px 2px 8px rgba(0,0,0,0.2);
            margin-bottom: 12px;
        }
        /* Expander header text inherits theme colors but bolded */
        div[data-testid="stExpander"] div[role="button"] p {
            font-weight: 600;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    with st.expander(f"{question_title}", expanded=False):
        st.write(question_text)

        answer = st.selectbox("Answer:", options=options, index=None, key=f"select_{key}")

        if st.button(label="Submit", key=f"btn_{key}"):
            if answer == correct_answer:
                st.success(correct_feedback)
            else:
                st.error(incorrect_feedback)

def calculate_average_cost(X,y, model):
    avg_cost = np.mean((y-model(X))**2)
    return avg_cost

st.title("Demo 3: Underfitting and Overfitting", width="content")

st.subheader("Goals:")
st.markdown(
    f"""
    1. Understand the concept of model {highlight("complexity")}.
    2. Visually identify when a model is {highlight("overfit")}.
    3. Visually identify when a model is {highlight("underfit")}.

    **NOTE:** \\
    (1) Any questions posed within Checkpoints are only for you to monitor your understanding and will not be graded.
    The graded component of each walkthrough is the associated worksheet, which will ask questions based directly on the content
    shown here. 
    
   (2) Any quantity which has *parentheses in the exponent* is only a label
    but NOT an exponent in the multiplicative sense!
    """, unsafe_allow_html=True
)
st.markdown("---")


st.markdown("""
             ### 0. Motivating Example: Employee Training vs Productivity 
             
            In virtually any company, when a new employee is hired they will
            go through an onboarding or training process to prepare them for
            the role they will take on. Something the company must consider is
            how much time to dedicate to employee training; hours of training
            do not contribute to production immediately, but will typically
            pay off in the long run.

            We are considering a collection of fictional barista hires made
            at select Starbucks locations. Training time is not standardized,
            so managers are allowed to ease their hirees into work at their
            discretion. The **input is training hours**, while the measured
            output is **number of positively reviewed orders completed by
            the employee**, simply referred to as **production**. Starbucks
            wants to know: **after what amount of training does production
            stop increasing?**

            Our data science team is initially presented with only 10
            collected data points. 
             """)
st.markdown("---")

employee_data = generate_employee_data(seed_choice=1)

st.markdown("""
             ### 1. Exploratory Data Analysis
             
            As in previous demos, plotting the data is always
            a good first step. We display the observed values in
            a scatterplot (no connecting lines), since the straight
            line visual could influence our interpretation of the trend.
            The numeric values for each point are displayed as well. In the
            next expander, standard descriptive statistics have been calculated.
             """)
with st.expander("Employee Training Data"):
    fig, ax = plt.subplots()
    ax.scatter(employee_data['Training Hours'], employee_data['Production'], marker='o', color='brown')
    ax.set_xlabel("Hours of Training")
    ax.set_ylabel("Production")
    ax.set_title("Production vs. Training")
    st.pyplot(fig)

    st.write(employee_data)

with st.expander("Descriptive Statistics"):
    st.write(employee_data.describe())

quiz_card(
    question_title="1.1 Checkpoint",
    question_text="""As number of training hours increases,
    production...  
    """,
    options=['(a) Decreases.', 
             "(b) Is roughly constant.", 
             "(c) Increases linearly.",
             "(d) Increases sharply, then levels off."],
    key="check11",
    correct_answer="(d) Increases sharply, then levels off.",
    correct_feedback="""Correct! The employees with almost no training
    have quite low production, but production rises quickly until
    training reaches roughly 15 hours. Then, production rises at a
    much slower rate. 
    """,
    incorrect_feedback="Try again. If we plotted a curve which fit " \
    "the data well, how would its slope change as training hours increase?"
)

quiz_card(
    question_title="1.2 Checkpoint",
    question_text="""What is the minimum number of hours an employee
    was trained for?  
    """,
    options=['10.135', 
             "0", 
             "735.1255",
             "Not enough info to determine."],
    key="check12",
    correct_answer="0",
    correct_feedback="""Correct! Under Descriptive Statistics, in
    the row labeled 'min' the 'Training Hours' column shows 0.
    """,
    incorrect_feedback="Try again. Use the Descriptive Statistics table."
)

st.markdown(
""" Already we are able to draw a valuable (albeit rough)
conclusion for the company: it appears that the most value
is gained from training up to approximately 15 hours,
but there are diminishing returns thereafter. While this
is good for a start, our work is clearly not sufficient
to make decisions from. Specifically:

- we have not produced a tangible model which accurately predicts 
production based on hours of training;
- unfortunately, 10 data points is an extremely small sample given that
Starbucks has over 300,000 total employees. We may not
have enough data to capture the true trend of the population.

The first of these bullet points is a non-issue, as we will fit
several models in the next sections. The second bullet point is an
issue which is at times unavoidable; despite living in the age of
data and analytics, data is not always readily available to us.

""")
st.markdown("---")
st.markdown(f"""
            ### 2. Model Selection
            Once we have our dataset in front of us and we've done 
            some exploratory analysis, it's time to find the best model
            for our purposes. In theory, we can choose from any function
            or {highlight("parametric model")} (and a wide array of {highlight("non-parametric models")}
            exists too, but we'll steer clear for now). One of our first jobs is to
            determine which *type* of model will work best.
            """, unsafe_allow_html=True)

st.markdown(f"""
            #### 2.1 High Model Complexity
            Something which *seems* logical to do is to fit a model function
            whose graph passes through as many of the data points as
            possible. In Demo 2 we learned that minimizing the 
            {highlight("average squared error")} across all points amounts
            to finding the best model, so fitting a model curve which nearly
            passes through or {highlight("interpolates")} the data seems
            like the way to go. Here, we can achieve that by expressing
            $$p(x)$$, the production of an employee as a function of $$x$$ training
            hours, as a polynomial of sufficiently large degree.
            """, unsafe_allow_html=True)

employee_data_extra = employee_data.copy()

X = employee_data_extra[['Training Hours']]
y = employee_data_extra['Production']
degree = 9
poly = PolynomialFeatures(degree=degree, include_bias=False)
X_poly = poly.fit_transform(X)

model = LinearRegression().fit(X_poly, y)

x_range = np.linspace(X.min(), X.max(), 200).reshape(-1, 1)
y_pred = model.predict(poly.transform(x_range))

training_hours_range = np.arange(start=np.min(employee_data_extra['Training Hours']),
                             stop=np.max(employee_data_extra['Training Hours']))


fig_quad_b, ax_quad_b = plt.subplots()
ax_quad_b.scatter(employee_data_extra['Training Hours'], employee_data_extra['Production'], color="black")
ax_quad_b.plot(x_range, y_pred, linestyle='--', color='blue', linewidth=2)
ax_quad_b.set_xlabel("Training Hours")
ax_quad_b.set_ylabel("Production")
ax_quad_b.set_title(f"Degree {degree} Polynomial Fit")
st.pyplot(fig_quad_b)

st.markdown(f"""
            The curve shown above passes directly through, or very close
            to, every single data point. The polynomial is explicitly given by

            $$p(x) = 0.000000004x^9-0.0000008x^8+0.00007x^7-0.0029x^6+0.068x^5$$

            $$   \qquad\qquad -0.8237x^4+3.9587x^3+1.3628x^2+0.287x+751.6221$$

            By adding many terms to our model (e.g. powers of $$x$$ as in a polynomial,
            exponential functions of $$x$$, or others), it gains 
            {highlight("complexity")}. Here, a polynomial of degree nine is
            able to take on a very complicated shape, meaning model
            {highlight("flexibility")} increases with complexity. **Our
            main question in this section**: is more complexity/flexibility 
            always better?
            """, unsafe_allow_html=True)

quiz_card(
    question_title="2.1 Checkpoint",
    question_text="""Does the above model look *intuitively* correct?
    """,
    options=['Yes, the curve passes near to every data point which is all that matters.', 
             "No, the function captures noise in addition to the underlying signal.", 
             "Not enough information to determine."],
    key="check21",
    correct_answer="No, the function captures noise in addition to the underlying signal.",
    correct_feedback="""Correct! We expect or assume that production $p(x) = p_0(x) + \\text{noise}$, where 
    $p(x)$ is the observed production given $x$ training hours, $p_0(x)$ is the
    true relationship between training and production (a deterministic signal function),
    and $\\text{noise}$ is a random deviation (non-deterministic). We don't 
    want our model to capture $\\text{noise}$.
    """,
    incorrect_feedback="Try again. Do we expect the true relationship between training and production" \
    " is super wiggly and complicated?"
)

employee_data_new = generate_employee_data(seed_choice=22)

st.markdown(f"""
            The above checkpoint indicates that higher complexity is not
            always better, which may be nonintuitive. In the next two sections,
            we will see why this is the case.

            ##### 2.1.1 Poor Generalizability

            A model with too much complexity will not 
            {highlight("generalize")} well to new data. Suppose ten new data
            points are collected; since they are sampled from
            the same population as before, we expect the true relationship between
            training hours and production to be the same. This means 
            the model which we trained before should have {highlight("roughly the same performance"\
            " on data which it hadn't previously seen")}. However, plotting the
            new data with the previous model curve, we see:

            """, unsafe_allow_html=True)

with st.expander("Model Performance on New Data"):
    fig_new, ax_new = plt.subplots()
    ax_new.scatter(employee_data_new['Training Hours'], employee_data_new['Production'], color="black")
    ax_new.plot(x_range, y_pred, linestyle='--', color='blue', linewidth=2)
    ax_new.set_xlabel("Training Hours")
    ax_new.set_ylabel("Production")
    ax_new.set_title(f"Degree {degree} Polynomial Fit with New Data")
    st.pyplot(fig_new)

st.markdown("""
            The model fits the new data...well, terribly. Because it was complex enough
            to follow the original ten data points, it did exactly that; however,
            this leaves unseen data points out in the cold. When given the
            license to do so, the model tries too hard to match the data it is given.
            In so doing, its performance on new data plummets, which we want to
            avoid. 

            ##### 2.1.2 Capturing Signal vs Capturing Noise
            Our expectation (as explained in Demo 1 and Checkpoint 2.1 here)
            is that 
            
            $$p(x) = p_0(x) + \\text{noise},$$

            where $$p_0(x)$$ is the true or deterministic relationship between
            training hours and production. We want our model to be sufficiently
            complex to match the graph of $p_0(x)$, but not so complex that
            it also captures $\\text{noise}$. The $\\text{noise}$ can be viewed
            as random measurement error associated with an experiment, and is **not**
            what our model should follow! 
            """, unsafe_allow_html=True)

st.markdown(f"""
            When this happens the model will fail
            to generalize, and is known as {highlight("overfitting")}. In other
            words, the model has {highlight("high variance")}. This is because
            if we were to train a model of the same complexity with a different
            sample of data, the coefficients of the resulting curve vary
            drastically. For instance, below we train a new degree-9 polynomial
            model on the new set of ten data points. Notice how different
            the resulting curve is!""", unsafe_allow_html=True)

with st.expander("Polynomial Models from Different Training Sets"):

    X_new = employee_data_new[['Training Hours']]
    y_new = employee_data_new['Production']
    degree = 9
    poly = PolynomialFeatures(degree=degree, include_bias=False)
    X_poly_new = poly.fit_transform(X_new)

    model = LinearRegression().fit(X_poly_new, y_new)

    x_range = np.linspace(X_new.min(), X_new.max(), 200).reshape(-1, 1)
    y_pred_new = model.predict(poly.transform(x_range))

    training_hours_range = np.arange(start=np.min(employee_data_new['Training Hours']),
                                stop=np.max(employee_data_new['Training Hours']))


    fig_var, ax_var = plt.subplots()
    ax_var.scatter(employee_data_new['Training Hours'], employee_data_new['Production'], color="maroon",s=100, label='New Data')
    ax_var.scatter(employee_data_extra['Training Hours'], employee_data_extra['Production'], color="black", s=100, label='Original Data')
    ax_var.plot(x_range, y_pred_new, linestyle='--', color='red', linewidth=2.5, label='New Model')
    ax_var.plot(x_range, y_pred, linestyle='--', color='blue', linewidth=2.5, label='Original Model')
    ax_var.set_xlabel("Training Hours")
    ax_var.set_ylabel("Production")
    ax_var.set_title(f"Comparing Polynomial Fits for Different Samples")
    ax_var.legend()
    st.pyplot(fig_var)           
 
st.markdown(f"""
           #### 2.2 Low Model Complexity

            On the opposite end of the spectrum, we could try to
            fit a {highlight("low complexity")} model such as a line
            to the original ten data points. Using a simple model has
            two immediate benefits: (1) it is relatively {highlight("easy to interpret")}
            or explain, and (2) it will likely be {highlight("computationally inexpensive")}.
            The linear model of our data is shown below.
            """, unsafe_allow_html=True)

X = employee_data_extra[['Training Hours']]
y = employee_data_extra['Production']
degree_low = 1
poly = PolynomialFeatures(degree=degree_low, include_bias=False)
X_poly = poly.fit_transform(X)

model = LinearRegression().fit(X_poly, y)

x_range = np.linspace(X.min(), X.max(), 200).reshape(-1, 1)
y_pred = model.predict(poly.transform(x_range))

training_hours_range = np.arange(start=np.min(employee_data_extra['Training Hours']),
                             stop=np.max(employee_data_extra['Training Hours']))


fig_quad_b, ax_quad_b = plt.subplots()
ax_quad_b.scatter(employee_data_extra['Training Hours'], employee_data_extra['Production'], color="black")
ax_quad_b.plot(x_range, y_pred, linestyle='--', color='blue', linewidth=2)
ax_quad_b.set_xlabel("Training Hours")
ax_quad_b.set_ylabel("Production")
ax_quad_b.set_title(f"Degree {degree_low} Polynomial Fit")
st.pyplot(fig_quad_b)

st.markdown(f"""
            Immediate drawback of such a simple model: it is not
            flexible enough to follow the signal function
            $p_0(x)$. We identified immediately that the
             trend of the data does not have a constant
            slope, but a linear function can never capture this! So,
            while it *does* produce similar results on new data (see expander below),
            it makes consistently large errors. This is known as {highlight("underfitting")},
            {highlight("high bias")}, or {highlight("low variance")}. What this means is that fitting a
            model of the same complexity on new data will result in a similar
            curve. For instance, in the plot below, the two lines are nearly parallel and only
            separated by a relatively small amount.
            """, unsafe_allow_html=True)


with st.expander("Fitting Low-Complexity Models on Different Data"):
    X_0 = employee_data_extra[['Training Hours']]
    y_0 = employee_data_extra['Production']

    X_1 = employee_data_new[['Training Hours']]
    y_1 = employee_data_new['Production']
    degree_low = 1
    poly = PolynomialFeatures(degree=degree_low, include_bias=False)
    X_poly_0 = poly.fit_transform(X_0)
    X_poly_1 = poly.fit_transform(X_1)

    model_0 = LinearRegression().fit(X_poly_0, y_0)
    model_1 = LinearRegression().fit(X_poly_1, y_1)

    x_range = np.linspace(X.min(), X.max(), 200).reshape(-1, 1)
    y_pred_0 = model_0.predict(poly.transform(x_range))
    y_pred_1 = model_1.predict(poly.transform(x_range))


    fig_lines, ax_lines = plt.subplots()
    ax_lines.scatter(employee_data_extra['Training Hours'], employee_data_extra['Production'], color="black", s=100, label='Original Data')
    ax_lines.scatter(employee_data_new['Training Hours'], employee_data_new['Production'], color="maroon", s=100, label='New Data')

    ax_lines.plot(x_range, y_pred_0, linestyle='--', color='blue', linewidth=2.5, label="Original Line")
    ax_lines.plot(x_range, y_pred_1, linestyle='--', color='red', linewidth=2.5, label="New Line")

    ax_lines.set_xlabel("Training Hours")
    ax_lines.set_ylabel("Production")
    ax_lines.set_title(f"Degree {degree_low} Polynomial Fit")
    ax_lines.legend()
    st.pyplot(fig_lines)

quiz_card(
    question_title="2.2.1 Checkpoint",
    question_text="""If we fit a very complex model to a sample dataset $A$,
    then fit a model of the same complexity to a different sample $B$ from
    the same population as $A$, how will the models *likely* compare?  
    """,
    options=['They will look very different from each other visually, and have very different coefficients.', 
             "They will look the same, since they are of the same complexity.", 
             "They will both be ugly.",
             "They will be invisible."],
    key="check221",
    correct_answer="They will look very different from each other visually, and have very different coefficients.",
    correct_feedback="""Correct! High complexity means high variance, i.e.
    the curve of best fit will change drastically when we use different data.
    """,
    incorrect_feedback="Try again. And don't call the models ugly!"
)
#feature_names = poly.get_feature_names_out(['Training Hours'])
#coef_table = pd.DataFrame({
  #  "Feature": ["Intercept"] + list(feature_names),
 #   "Coefficient": [model.intercept_] + list(model.coef_)
#})

# R^2 score
#r2 = model.score(X_poly, y)

#with st.expander(f"Polynomial Regression Coefficients (degree={degree})"):
 #   st.dataframe(coef_table, width='stretch')
  #  st.write(f"**R²:** {r2:.3f}")

st.markdown("---")
st.markdown(f"""
            ### 3. Evaluation of Model Performance 
            In order to identify when a model is overfitting or 
            underfitting, there many standard tools. Two which we will
            briefly introduce here before exploring further
            in future demos are {highlight("residual plots")} and
            {highlight("train-test splits")}.

            #### 3.1 Plotting Residuals

            A {highlight("residual")} is a fancy term for the error
            associated with a particular prediction made by our model.
            Here, given a new employee receives $x$ hours of training,
            denote the *predicted production of that employee according
            to our model* by $p_{{pred}}(x)$. The residual or error associated
            with this prediction is $$err(x) = p_{{pred}}(x) - p(x)$$, where 
            $p(x)$ is the actual production associated with that employee.

            Notice: we can treat error as a function of $x$ too, so we
            can produce a (scatter) plot of $err(x)$, the error associated with each
            prediction! This is a residual plot, which is most useful for
            identifying an underfit model. If the residual plot
            exhibits a noticeable trend rather than being centered around
            0, i.e. the $x$-axis, then the model has failed to capture 
            something besides noise. For a visual example of this,
            see the plots below which extend the linear model of 2.2
            to a larger sample size.

            """, unsafe_allow_html=True)
employee_data_big = generate_employee_data(n_total=400)
X_big = employee_data_big[['Training Hours']]
y_big = employee_data_big['Production']
degree_low = 1
poly = PolynomialFeatures(degree=degree_low, include_bias=False)
X_poly_big = poly.fit_transform(X_big)

model_big = LinearRegression().fit(X_poly_big, y_big)

x_range = np.linspace(X_big.min(), X_big.max(), 400).reshape(-1, 1)
y_pred = model_big.predict(poly.transform(x_range))
residuals = y_pred - y_big

training_hours_range = np.arange(start=np.min(X_big['Training Hours']),
                             stop=np.max(X_big['Training Hours']))


fig_quad_b, ax_quad_b = plt.subplots(nrows=2, ncols=1, sharex=True)
ax_quad_b[0].scatter(X_big, y_big, color="black")
ax_quad_b[1].scatter(X_big, residuals, color="black")
ax_quad_b[1].set_ylabel("Residual")
ax_quad_b[0].plot(x_range, y_pred, linestyle='--', color='red', linewidth=2.5, label="Best Fit Line")
ax_quad_b[0].set_xlabel("Training Hours")
ax_quad_b[0].set_ylabel("Production")
ax_quad_b[0].legend()
ax_quad_b[0].set_title(f"Degree {degree_low} Polynomial Fit and Residuals")
st.pyplot(fig_quad_b)

st.markdown(f"""
            Clearly the residuals have a trend which results from the
            fact that our linear model doesn't capture the initial 
            steepness and subsequent leveling off of the production curve.
            
            
            #### 3.2 Train-Test Split
            Something which was briefly mentioned in section 2.1 
            is the need for a model which generalizes well to new
            data. In real-world applications, a company will want
            a model which has been trained on old data to be used
            in the future on new data. This means that strong
            performance on data used for training is **not** the most important
            measure of model accuracy. One way to simulate performance
            on unseen data is to {highlight("split the available data into" \
            "a training set and a test set")}. After using *only* the training
            data to fit the curve, we use the model to make predictions 
            on the test set. If the model makes very small errors on the 
            training set (fits the training data closely) but large errors
            on the test set, the model is likely overfitting. This is
             exemplified by the Model Performance on New Data expander in
            section 2.1.1.
             """, unsafe_allow_html=True)
st.markdown("---")

st.markdown("""
            ### 4. An Acceptable Model?
            To select a model with the appropriate level of complexity requires us
            to go through a full workflow, which we will not do here.
            For completeness, plots for quadratic and cubic polynomial
            models are shown below. Which one fits the original set of
            ten data points best? Which generalizes better? You be the
            judge!

            """)
with st.expander("Quadratic Model"):
    with st.expander("Original Training Data"):
        st.markdown("""This plot shows the quadratic curve of
                    best fit plotted with the original ten data
                    points on which it was trained.""")
        X = employee_data_extra[['Training Hours']]
        y = employee_data_extra['Production']
        degree_quad = 2
        poly = PolynomialFeatures(degree=degree_quad, include_bias=False)
        X_quad = poly.fit_transform(X)

        model = LinearRegression().fit(X_quad, y)

        x_range = np.linspace(X.min(), X.max(), 200).reshape(-1, 1)
        y_pred = model.predict(poly.transform(x_range))

        training_hours_range = np.arange(start=np.min(employee_data_extra['Training Hours']),
                                    stop=np.max(employee_data_extra['Training Hours']))


        fig_quad, ax_quad = plt.subplots()
        ax_quad.scatter(employee_data_extra['Training Hours'], employee_data_extra['Production'], color="black")
        ax_quad.plot(x_range, y_pred, linestyle='--', color='blue', linewidth=2)
        ax_quad.set_xlabel("Training Hours")
        ax_quad.set_ylabel("Production")
        ax_quad.set_title(f"Degree {degree_quad} Polynomial Fit")
        st.pyplot(fig_quad)
    
    with st.expander("Plotted with New Data"):
        st.markdown("""This plot shows the quadratic curve of
                    best fit from above, but plotted with the new
                    ten data points.""")
        X = employee_data_extra[['Training Hours']]
        y = employee_data_extra['Production']
        degree_quad = 2
        poly = PolynomialFeatures(degree=degree_quad, include_bias=False)
        X_quad = poly.fit_transform(X)

        model = LinearRegression().fit(X_quad, y)

        x_range = np.linspace(X.min(), X.max(), 200).reshape(-1, 1)
        y_pred = model.predict(poly.transform(x_range))

        training_hours_range = np.arange(start=np.min(employee_data_extra['Training Hours']),
                                    stop=np.max(employee_data_extra['Training Hours']))


        fig_quad, ax_quad = plt.subplots()
        ax_quad.scatter(employee_data_new['Training Hours'], employee_data_new['Production'], color="black")
        ax_quad.plot(x_range, y_pred, linestyle='--', color='blue', linewidth=2)
        ax_quad.set_xlabel("Training Hours")
        ax_quad.set_ylabel("Production")
        ax_quad.set_title(f"Degree {degree_quad} Polynomial Fit")
        st.pyplot(fig_quad)

with st.expander("Cubic Model"):
    with st.expander("Original Training Data"):
        st.markdown("""This plot shows the cubic curve of
                    best fit plotted with the original ten data
                    points on which it was trained.""")
        X = employee_data_extra[['Training Hours']]
        y = employee_data_extra['Production']
        degree_quad = 3
        poly = PolynomialFeatures(degree=degree_quad, include_bias=False)
        X_quad = poly.fit_transform(X)

        model = LinearRegression().fit(X_quad, y)

        x_range = np.linspace(X.min(), X.max(), 200).reshape(-1, 1)
        y_pred = model.predict(poly.transform(x_range))

        training_hours_range = np.arange(start=np.min(employee_data_extra['Training Hours']),
                                    stop=np.max(employee_data_extra['Training Hours']))


        fig_quad, ax_quad = plt.subplots()
        ax_quad.scatter(employee_data_extra['Training Hours'], employee_data_extra['Production'], color="black")
        ax_quad.plot(x_range, y_pred, linestyle='--', color='blue', linewidth=2)
        ax_quad.set_xlabel("Training Hours")
        ax_quad.set_ylabel("Production")
        ax_quad.set_title(f"Degree {degree_quad} Polynomial Fit")
        st.pyplot(fig_quad)

    with st.expander("Plotted with New Data"):
        st.markdown("""This plot shows the cubic curve of
                    best fit from above, but plotted with the
                    ten new data points.""")
        X = employee_data_extra[['Training Hours']]
        y = employee_data_extra['Production']
        degree_quad = 3
        poly = PolynomialFeatures(degree=degree_quad, include_bias=False)
        X_quad = poly.fit_transform(X)

        model = LinearRegression().fit(X_quad, y)

        x_range = np.linspace(X.min(), X.max(), 200).reshape(-1, 1)
        y_pred = model.predict(poly.transform(x_range))

        training_hours_range = np.arange(start=np.min(employee_data_extra['Training Hours']),
                                    stop=np.max(employee_data_extra['Training Hours']))


        fig_quad, ax_quad = plt.subplots()
        ax_quad.scatter(employee_data_new['Training Hours'], employee_data_new['Production'], color="black")
        ax_quad.plot(x_range, y_pred, linestyle='--', color='blue', linewidth=2)
        ax_quad.set_xlabel("Training Hours")
        ax_quad.set_ylabel("Production")
        ax_quad.set_title(f"Degree {degree_quad} Polynomial Fit")
        st.pyplot(fig_quad)
st.markdown("---")
st.markdown("### Looking Forward")
st.markdown(
    """
Thank you for reading this demo! Model generalizability and finding
a balance between overfitting/underfitting is a
central tenet of data science. In the future, we will explore
model selection in greater detail, as well as a new type of
problem: classification.


If you have any questions or comments, 
please enter them using the form below.
    """
    )


with st.form("comment_form"):
    name = st.text_input("Name (optional)")
    comment = st.text_area("Comments, questions, or suggestions for future topics:")
    submit_comment = st.form_submit_button("Submit")
    if submit_comment and comment.strip():
        save_comment(name.strip(), comment.strip())
        st.success("Comment submitted!")