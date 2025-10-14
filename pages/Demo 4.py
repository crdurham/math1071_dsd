import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
from matplotlib.ticker import ScalarFormatter
from pathlib import Path
from datetime import datetime
from db import save_comment
from utils import lin_reg
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression, LogisticRegression

ROOT = Path(__file__).resolve().parent.parent
HOUSING_DATA_FILE = ROOT / "data" / "Housing.csv"

def highlight(text, color="#00cc6633"):
    """Return text string wrapped in HTML span with background highlight."""
    return f'<span style="background-color:{color}; padding:2px 4px; border-radius:3px"><b>{text}</b></span>'

def generate_subscription_data(n_total=100, seed_choice=1):
    seed=seed_choice
    rng = np.random.default_rng(seed)
    time_spent = rng.normal(400, 600, n_total).clip(0,1100)
    status = []
    for x in time_spent:
        p_x = 1/(1+np.exp(-(x -450 + rng.normal(0,130))))
        status_x = random.choices([0,1], weights=[1-p_x, p_x])[0]
        status.append(status_x)
    df = pd.DataFrame({
        "Time": time_spent,
        "Subscription Status": status 
    })

    return df

def generate_age_data(X, seed_choice=1):
    seed = seed_choice
    rng = np.random.default_rng(seed)
    ages = []
    for x in X:
        x_z = (x-np.mean(X))/np.std(X)
        age_range = [5*x_z**2-15*x_z +20 +rng.uniform(25,35),
        3*x_z**2-1*x_z + 15+rng.uniform(30,40)]
        age = rng.normal((min(age_range)+max(age_range))/2, 10) 
        ages.append(age)
    return ages

def threshold(x, thresh=0.5):
    return (x >= thresh).astype(int)

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

st.title("Demo 4: Classification", width="content")

st.subheader("Goals:")
st.markdown(
    f"""
    1. Understand the difference between {highlight("regression")} and {highlight("classification")}.
    2. Be able to calculate particular output values from the {highlight("sigmoid")} or {highlight("logistic")} function.
    3. Interpret probabilities associated with predictions made by logistic regression.

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
            ### 0. Motivating Example: Predicting Subscription Cancellation
            Subscription services have become an all-pervasive fact of life. Commodities of all shapes,
            sizes, modes, and mediums are sold in this fashion by almost every major business we
            can think of. That being said, it is very important for businesses to be able to
            **predict which customers will cancel their subscribtions**.

            Suppose we are working for the data analytics team at Netflix. We are given
            a sample dataset with **100 customer data points**, each with the following information:
            - Total **active time spent using the app during a 3 month period**.
            - **Whether or not the subscription was cancelled during the next 3 months**, recorded
            as a 1 if it remained active, 0 if it was stopped.

            Our task is to **predict whether or not a customer will cancel their
            subscription based on their activity level**.
            """)
st.markdown("---")
st.markdown(f"""
            ### 1. Classification vs Regression
            The problem at hand is of a different type than we have seen
            in the previous three demos. To see why, consider the brief
            descriptions provided below to summarize the setup and objective from each
            project.
            1. Input(s) or Feature(s) $x$: house size in square feet, $0<x<\infty$.\\
            Output or Label: house price $p(x)$ in dollars, $0<p(x)<\infty$.
            2. Input(s)or Feature(s) $x$: average weekly patrons, average book price,
            money spent on sectors, $0<x<\infty$.\\
            Output or Label: total revenue $R(x)$, $0<R(x)<\infty$.
            3. Input(s) or Feature(s) $x$: employee training time, $0<x<\infty$.\\
            Output or Label: production $P(x)$, $0<P(x)<\infty$.
            4. Input(s) or Feature(s) $x$: minutes of activity, $0<x<\infty$.\\
            Output or Label: subscription cancelled ($y=0$) or not cancelled ($y=1$).

            Above, any variable $z$ with $0<z<\infty$ which can take values in a continuous
            spectrum is called {highlight("continuous")},
            meaning it can take on any real number in the specified interval. Here the output $y$ indicating 
            subscription status of
            a Netflix customer *doesn't* take on a continuous spectrum of
            values! This distinction is important enough to get its own definition.
            """, unsafe_allow_html=True)

with st.container(border=True):
    st.markdown("""<h5><u>Definition:</u></h5>""", unsafe_allow_html=True)
    st.markdown(
        f"""
        A function or random variable $y$ which can only take on a finite set of outputs
        is called a {highlight("discrete or categorical variable")}.
        """, unsafe_allow_html=True
    )

st.markdown(f"""
            As indicated above, we denote the customer screen time by $x$, and denote their
            subscription status at the end of the next 3 months by $y=y(x)$. The only possibilities
            are $y(x)=1$, meaning the subscription remained active,
             or $y(x)=0$, meaning the subscription was discontinued.
            """, unsafe_allow_html=True)

quiz_card(
    question_title="1.1 Checkpoint",
    question_text="""In the real world, is it possible that $y(100)=0$ *and*
    $y(100)=1$ for two distinct accounts with $x=100$?  
    """,
    options=["Couldn't tell ya.", 
             "No, that would mean y is not well-defined.", 
             "Yes, y is non-deterministic so it can take on different outputs given the same input.",
             "Math, man, I tell ya."],
    key="check11",
    correct_answer="Yes, y is non-deterministic so it can take on different outputs given the same input.",
    correct_feedback="""Correct! Whether or not a customer will cancel their
    subscription is a random or non-deterministic event.
    """,
    incorrect_feedback="Try again, keep grinding."
)

st.markdown(f"""
            In data science, when our goal is to predict the output of
            a continuous variable we say we are solving a 
            {highlight("regression problem")}. If on the other hand we want to
            predict the output of a categorical variable (as in this instance), we are solving a
            {highlight("classification problem")}. Some canonical 
            examples of classification problems are:
            - Categorizing emails as spam/not spam (often written as
            spam vs ham).
            - Identifying medical condition of emergency room arrivals
            based on measured symptoms.
            - Predicting whether or not a credit card holder will default
            on their payment.

            In each instance, the output we want to predict can only be
            one of finitely many classes. Some examples of
            regression problems are:
            - Based on the size $x$ of a house, predict the price $P(x)$
            - Predict the exam score a student will receive based on amount
            of time they spend studying.
            - Predict car gas mileage based on vehicle age
            """, unsafe_allow_html=True)
st.markdown("---")
data = generate_subscription_data()

age_data = generate_age_data(data['Time'])
data_aug = pd.DataFrame({
    'Time': data['Time'],
    'Age': age_data,
    'Status': data['Subscription Status']
})

st.markdown(f"""
            ### 2. Getting to Know the Data

            Below, we can see the data plotted. Any point represented by a
            red x indicates a subscription which was cancelled, while the black
            x's indicate an ongoing service. 
            """, unsafe_allow_html=True)
with st.expander("Plot of Data"):
    fig, ax = plt.subplots()
    color_map = {0:'red', 1:'black'}
    label_map = {0: 'Cancelled', 1: 'Active'}
    colors = [color_map[stat] for stat in data["Subscription Status"]]
    for status in data['Subscription Status'].unique():
        subset = data[data['Subscription Status'] == status]
        ax.scatter(
            subset['Time'],
            subset['Subscription Status'],
            c=color_map[status],
            marker='x',
            s=25,
            label=label_map[status] 
        )
    ax.set_xlabel("Time (Minutes)")
    ax.set_ylabel("Subscription Status")
    ax.set_title("Subscription Status by Activity Level")
    ax.legend() 
    st.pyplot(fig)

st.markdown("""
            As expected, customers who watch more shows and movies
            are more likely to maintain their account. For this reason, the data looks like two groups;
            one clump is at $y=0$ (cancellations), while the other is at $y=1$
            (continued subscriptions).
            
            In a setting where our data falls into two classes or categories,
            it is useful to calculate relevant statistics within each
            category rather than for the whole dataset. This will
            allows us to numerically distinguish the two classes. In particular,
            note how the average active time spent by a customer in the
            cancellation class compares to that of the average customer in the
            active class.
            """)

with st.expander("Statistics for Each Class"):
    with st.expander("Cancelled"):
        st.write(data[data['Subscription Status'] == 0].describe()['Time'])
    with st.expander("Continued"):
        st.write(data[data['Subscription Status'] == 1].describe()['Time'])
quiz_card(
    question_title="2.1 Checkpoint",
    question_text="""Among customers who ended their subscriptions,
    what was the most time spent using Netflix (in minutes)? 
    """,
    options=["1000", 
             "750.78", 
             "0",
             "549.61"],
    key="check21",
    correct_answer="549.61",
    correct_feedback="""Correct! Under 'Cancelled' we look at the row
    labeled 'max,' and it shows 634.1802. After slight rounding, we obtain
    the answer.
    """,
    incorrect_feedback="Incorrect. Remember we want the *max* time among" \
    " *Cancelled* subscriptions."
)
st.markdown("---") 
st.markdown("""
            ### 3. Fitting a Model
            As with any labeled dataset, our goal is to construct a
            model which takes as input the *feature(s)* of the data
            and predicts the *label* for that input. In this scenario,
            our one feature is the active minutes spent by a user on the
            app or site, while the label we want to predict is 0 for
            a cancelled subscription and 1 otherwise. We will see
            that predicting a discrete output will require a different approach
            than predicting continuous outputs in previous demos.

            #### 3.1 First Idea: Linear Regression

            At first glance, it is natural to try and use a linear model
            to fit the data (see section 2 of Demo 2 for reference). After all,
            a line can be made which will pass through the two clumps of data.
            """)
data1 = data.copy()

X = data[['Time']]
y = data['Subscription Status']
degree = 1
poly = PolynomialFeatures(degree=degree, include_bias=False)
X_poly = poly.fit_transform(X)

model = LinearRegression().fit(X_poly, y)

x_range = np.linspace(X.min(), X.max(), 200).reshape(-1, 1)
y_pred = model.predict(poly.transform(x_range))

training_hours_range = np.arange(start=np.min(data['Time']),
                             stop=np.max(data['Time']))


fig1, ax1 = plt.subplots()
color_map = {0:'red', 1:'black'}
label_map = {0: 'Cancelled', 1: 'Active'}
colors = [color_map[stat] for stat in data["Subscription Status"]]
for status in data['Subscription Status'].unique():
    subset = data[data['Subscription Status'] == status]
    ax1.scatter(
        subset['Time'],
        subset['Subscription Status'],
        c=color_map[status],
        marker='x',
        s=25,
        label=label_map[status] 
    )
ax1.set_xlabel("Time (Minutes)")
ax1.set_ylabel("Subscription Status")
ax1.set_title("Subscription Status by Activity Level")
ax1.plot(x_range, y_pred, linestyle='--', color='blue', linewidth=2, label='Linear Fit')
ax1.set_xlabel("Time (Minutes)")
ax1.set_ylabel("Subscription Status")
ax1.set_title(f"Degree {degree} Polynomial Fit")
ax1.legend() 
st.pyplot(fig1)

pred = pd.DataFrame({"Time":data['Time'],
                    "Line_Status": model.predict(data[['Time']])})
st.markdown(f"""
            Pictured above is the line of best fit
            for our dataset, constructed to **minimize the average
            squared error**. It has the equation

            $$y_L = 0.0012x - 0.0529.$$

             Note that some values of $x$ occur multiple
            times, meaning that some data points get plotted in exactly the
            same place; the line does pass through the center of the data, though
            we are unable to see the regions of **high density**. There are
            a couple of obvious drawbacks which we list below.

            1. **Predicted values which are not 0 or 1**: since the
            model is a line which is not constant 1 or 0, it takes on
            all real numbers. This does not match our scenario, where
            the only possibilities are $y=0$ or $y=1$. """, unsafe_allow_html=True)
with st.expander("Predicted Status via Linear Model"):
    st.write(pred)
st.markdown("""
            2. **Does not sense the jump in data**: the linear model
            identifies where the data is clustered, but does not match
            the transition or jump which occurs.

            #### 3.2 Second Idea: Modify via Threshold

            One natural way to try and resolve both issues above
            is by instituting a **threshold**. To be precise,
            we will compose our linear model $y_L$ with the following piecewise
            function""", unsafe_allow_html=True)

st.latex(r"""
p(x) = 
\begin{cases} 
0, & x < 0.5 \\
1, & x \ge 0.5
\end{cases}.
""")
st.markdown("""
In other words, our new model will output 0 if the original
linear model was below 0.5, and will output 1 if the linear model
was at least as large as 0.5. As an equation,
""")
st.latex(r"""
y_{thresh}(x) = p(y_L(x)).
""")

y_thresh_pred = threshold(np.array(pred['Line_Status']))
st.markdown(f"""
Now $y_{{thresh}}$ will only output 0 or 1, and we've forced our model
to make a jump according to how large the linear model is. With the current
dataset, this works reasonably well, as can be seen in the plot below. The
vertical dividing line between the red and black regions is called
the {highlight("decision boundary")} of the model. In this case,
it occurs at $x=460.75$. This line contains tells us all we need
to know as far as model predictions; if $x<460.75$, $y_{{thresh}}(x)=0$,
while if $x\geq 460.75$ then $y_{{thresh}}(x)=1$.


""", unsafe_allow_html=True)      

with st.expander("Plot of Data with Model Prediction"):
    fig2, ax2 = plt.subplots()
    color_map = {0: 'red', 1: 'black'}
    label_map = {0: 'Cancelled', 1: 'Active'}

    # Plot data
    for status in data['Subscription Status'].unique():
        subset = data[data['Subscription Status'] == status]
        ax2.scatter(
            subset['Time'],
            subset['Subscription Status'],
            c=color_map[status],
            marker='x',
            s=25,
            label=label_map[status]
        )

    # Plot linear fit
    ax2.plot(x_range, y_pred, linestyle='--', color='blue', linewidth=2, label='Linear Fit')

    # Plot threshold line
    ax2.axhline(0.5, color='gray', linestyle=':', label='Threshold (0.5)')

    # Shade predicted regions
    ax2.fill_between(
        x_range.flatten(),
        -0.2, 1.2,
        where=(y_pred >= 0.5),
        color='black', alpha=0.15, label='Predicted Active'
    )
    ax2.fill_between(
        x_range.flatten(),
        -0.2, 1.2,
        where=(y_pred < 0.5),
        color='red', alpha=0.15, label='Predicted Cancelled'
    )

    ax2.set_xlabel("Time (Minutes)")
    ax2.set_ylabel("Subscription Status")
    ax2.set_title("Thresholded Linear Model Predictions")
    ax2.legend(loc='lower right')
    st.pyplot(fig2)
st.markdown("""
##### 3.2.1 A Cause for Pause
We can see that all black x's which lie in
the gray shaded region are correctly classified, and so
are all red x's which lie in the red region. Visually,
this shows our model is performing quite well
on this dataset; however, there is a caveat which
will show us that this model may not be so great
in general. Consider the new dataset below, which is the
same as the original set but with three new accounts. The new
data points have $x=6500, x=7300,$ $x=8000$, and all
have been maintained ($y=1$).
""")

new_points = pd.DataFrame({"Time": [7300, 6500, 8000],
                        "Subscription Status": [1,1,1]})
new_data = pd.concat([data, new_points], ignore_index=True)
with st.expander("Plot of New Data"):
    fig3, ax3 = plt.subplots()
    color_map = {0:'red', 1:'black'}
    label_map = {0: 'Cancelled', 1: 'Active'}
    colors = [color_map[stat] for stat in data["Subscription Status"]]
    for status in new_data['Subscription Status'].unique():
        subset = new_data[new_data['Subscription Status'] == status]
        ax3.scatter(
            subset['Time'],
            subset['Subscription Status'],
            c=color_map[status],
            marker='x',
            s=25,
            label=label_map[status] 
        )
    ax3.set_xlabel("Time (Minutes)")
    ax3.set_ylabel("Subscription Status")
    ax3.set_title("Subscription Status by Activity Level")
    ax3.legend() 
    st.pyplot(fig3)

st.markdown("""
Intuitively, we would expect the output of the previous
model to be the same on this new set; however, **the presence
of outliers will greatly impact the line of best fit**. In
a classification scenario such as the above, this should
not happen. We see that the predictive accuracy of our model
has been greatly damaged.
""")
with st.expander("New Line of Best Fit"):
    X_outlier = new_data[['Time']]
    y_outlier = new_data['Subscription Status']
    degree = 1
    poly = PolynomialFeatures(degree=degree, include_bias=False)
    X_poly_outlier = poly.fit_transform(X_outlier)

    model_outlier = LinearRegression().fit(X_poly_outlier, y_outlier)

    x_range_outlier = np.linspace(X_outlier.min(), X_outlier.max(), 200).reshape(-1, 1)
    y_pred_outlier = model_outlier.predict(poly.transform(x_range_outlier))

    training_hours_range = np.arange(start=np.min(new_data['Time']),
                                stop=np.max(new_data['Time']))


    fig4, ax4 = plt.subplots()
    color_map = {0:'red', 1:'black'}
    label_map = {0: 'Cancelled', 1: 'Active'}
    colors = [color_map[stat] for stat in data["Subscription Status"]]
    for status in new_data['Subscription Status'].unique():
        subset = new_data[new_data['Subscription Status'] == status]
        ax4.scatter(
            subset['Time'],
            subset['Subscription Status'],
            c=color_map[status],
            marker='x',
            s=25,
            label=label_map[status] 
        )
    ax4.set_xlabel("Time (Minutes)")
    ax4.set_ylabel("Subscription Status")
    ax4.set_title("Subscription Status by Activity Level")
    ax4.plot(x_range_outlier, y_pred_outlier, linestyle='--', color='blue', linewidth=2, label='New Linear Fit')
    ax4.set_xlabel("Time (Minutes)")
    ax4.set_ylabel("Subscription Status")
    ax4.set_title(f"Degree {degree} Polynomial Fit")
    ax4.legend() 
    st.pyplot(fig4)

with st.expander("New Decision Boundary"):
    fig5, ax5 = plt.subplots()
    color_map = {0: 'red', 1: 'black'}
    label_map = {0: 'Cancelled', 1: 'Active'}

    # Plot data
    for status in new_data['Subscription Status'].unique():
        subset = new_data[new_data['Subscription Status'] == status]
        ax5.scatter(
            subset['Time'],
            subset['Subscription Status'],
            c=color_map[status],
            marker='x',
            s=25,
            label=label_map[status]
        )

    # Plot linear fit
    ax5.plot(x_range_outlier, y_pred_outlier, linestyle='--', color='blue', linewidth=2, label='New Linear Fit')

    # Plot threshold line
    ax5.axhline(0.5, color='gray', linestyle=':', label='Threshold (0.5)')

    # Shade predicted regions
    ax5.fill_between(
        x_range_outlier.flatten(),
        -0.2, 1.7,
        where=(y_pred_outlier >= 0.5),
        color='black', alpha=0.15, label='Predicted Active'
    )
    ax5.fill_between(
        x_range_outlier.flatten(),
        -0.2, 1.7,
        where=(y_pred_outlier < 0.5),
        color='red', alpha=0.15, label='Predicted Cancelled'
    )

    ax5.set_xlabel("Time (Minutes)")
    ax5.set_ylabel("Subscription Status")
    ax5.set_title("Thresholded Linear Model Predictions")
    ax5.legend(loc='lower right')
    st.pyplot(fig5)

st.markdown("""
The line of best fit is influenced by the three
outliers, but unfortunately this causes it to predict a
larger portion of the $y=1$ (active subscription) class
incorrectly. Clearly this should not be the case!
""")
quiz_card(
    question_title="3.2.1 Checkpoint",
    question_text="""If $x_0$ falls on the left side of the 
    decision boundary and $x_1$ falls on the right side,
    what can we say about their predicted classes?  
    """,
    options=["(a) x_0 is predicted to be in class 0 (cancelled subscription).", 
             "(b) It is undecided.", 
             "(c) x_1 is predicted to be in class 1 (retained subscription).",
             "Both (a) and (c)."],
    key="check321",
    correct_answer="Both (a) and (c).",
    correct_feedback="""Correct! The decision boundary separates (or tries
    to separate) the data into the respective classes.
    """,
    incorrect_feedback="Incorrect, try again."
)
st.markdown("---") 
st.markdown(f"""
### 4. Logistic Regression
We have only considered using a linear function $y_L$ composed with a jump function to create a threshold
to model our data. There is another type of function which
can help us model categorical data in a more smooth, natural way: the {highlight("sigmoid")}
or {highlight("logistic")} function, defined by
""", unsafe_allow_html=True)

st.latex(r"""
g(x) = \frac{1}{1+e^{-x}}.
""")

st.markdown(f"""
More generally, for real numbers $m,b$ we can plot the
{highlight("shifted sigmoid")}
""", unsafe_allow_html=True)
st.latex(r"""
g_{m,b}(x) = \frac{1}{1+e^{-(mx+b)}}.
""")

m = st.slider(label="Set $m$:", min_value=-5.0, max_value=5.0, step=0.1, value=1.0)
b = st.slider(label="Set $b$:", min_value=-10, max_value=10, step=1, value=1)
x_range = np.linspace(-40,40,4000)
p_0 = 1/(1+np.exp(-(m*x_range+b)))

fig6, ax6 = plt.subplots()
#ax6.scatter(data['Time'], data['Subscription Status'], color="purple")
ax6.plot(x_range, p_0, color='black', linewidth=3)
ax6.set_xlabel("x")
ax6.set_ylabel("y=g(mx+b)")
ax6.set_title("Visualizing Sigmoid Properties")

ax6.yaxis.set_major_formatter(ScalarFormatter())
ax6.ticklabel_format(style='plain', axis='y')

st.pyplot(fig6)

st.markdown(f"""
A couple of key properties which are important to
us, assuming $m > 0$:

1. $g_{{m,b}}\\left(-\\frac{{b}}{{m}}\\right)=\\frac{1}{2}$.
2. For any $x$, $0<g_{{m,b}}(x)<1$.
3. $\lim_{{x\\to -\infty}} g_{{m,b}}(x)=0$, $\lim_{{x\\to \infty}} g_{{m,b}}(x)=1$

This gives us the following new option: instead of fitting
a linear function $y_L(x)=mx+b$ to our data by minimizing the mean squared
error cost function, fit the
shifted sigmoid function to the data by maximizing the {highlight("log-likelihood"\
" of the labels occuring")}; this is a different optimization problem which we will
discuss in the future.

From the properties listed above, we {highlight("can interpret"\
" $g_{{m,b}}(x)$ as the probability $x$ is in class 1")}. Moreover,
$x=-\\frac{{b}}{{m}}$ is the decision boundary of this model because
it is the input for which $g_{{m,b}}=\\frac{1}{2}$, i.e. the
data point is equally likely to be in either class!
This is known as {highlight("logistic regression")}. 
""", unsafe_allow_html=True)

poly = PolynomialFeatures(degree=degree, include_bias=False)
X_poly_original = poly.fit_transform(X)

model_logr = LogisticRegression().fit(X_poly_original, y)

x_range = np.linspace(X.min(), X.max(), 200).reshape(-1, 1)
y_pred_logr = model_logr.predict(poly.transform(x_range))
y_prob = model_logr.predict_proba(poly.transform(x_range))[:, 1]


time_range = np.arange(start=np.min(data['Time']),
                             stop=np.max(data['Time']))

with st.expander("Model Prediction for Original Dataset"):
    fig7, ax7 = plt.subplots()
    color_map = {0: 'red', 1: 'black'}
    label_map = {0: 'Cancelled', 1: 'Active'}

    # Plot data
    for status in data['Subscription Status'].unique():
        subset = data[data['Subscription Status'] == status]
        ax7.scatter(
            subset['Time'],
            subset['Subscription Status'],
            c=color_map[status],
            marker='x',
            s=25,
            label=label_map[status]
        )

    # Plot logistic regression fit
    ax7.plot(x_range, y_prob, linestyle='--', color='blue', linewidth=2, label='Logistic Model')

    # Plot threshold line
    ax7.axhline(0.5, color='gray', linestyle=':', label='Threshold (0.5)')
    boundary_x = -model_logr.intercept_[0] / model_logr.coef_[0][0]
    ax7.axvline(boundary_x, color='green', linestyle='--', label=f'Decision boundary ({boundary_x:0.2f})')

    # Shade predicted regions
    ax7.fill_between(
        x_range.flatten(),
        -0.2, 1.2,
        where=(y_pred_logr >= 0.5),
        color='black', alpha=0.15, #label='Predicted Active'
    )
    ax7.fill_between(
        x_range.flatten(),
        -0.2, 1.2,
        where=(y_pred_logr < 0.5),
        color='red', alpha=0.15, #label='Predicted Cancelled'
    )

    ax7.set_xlabel("Time (Minutes)")
    ax7.set_ylabel("Subscription Status")
    ax7.set_title("Thresholded Logistic Regression Predictions")
    ax7.legend(loc='lower right')
    st.pyplot(fig7)

st.markdown("""
The resulting
logistic function obtained by fitting the original dataset is
""")
#lr_coef = model_logr.coef_
#lr_intercept = model_logr.intercept_
#st.markdown(f"{lr_coef}")
st.latex(r"""
y_{lr}(x) = \frac{1}{1+e^{-(0.0169859x-7.72240282)}}.
""")

st.markdown("""
One major advantage this has over our previous model: it is
less sensitive to outliers. This is because the log-likelihood
cost function grows *logarithmically*, which is much slower
than the growth of the mean squared error, which is quadratic.
""")

poly = PolynomialFeatures(degree=degree, include_bias=False)
new_data_poly = poly.fit_transform(X_outlier)

model_logr_new = LogisticRegression().fit(new_data_poly, y_outlier)

x_range_new = np.linspace(X_outlier.min(), X_outlier.max(), 200).reshape(-1, 1)
y_pred_logr_new = model_logr_new.predict(poly.transform(x_range_new))
y_prob_new = model_logr_new.predict_proba(poly.transform(x_range_new))[:, 1]


time_range_new = np.arange(start=np.min(new_data['Time']),
                             stop=np.max(new_data['Time']))

with st.expander("Model Prediction for Dataset with Outliers"):
    fig8, ax8 = plt.subplots()
    color_map = {0: 'red', 1: 'black'}
    label_map = {0: 'Cancelled', 1: 'Active'}

    # Plot data
    for status in new_data['Subscription Status'].unique():
        subset = new_data[new_data['Subscription Status'] == status]
        ax8.scatter(
            subset['Time'],
            subset['Subscription Status'],
            c=color_map[status],
            marker='x',
            s=25,
            label=label_map[status]
        )

    # Plot logistic regression fit
    ax8.plot(x_range_new, y_prob_new, linestyle='--', color='blue', linewidth=2, label='New Logistic Model')

    # Plot threshold line
    ax8.axhline(0.5, color='gray', linestyle=':', label='Threshold (0.5)')
    boundary_x_new = -model_logr_new.intercept_[0] / model_logr_new.coef_[0][0]
    ax8.axvline(boundary_x_new, color='green', linestyle='--', label=f'Decision boundary ({boundary_x_new:0.2f})')

    # Shade predicted regions
    ax8.fill_between(
        x_range_new.flatten(),
        -0.2, 1.2,
        where=(y_pred_logr_new >= 0.5),
        color='black', alpha=0.15, #label='Predicted Active'
    )
    ax8.fill_between(
        x_range_new.flatten(),
        -0.2, 1.2,
        where=(y_pred_logr_new < 0.5),
        color='red', alpha=0.15, #label='Predicted Cancelled'
    )

    ax8.set_xlabel("Time (Minutes)")
    ax8.set_ylabel("Subscription Status")
    ax8.set_title("Thresholded Logistic Regression Predictions")
    ax8.legend(loc='lower right')
    st.pyplot(fig8)

st.markdown("""Notice: our model has maintained the same
level of performance after adding outliers. Indeed (to 1 decimal
place), the decision boundary is identical to before.""")

quiz_card(
    question_title="4.1 Checkpoint",
    question_text="""Why is the logistic regression model, fit by maximizing
    the log-likelihood, less sensitive to outliers than $y_{thresh}$, which
    was created by fitting a line which minimized average squared error
    and then composing with a jump function?
    """,
    options=["The log-likelihood function *does'nt* penalize outliers as harshly as mean squared error.", 
             "Because the logistic function is continuous.", 
             "Hard tellin' not knowin'.",
             "Neither model is sensitive to outliers."],
    key="check41",
    correct_answer="The log-likelihood function *does'nt* penalize outliers as harshly as mean squared error.",
    correct_feedback="""Correct! The growth of the log-likelihood function is
    determined by the natural logarithm, which grows much more slowly
    than a quadratic function.
    """,
    incorrect_feedback="Incorrect, try again. The answer is more or less stated above."
)
st.markdown("---") 

st.markdown(f"""
### 6. Conclusion
We can report to our fictitious employer, Netflix,
that $x=454.6$ minutes is an important threshold based
on the data we have been able to analyze. In particular,
if a customer has watched less than 454.6 minutes of content
over a 3 month period, they are more likely to cancel than
retain their subscription in the next 3 months. Even more
concretely, we can produce the following probabilities from the
training data.
""")

logr_pred_set = pd.DataFrame({
    "Time": new_data['Time'],
    "Probability of Retention": model_logr_new.predict_proba(poly.transform(new_data[['Time']]))[:, 1]
})
logr_pred_set = logr_pred_set.reset_index(drop=True)
with st.expander("Predicted Probability of Retention"):
    st.write(logr_pred_set)

st.markdown("""We can see that according to our model,
 any customer who watched less
than approximately 388 minutes will retain their account
at most 25\% of the time.""")
st.markdown("---") 
st.markdown("### Looking Forward")
st.markdown(
    f"""
Thank you for reading this demo! Logistic regression is
a standard model for {highlight("binary classification")} problems
(classification problems with exactly two categories). 
In the next several demos, we will explore the log-likelihood
function and cost function minimization in greater detail. 


If you have any questions or comments, 
please enter them using the form below.
    """, unsafe_allow_html=True
    )


with st.form("comment_form"):
    name = st.text_input("Name (optional)")
    comment = st.text_area("Comments, questions, or suggestions for future topics:")
    submit_comment = st.form_submit_button("Submit")
    if submit_comment and comment.strip():
        save_comment(name.strip(), comment.strip())
        st.success("Comment submitted!")