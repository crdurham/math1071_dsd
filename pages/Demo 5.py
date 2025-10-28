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

def compute_cost(x, y, m, b=134): 
#Computes the cost function for linear regression.

#Args:
    #x (ndarray): Shape (M,) Input to the model (Population of cities) 
    #y (ndarray): Shape (M,) Label (Actual profits for the cities)
    #m (scalar): Slope parameter of the model

#Returns
    #total_cost (float): The cost of using m as the parameter for 
    # linear regression to fit the data points in x and y

    # M = number of training examples
    M = x.shape[0] 
    
    total_cost = 0
       
    for i in range(M):
        fm_i = m*x[i] + b #Predicted value
        cost_i = (fm_i - y[i])**2 #Squared error associated with prediction
        total_cost = total_cost + cost_i #Add up over all observed data

    total_cost = (1/(M))*total_cost #Take average

    return total_cost 

def compute_derivative(x, y, m, b=134): 
   # Computes the derivative for linear regression 
   # Args:
      #x (ndarray): Shape (M,) Input to the model 
      #y (ndarray): Shape (M,) Label 
      #m (scalar): Slope parameter of the model  
   # Returns
    #  dj_dm (scalar): The derivative of the cost w.r.t. the parameter m  
    
    # Number of training examples
    M = x.shape[0]
    dj_dm = 0
    for i in range(M):
        fm_i = m*x[i] + b #ith redicted value 
        dj_dm_i = (fm_i - y[i])*x[i] #Term in derivative
                                    #contributed by ith value

        dj_dm += dj_dm_i #Add up all contributing terms

    dj_dm = (2/M)*dj_dm #Take average (and mult by 2 from C.R.)
        
    return dj_dm
    
def gradient_descent(x, y, m_in, cost_function, gradient_function, alpha, num_iters): 
    
    #Args:
     #x :    (ndarray): Shape (M,)
      #y :    (ndarray): Shape (M,)
     # m_in: (scalar) Initial value of slope
      #cost_function: function to compute cost
      #gradient_function: function to compute the gradient
      #alpha : (float) Learning rate
      #num_iters : (int) number of iterations to run gradient descent
    #Returns:
      #m (scalar): Updated value of parameter after running gradient descent
      #J_history (List): History of cost values
      #p_history (list): History of parameters m 

    J_history = []
    p_history = []
    m = m_in
    
    for i in range(num_iters):
        # Calculate the derivative
        dj_dm = gradient_function(x, y, m)     

        # Update parameter                      
        m = m - alpha * dj_dm                            
      
        J_history.append(cost_function(x, y, m))
        p_history.append(m)
 
    return m, J_history, p_history  

def highlight(text, color="#00cc6633"):
    """Return text string wrapped in HTML span with background highlight."""
    return f'<span style="background-color:{color}; padding:2px 4px; border-radius:3px"><b>{text}</b></span>'

def generate_tickets_data(n_total=10, seed_choice=42):
    seed=seed_choice
    rng = np.random.default_rng(seed)
    prices = rng.normal(100, 100, n_total).clip(0,400).round(2)
    x = []
    for p in prices:
        if p == 0:
            x_p = 134 - rng.uniform(0,1)
        else:
            x_p = (min(134 - 0.28*p, 134)+rng.normal(0, 10)).round(2)
        x.append(x_p)
    df = pd.DataFrame({
        "Price": prices,
        "Demand": x
    })

    return df

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

st.title("Demo 5: Optimization Applied to Cost Function Minimization")

st.subheader("Goals:")
st.markdown(
    f"""
    1. Utilize optimization techniques to explicitly minimize the cost function
    associated with linear regression (mean squared error).
    2. Understand how {highlight("gradient descent")} can be used to
    iteratively find the (local) minimum of a function, and apply
    this reasoning to the cost function.
    3. Visualize gradient descent in 1D.

    **NOTE:** \\
    (1) Any questions posed within Checkpoints are only for you to monitor your understanding and will not be graded.
    The graded component of each walkthrough is the associated worksheet, which will ask questions based directly on the content
    shown here. 
    
   (2) Any quantity which has *parentheses in the exponent* is only a label
    but NOT an exponent in the multiplicative sense!
    """, unsafe_allow_html=True
)

st.markdown("---") 
st.markdown(f"""
            ### 0. Motivating Example: Airline Ticket Pricing
            An airline company called **AirBud Economy Flights (ABEF)**
            has been experimenting with the ticket prices for their regional
            flights. The plane used for these flights is currently
            the Boeing 717 which can carry 134 passengers. ABEF knows that if
            the average price per ticket is set to \$100 or lower the flight will sell 
            out almost every time, so our focus as a part of their analytics team
            is the following: **predict the average number of seats filled (average
            number of tickets sold across all ABEF hubs) $x(p)\leq 134$
            as a function of price $p$, where $p\geq 0$ is given as dollars above 100**.

            **Initially the company provides us with a small dataset of 10 price points
            along with the corresponding average number of tickets sold**.
            """, unsafe_allow_html=True)

quiz_card(
    question_title="0.1 Checkpoint",
    question_text="""Is this a *regression* problem or a *classification* problem?  
    """,
    options=["For in that sleep of death, what dreams may come...", 
             "It's a classification problem because we can find out if the plane sells out or not.", 
             "It's a regression problem because we are predicting the number of seats sold.",
             "It's both."],
    key="check01",
    correct_answer="It's a regression problem because we are predicting the number of seats sold.",
    correct_feedback="""Correct! Even though there are technically only a finite
    a finite number of seats, we are predicting the average number of tickets sold across
    all operating locations of ABEF. Even if we were predicting the number
    of seats filled at a single location, the number of possible outputs
    is sufficiently large to make regression reasonable.
    """,
    incorrect_feedback="Try again Hamlet, only finitely many options to choose from."
)
quiz_card(
    question_title="0.2 Checkpoint",
    question_text="""Which business quantity or quantities will our model ultimately help to estimate?  
    """,
    options=["The cost function C(x) of operating each flight.", 
             "...when we have shuffled off this mortal coil, must give us pause.", 
             "The demand x as a function of price p, and then the elasticity of demand.",
             "The marginal profit MP(x)."],
    key="check02",
    correct_answer="The demand x as a function of price p, and then the elasticity of demand.",
    correct_feedback="""Correct! The input is the ticket price $p$; the output
    is the number of tickets sold $x(p)$, which is the demand. We can then compute
    $E(p)=-\\frac{p}{\\hat{x}}\\frac{d\\hat{x}}{dp}$, where $\\hat{x}$ is our model
    function.
    """,
    incorrect_feedback="Keep at it, only finitely many options to choose from."
)
st.markdown("---") 

st.markdown(f"""
            ### 1. Getting to Know the Data

            As always, we should begin by looking at the data we are provided
            to get an overview or intuitive understanding of its structure.
            """, unsafe_allow_html=True)
data = generate_tickets_data()
with st.expander("Observed Data"):
    st.write(data)
with st.expander("Descriptive Statistics"):
    st.write(data.describe())
with st.expander(f"Plot of $x(p)$"):
    fig, ax = plt.subplots()
    ax.scatter(
            data['Price'],
            data['Demand'],
            marker='o',
            color = 'black',
            s=25,
        )
    ax.set_ylabel("Tickets Sold x(p)")
    ax.set_xlabel("Price Above $100")
    
    st.pyplot(fig)

st.markdown(f"""The plot above shows a distinct, consistent, downward trend.
            This is expected, since higher prices generally lead to lower customer
            demand. We also can see from the plot or the observed data that when
            the price is set to \$100, the number of tickets sold is approximately
            134; this confirms that regional flights tend to sell out when
            the price is set sufficiently low. This gives us useful information
            about the *intercept* of our model.""")
quiz_card(
    question_title="1.1 Checkpoint",
    question_text="""What value should we have for $\\hat{x}(0)$, where $\\hat{x}$ is our model function? That is, how many tickets
    should we predict ABEF will sell when the price per ticket is \$100?
    """,
    options=["Should set the value to 134 because the plane sells out at this price.", 
             "Expect a value of 114.42 since this is the average demand across all points.", 
             "Guess #1!",
             "We will set this value to infinity since p=0."],
    key="check11",
    correct_answer="Should set the value to 134 because the plane sells out at this price.",
    correct_feedback="""Correct! We know the plane will sell out at any
    price $\leq 100$, so we can set our model output to $\\hat{x}(0)=134$.
    """,
    incorrect_feedback="... on to guess #2?"
)

st.markdown("---") 

st.markdown(f"""
            ### 2. Model Selection and Cost Function

            Based on the plot of the data in the previous section, it is
            reasonable to {highlight("assume a linear model")} of the form $\\hat{{x}}(p)=mp + b$.
            The variables $m$ and $b$ representing the slope and intercept respectively
            are known as the {highlight("model parameters")}. We know
            from the 1.1 Checkpoint that $b=134$ is the best choice of the
            intercept, meaning our model is """, unsafe_allow_html=True)
st.latex(r"""
\hat{x}(p) = 
mp + 134 ,
""")
st.markdown(f"""
            which has a single parameter $m$. {highlight("Our goal is to determine what value of $m$ results in the line which best fits the data")}.
            Stated more precisely, we wish to minimize the mean squared error
            cost function. Below we recall how this cost function is constructed 
            or defined. """, unsafe_allow_html=True)

with st.container(border=True):
    st.markdown("""<h5><u>Recall: Mean Squared Error Cost Function</u></h5>""", unsafe_allow_html=True)
    st.markdown(
        """
        Suppose for example that our model is $\\hat{x}(p)=-0.5p+134$, i.e.
        we choose $m=-0.5$. Note that in the observed data we have one point $(130.47, 106.26)$ where $p= 130.47$ and 
        the actual demand at this price is seen to be $x=106.26$, and another point $(68.38, 119.53)$ where $p= 68.38$ and 
        the actual demand at this price is $x=119.53$.

        1. Given $p=130.47$, our model predicts that the demand will take the value
        $\\hat{x}(130.47)=68.765$. Similarly, given $p=68.38$, our model predicts the demand will be
        $\\hat{x}(68.38)=99.81$. 

        2. The actual value of the demand for the first observed data point is $x=106.26$; the **squared error**
        associated with our model prediction is then\\
        $(68.765-106.26)^2=1434.89$. The actual value of the demand for the second observed data point is $x=119.53$; the **squared error**
        associated with our model prediction is then
        $(99.81-119.53)^2=388.88$.

        3. The mean squared error across the two data points is (as the name 
        suggests) the average of the squared errors:
        $$\\frac{1}{2}(1434.89 + 388.88) = 911.89$$

        4. In general, let our dataset consist of the ten points $(p^{(0)}, x^{(0)}),\ldots ,(p^{(9)}, x^{(9)})$ and 
        suppose our model $\hat{x}(p)$ predicts the demand for the given prices to be 
        $\hat{x}(p^{(0)})=\hat{x}^{(0)},\ldots, \hat{x}(p^{(9)})=\hat{x}^{(9)}$. Note the numbers within
        parenthese in the exponent are *only labels*, they do not indicate repeated multiplication. Then the mean squared error can
        be written as

        $$J = \\frac{1}{10}((\hat{x}^{(0)}-x^{(0)})^2+(\hat{x}^{(1)}-x^{(1)})^2+\cdots + (\hat{x}^{(9)}-x^{(9)})^2).$$
        """
    )

st.markdown(f"""
            Following this reasoning, we can write the mean squared error
            cost function for the observed data: it's a function of $m$, since
            (until we pick a particular value of $m$) our model depends on $m$! In the table
            below, the Price column contains $p^{{(0)}}, \ldots, p^{{(9)}}$ while the
            Demand Prediction column contains $\hat{{x}}(p^{{(0)}})=\hat{{x}}^{{(0)}}=p^{{(0)}}m+134
            \ldots, \hat{{x}}(p^{{(9)}})=\hat{{x}}^{{(9)}}=p^{{(9)}}m+134$. Lastly, the Demand column
            contains the actual demand from the observed data $x^{{(0)}},\ldots,x^{{(9)}}$.
""", unsafe_allow_html=True)

model_output_m = pd.DataFrame({
    "Price": data['Price'],
    "Demand Prediction": ['130.47m +134', '134', '175.05m + 134', '194.06m + 134', '134', '134',
     '112.78m + 134', '68.38m+134', '98.32m+134', '14.7m+134'],
     "Demand": data['Demand']
})

st.write(model_output_m)


st.markdown(f"""
            From this table and our knowledge of the mean squared error cost function, 
            we know that
            """, unsafe_allow_html=True)
st.latex(r"""
J(m) = \frac{1}{10}((\hat{x}^{(0)}-x^{(0)})^2+(\hat{x}^{(1)}-x^{(1)})^2+\cdots + (\hat{x}^{(9)}-x^{(9)})^2)\\
         
         = \frac{1}{10}\left((130.47m+134-106.26)^2+(134-133.0732)^2+\cdots + (14.7m+134-129.38)^2\right)\\
         =\frac{1}{10}(112602 m^2 + 54922.3m + 6939.22)\\
         = 11260.2m^2+5492.23m + 693.922
""")

st.markdown(f"""
            This shows the mean squared error cost function is a positive
            quadratic as a function of the slope parameter $m$. From this
            we know that $J(m)$ must have a global minimum which occurs
            at the vertex of the parabola; all that remains is to determine
            the value of the input $m$ at which the minimum occurs!

            As an intermediate step, for each value of $m$ we can plot
            (1) the line $\hat{{x}}(p)=mp+134$, and (2) the point $(m,J(m))$
            on the parabola. 

            #### 2.1 Graphs of Cost and Corresponding Model
            Below are the plots of lines $mp+134$ for slopes 
            $m=0,-0.1, -0.2, -0.3$, $-0.4$, and $-1$, shown along with the scatterplot
            of the observed data.
            """, unsafe_allow_html=True)
m_vals = [0, -0.1, -0.2, -0.3, -0.4, -1]
min_price = 0
max_price = max(data["Price"])
prices = np.linspace(start=min_price, stop=max_price)

with st.expander("Possible Models"):
    fig1, ax1 = plt.subplots(nrows=6, figsize=(10,20))

    for i in range(6):
        m = m_vals[i]
        demands = m*prices + 134
        ax1[i].scatter(data["Price"], data['Demand'], color='black', s=40)
        ax1[i].plot(prices, demands, linestyle='--', color='blue', lw=3, label=f'm={m}')
        ax1[i].legend(loc='center left')
    fig1.supxlabel('Price Over $100', fontsize=16)
    fig1.supylabel('Demand', fontsize=16)
    st.pyplot(fig1)


st.markdown(f"""
            Among the above model options, the ones with slope
            $m=-0.2$ and $m=-0.3$ appear to fit the data best. When we plot the graph
            of $J(m)$, it is no surprise to see the vertex occurs near $m=-0.25$.
            """, unsafe_allow_html=True)
with st.expander("Plot of Cost Function"):
    fig2, ax2 = plt.subplots()
    J = 11260.2*np.linspace(start=-1.5, stop=1)**2+5492.23*np.linspace(start=-1.5, stop=1) + 693.922
    ax2.plot(np.linspace(start=-1.5, stop=1), J, color='blue', label='Cost Function')
    ax2.scatter(np.array(m_vals), 11260.2*np.array(m_vals)**2+5492.23*np.array(m_vals) + 693.922,
                color='red', s=55, label='m=0, -0.1, -0.2, -0.3, -0.4, -1')
    ax2.set_xlabel("Slope (m)")
    ax2.set_ylabel("MSE ( J )")
    ax2.legend()
    st.pyplot(fig2)
quiz_card(
    question_title="2.1.1 Checkpoint",
    question_text="""True or False: Since the cost function appears to reach a minimum between
    $m=-0.2$ and $m=-0.3$, we should choose the model $\hat{x}(p)=-0.25p+134$.
    """,
    options=["(A) Yes, because our goal was to minimize cost.", 
             "(B) No, because we can find the exact value of m to minimize cost.", 
             "(C) Either (A) or (B), depending on the needs of the company ABEF."
             ],
    key="check211",
    correct_answer="(C) Either (A) or (B), depending on the needs of the company ABEF.",
    correct_feedback="""Correct! If the company is only doing this analysis in
    an exploratory fashion at the moment, then it is possible they do not
    require us to obtain more than an estimate of $m$. However, if
    they want to make significant decisions based on our results, they 
    may need a higher degree of accuracy.
    """,
    incorrect_feedback="Try again. Can you reason why both (A) and (B) could be true?"
)
st.markdown(f"""
            While we could choose $m=-0.25$ and state that
            the resulting model {highlight("$\\hat{{x}}(p)=-0.25p+134$")}
            is sufficiently accurate for our purposes, we
            can explicitly determine the optimal value of $m$.
            The two methods for doing this are shown in the
            next sections.
            """, unsafe_allow_html=True)

st.markdown("---") 

st.markdown(f"""
            ### 3. Minimizing the Cost Function: Exact Version 
            From Fermat's Theorem, we know the derivative of the
            cost function will be 0 at the minimum. From the equation
            which we found for $J(m)$, we find its only critical point is
            (rounded to 4 decimal places)
            """, unsafe_allow_html=True)

st.latex(r"""
J'(m) = \frac{d}{dm}(11260.2m^2+5492.23m + 693.922)=22520.4m+5492.23=0\\
         
         \implies m=-\frac{5492.23}{22520.23}=\boxed{-0.2439}
""")


st.markdown(f"""
            This shows that the line of best fit for out data is {highlight("$\\hat{{x}}(p)=-0.2439p+134$")}. The
            difference between the slopes $-0.25$ and $-0.2439$ may seem small,
            but remember: the price $p$ can be on the order of \$100,
            meaning this difference is magnified in the model output.
            """, unsafe_allow_html=True)
quiz_card(
    question_title="3.1 Checkpoint",
    question_text="""Suppose ABEF sets the ticket price to \$400. What
    is the difference in predicted demand from the estimate model
    $\hat{x}(p)=-0.25p+134$ and the optimal model $\hat{x}(p)=-0.2439p+134$?
    """,
    options=["59", 
             "60.83", 
             "Undefined.",
             "1.83"
             ],
    key="check31",
    correct_answer="1.83",
    correct_feedback="""Correct! Remember, $p$ should be the price
    minus \$100 (the amount of money charged beyond \$100). 
    We can calculate $-0.25(300)+134=59$, while $-0.2439(300)+134=60.83$.
    This is significant because the money brought in for ABEF
    depends on how many tickets get sold, and even two more/less makes a difference.
    """,
    incorrect_feedback="Try again. Remember that $p$ is given by ticket price" \
    " minus \$100 (it's the price beyond \$100)."
)
st.markdown("---") 

data_big = generate_tickets_data(n_total=500)
st.markdown(f"""
            ### 4. Minimizing the Cost Function: Gradient Descent 
            Something which we should not overlook is that the
            process of computing $J$ can be difficult. Here with
            just a tiny collection of data points, we had to sum ten
            squared terms! Finding the minimum of $J$ is even more
            tricky, especially when we have **multiple features or parameters**, so
            we need a method for finding the optimal parameter(s) which
            *scales well*. We will describe the standard method here for
            the case of one parameter: {highlight("gradient descent")}.

            #### 4.1 Overview of Algorithm
            Gradient descent is an iterative algorithm which allows
            us to estimate the optimal parameter $m$ in the following way:

            1. Initialize $m=m_0$ randomly. Just pick a real number $m_0$, set $m=m_0$.
            2. Calculate $J'(m_0)$.
            3. Update $m=m_0$ to $m=m_1$ by shifting $m=m_0$ a small amount in 
            a direction determined by whether $J'(m_0)>0$ or $J'(m_0)<0$.
            4. Repeat steps 2-3 until $J$ no longer decreases a significant amount.
            (Formally, at the $k^{{th}}$ iteration, we calculate $J'(m_{{k-1}})$, then set
            $m=m_k$ by shifting $m_{{k-1}}$ in a direction depending on the sign of $J'(m_{{k-1}})$.
            If $J(m_k)-J(m_{{k-1}})$ is small, stop iterating.)

            #### 4.2 Applying to Larger Flights Dataset
            Suppose that ABEF continues to collect data, and we now
            have access to 500 observed price-demand pairs.
            """, unsafe_allow_html=True)
with st.expander(f"Plot of $x(p)$, Large Dataset"):
    fig3, ax3 = plt.subplots()
    ax3.scatter(
            data_big['Price'],
            data_big['Demand'],
            marker='o',
            color = 'black',
            s=25,
        )
    ax3.set_ylabel("Tickets Sold x(p)")
    ax3.set_xlabel("Price Above $100")
    
    st.pyplot(fig3)

st.markdown(f"""
            In order to *implement* gradient descent, we need to
            write it in code! This is simply not something we can compute
            by hand. We will need to compute $J(m)$, $J'(m)$, and update
            our choice of $m$ in each iteration.
            
            Please note: you do NOT need to understand or  even read the
            code! It is included for completeness.
            """)

with st.expander("Code for Gradient Descent", expanded=False):
    code = f"""
def compute_cost(x, y, m): 
#Computes the cost function for linear regression.

#Args:
    #x (ndarray): Shape (M,) Input to the model (Population of cities) 
    #y (ndarray): Shape (M,) Label (Actual profits for the cities)
    #m (scalar): Slope parameter of the model

#Returns
    #total_cost (float): The cost of using m as the parameter for 
    # linear regression to fit the data points in x and y

    # M = number of training examples
    M = x.shape[0] 
    
    total_cost = 0
       
    for i in range(M):
        fm_i = m*x[i] #Predicted value
        cost_i = (fm_i - y[i])**2 #Squared error associated with prediction
        total_cost = total_cost + cost_i #Add up over all observed data

    total_cost = (1/(M))*total_cost #Take average

    return total_cost 

def compute_derivative(x, y, m): 
   # Computes the derivative for linear regression 
   # Args:
      #x (ndarray): Shape (M,) Input to the model 
      #y (ndarray): Shape (M,) Label 
      #m (scalar): Slope parameter of the model  
   # Returns
    #  dj_dm (scalar): The derivative of the cost w.r.t. the parameter m  
    
    # Number of training examples
    M = x.shape[0]

    for i in range(m):
        fm_i = m*x[i] #ith redicted value 
        dj_dm_i = (fm_i - y[i])*x[i] #Term in derivative
                                    #contributed by ith value

        dj_dm += dj_dm_i #Add up all contributing terms

    dj_dm = (2/M)*dj_dm #Take average (and mult by 2 from C.R.)
        
    return dj_dm
    
def gradient_descent(x, y, m_in, cost_function, gradient_function, alpha, num_iters): 
    
    #Args:
     #x :    (ndarray): Shape (M,)
      #y :    (ndarray): Shape (M,)
     # m_in: (scalar) Initial value of slope
      #cost_function: function to compute cost
      #gradient_function: function to compute the gradient
      #alpha : (float) Learning rate
      #num_iters : (int) number of iterations to run gradient descent
    #Returns:
      #m (scalar): Updated value of parameter after running gradient descent
      #J_history (List): History of cost values
      #p_history (list): History of parameters m 

    J_history = []
    p_history = []
    m = m_in
    
    for i in range(num_iters):
        # Calculate the derivative
        dj_dm = gradient_function(x, y, m)     

        # Update parameter                      
        m = m - alpha * dj_dm                            
      
        J_history.append(cost_function(x, y, m))
        p_history.append(m)
 
    return m, J_history, p_history   
    """
    st.code(code)


st.markdown(f""" 
            ##### 4.2.1 Visualizing Gradient Descent 
            Input values below for the initial slope $m$, the 
            learning rate or step size $\\alpha$, and the number of
            iterations.
            """)
input_vals = data_big['Price'].values
labels = data_big['Demand'].values

with st.expander("Gradient Descent Visual"):
    m_init = st.slider("Initial slope $m$:", -4.0, 4.0, 4.0)
    alpha = st.slider(
        "Step Size:", 
        min_value=0.000001, 
        max_value=0.000007, 
        value=0.000003,
        step=0.000001,
        format="%.7f"
    )
    num_iterations = st.slider("Number of Iterations:", 10, 100, 15)

    m_best, J_history, p_history = gradient_descent(
        x=input_vals, y=labels, m_in=m_init, 
        cost_function=compute_cost,
        gradient_function=compute_derivative, 
        alpha=alpha, num_iters=num_iterations
    )
    iterations = np.arange(len(J_history))
    m_min, m_max = min(p_history)-0.5, max(p_history)+0.5
    m_vals_curve = np.linspace(m_min, m_max, 200)
    full_J_curve = [compute_cost(input_vals, labels, m) for m in m_vals_curve]

    fig4, ax4 = plt.subplots()
    ax4.plot(m_vals_curve, full_J_curve, label="J(m)")
    scatter = ax4.scatter(p_history, J_history, c=iterations,       
    cmap='viridis', s=75, label='Descent Point')
    cbar = plt.colorbar(scatter, ax=ax4)
    cbar.set_label("Iteration") 
    ax4.set_xlabel("Slope (m)")
    ax4.set_ylabel("Cost (J(m))")
    ax4.set_title("Cost Function and Gradient Descent")
    ax4.legend()
    st.pyplot(fig4)

st.markdown(f"""
            We find using gradient descent that the slope $m$
            which results in the smallest error is $m={m_best:0.4f}$.
            Thus for this larger dataset, the line
            of best fit is $\hat{{x}}(p)=-0.2826p$. Notice:

            - If the initial value of $m$ is larger than the optimal value,
            gradient descent updates to a smaller value. This is because $J'(m)>0$ here,
            and the algorithm changes $m$ to $m-\\alpha J'(m)$ where $\\alpha$ is the
            step size.
            - If the initial value of $m$ is smaller than the optimal value,
            gradient descent updates to a larger value. This is because $J'(m)<0$ here,
            and the algorithm changes $m$ to $m-\\alpha J'(m)$ where $\\alpha$ is the
            step size.
            """) 




st.markdown("---") 
st.markdown("### Looking Forward")
st.markdown(
    f"""
Thank you for reading this demo! We've seen the concept of mean squared
error cost before, but now we see how it can be viewed as a function which,
when minimized, leads us to optimal model parameter(s). Moreover,
gradient descent is an iterative method by which we can find minima of
the cost function.

In the next demo we will explore a bit of
{highlight("unsupervised learning")}: data science with
data that is not labeled, i.e. we only have input features. In particular,
we will look at {highlight("K-means clustering")}.


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