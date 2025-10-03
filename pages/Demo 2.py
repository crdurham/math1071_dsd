import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from pathlib import Path
from datetime import datetime
from db import save_comment
from utils import lin_reg

ROOT = Path(__file__).resolve().parent.parent
HOUSING_DATA_FILE = ROOT / "data" / "Housing.csv"

def highlight(text, color="#00cc6633"):
    """Return text string wrapped in HTML span with background highlight."""
    return f'<span style="background-color:{color}; padding:2px 4px; border-radius:3px"><b>{text}</b></span>'

housing = pd.read_csv(HOUSING_DATA_FILE)

def line_b(x):
    y = 100 + (10/800)*x
    return y

def line_g(x):
    y = (130/800)*x - 10
    return y

def quadratic(a,b,c,x):
    y = a*x**2+b*x+c
    return y
def quad_3(x):
    y = (-1/20)*x**2+4*x+105
    return y

def quad_4(x):
    y = (-1/16)*x**2+(17/4)*x+105
    return y

def generate_bookstore_data(n_total=500, reveal_tenth=False, seed=42):
    rng = np.random.default_rng(seed)
    
    patrons_per_week = np.round(rng.normal(1000, 90, n_total))    # Total store visitors to the store in avg week
    avg_book_price = rng.normal(40, 17, n_total).clip(8,70)  # price of books
    cafe = rng.normal(20, 15, n_total).clip(0, 100)     # cafe spending (hundreds of dollars)
    landscaping_spending = rng.normal(6, 2, n_total).clip(3,30)  # landscaping spending (hundreds of dollars)
    
    patrons_avg = np.mean(patrons_per_week)
    cafe_avg = np.mean(cafe)
    avg_avg_book_price = np.mean(avg_book_price)
    avg_landscaping = np.mean(landscaping_spending)

    patrons_std = np.std(patrons_per_week)
    cafe_std = np.std(cafe)
    std_avg_book_price = np.std(avg_book_price)
    std_landscaping = np.std(landscaping_spending)

    # Revenue model
    revenue = 2*(
        10*((patrons_per_week-patrons_avg)/patrons_std) - 0.0012*((patrons_per_week-patrons_avg)/patrons_std)**2+
        25*np.exp(-(((avg_book_price-avg_avg_book_price)/std_avg_book_price+0.35)**2/0.85)) +
        15*np.log(np.abs(10*(cafe - cafe_avg)/cafe_std + 2*cafe_avg)) +
        0.25*(landscaping_spending - avg_landscaping)/std_landscaping + 10
        
        #rng.normal(0, 1, n_total)   # noise
    )
    
    df = pd.DataFrame({
        "Patrons": patrons_per_week.round(2),
        "Average Book Price": avg_book_price.round(2),
        "Cafe Costs": cafe.round(2),
        "Landscaping Costs": landscaping_spending.round(2),
        "Revenue": revenue.round(2)
    })
    
    if reveal_tenth:
        return df.iloc[:n_total//10].reset_index(drop=True)
    else:
        return df.reset_index(drop=True)

COMMENTS_FILE = Path("comments.csv")


def contains_negative(L):
    nonpos_sizes = 0
    for i in range(len(L)):
        if L[i] <= 0:
            nonpos_sizes+=1
        else:
            nonpos_sizes=nonpos_sizes
    return nonpos_sizes

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

st.title("Demo 2: Model Assessment with Applications to Business Data", width="content")

st.subheader("Goals:")
st.markdown(
    """
    1. Quantitatively describe how well a model fits or describes data.
    2. Understand how a company could use data to impact decision making with regards to revenue 
    and/or resource allocation.
    3. Visually identify how the addition of new data impacts model fit.

    **NOTE:** \\
    (1) Any questions posed within Checkpoints are only for you to monitor your understanding and will not be graded.
    The graded component of each walkthrough is the associated worksheet, which will ask questions based directly on the content
    shown here. 
    
   (2) Any quantity which has *parentheses in the exponent* is only a label
    but NOT an exponent in the multiplicative sense!
    """
)

st.markdown("---")

st.markdown(" ### 0. Motivating Example")
st.markdown("""
         Throughout this demo, we will explore how **revenue** is impacted by different sectors of a business. In particular, we will
         consider a fictitious chain of bookstores with the brand name *Amanda's Bookstore* which has 500 locations across the US. Of the
        500 total stores, **50 initially supply us with the following *__average weekly__* data**:

         - total number of patrons who entered the store 
         - average book price throughout the store
         - money spent on caf&eacute; 
         - money spent on landscaping 
         - total revenue
        
        Monetary quantities are given in hundreds of dollars, except for average book price which is in dollars. 
        From this we want to determine which factor(s) most strongly interact with revenue, either positively or negatively.
""")

quiz_card(
    question_title="0.0 Checkpoint",
    question_text="""A particular store location in Florida spends $200 each week to maintain an alligator enclosure (filed under landscaping). 
                     What type of cost is this?""",
    options=["(A) Fixed weekly cost, since the expense doesn't change with company production.", 
             "(B) Variable weekly cost, since pets have different needs depending on the week.", 
             "(C) This is known as the cost of being Floridian.",
             "(D) None of the above."],
    key="check00",
    correct_answer="(A) Fixed weekly cost, since the expense doesn't change with company production.",
    correct_feedback="""Correct! Based on the information given, the store spends a *constant* amount each week; constant
    means fixed.""",
    incorrect_feedback="Sorry, that's incorrect. Hint: is the weekly cost varying based on the information provided?"
)
st.markdown("---")

st.markdown("### 1. Exploratory Data Analysis")

st.markdown(f"""
            The first step when approaching a new problem is often just to become familiar with the setup. This can be done 
         through any number of methods when data is available, but most typically we will look at (i) **{highlight("descriptive statistics")}**
         (e.g. relevant means, medians, variances) and (ii) scatter plots or other **{highlight("visual representations")}** of the data structure.
         """, unsafe_allow_html=True)

st.markdown("""#### 1.1 Descriptive Statistics""")
data_first_tenth = generate_bookstore_data(reveal_tenth=True) #Only show half of the stores

st.write("For a glimpse at the data, the first 20 collected values are displayed here.")

st.write(data_first_tenth.head(20))

st.markdown(f"""
         What information would we want to gain? Simply put, we want to know what the typical store does or experiences; moreover, if
         a store deviates from the norm, in what way are they likely to deviate? For instance, a way to measure the typical number of patrons 
         to visit a store in a week is by computing the **{highlight("average or mean")}** weekly
         patrons across observed stores:""", unsafe_allow_html=True)

st.latex(r"\text{Average Patrons}=\overline{P}=\frac{(\text{Store 1 Patrons}) + (\text{Store 2 Patrons}) + \cdots + (\text{Store 50 Patrons})}{50}")       

st.markdown(""" This is useful on its own, but we can further contextualize an average value by also looking at the **variance** or **spread** of the
values within a category. We can compute variance by hand as
    """)

st.latex(r"\text{Var}(P)=\frac{(\text{Store 1 Patrons}-\overline{P})^2 + (\text{Store 2 Patrons}-\overline{P})^2 " \
r"+ \cdots + (\text{Store 50 Patrons}-\overline{P})^2}{50},") 

st.markdown(""" 
            meaning that it is the average squared deviation from the mean! Thankfully, most programming languages have built-in functions
            for computing mean, variance, and many other statistical quantities. In Python, using `DataFrame.describe()` provides us with
            many standard statistics for each column. Below we see this used on the 50 data points we have so far.
            """)

description = data_first_tenth.describe()

st.write(description)

with st.expander("Code for descriptive stats", expanded=False):

    code = f"""
description = data_first_half.describe()
            """
    st.code(code)


with st.expander("Histogram Plots"):
    fig, ax = plt.subplots()
    ax.hist(data_first_tenth['Revenue'], color='navy')
    ax.set_xlabel("Weekly Revenue (Hundreds of Dollars)")
    ax.set_ylabel("Occurrences")
    ax.set_title("Weekly Revenue Distribution")
    st.pyplot(fig)

    cola, colb = st.columns(2)
    with cola:
        fig, ax = plt.subplots()
        ax.hist(data_first_tenth['Patrons'], color='maroon')
        ax.set_xlabel("Weekly Patrons")
        ax.set_ylabel("Occurrences")
        ax.set_title("Weekly Patrons Distribution")
        st.pyplot(fig)

        fig1, ax1 = plt.subplots()
        ax1.hist(data_first_tenth['Average Book Price'], color='black')
        ax1.set_xlabel("Price per Book (Dollars)")
        ax1.set_ylabel("Occurrences")
        ax1.set_title("Unit Pricing Distribution")
        st.pyplot(fig1)

    with colb:
        fig, ax = plt.subplots()
        ax.hist(data_first_tenth['Cafe Costs'], color='orange')
        ax.set_xlabel("Cafe Spending (Hundreds of Dollars)")
        ax.set_ylabel("Occurrences")
        ax.set_title("Cafe Spending Distribution")
        st.pyplot(fig)

        fig1, ax1 = plt.subplots()
        ax1.hist(data_first_tenth['Landscaping Costs'], color='green')
        ax1.set_xlabel("Landscaping Spending (Hundreds of Dollars)")
        ax1.set_ylabel("Occurrences")
        ax1.set_title("Landscaping Spending Distribution")
        st.pyplot(fig1)

st.markdown("""
         The main takeaways so far:
         - The average store sees roughly 1000 patrons each week
         - Spending on the cafe is roughly four times spending on landscaping
         - Most stores have a weekly revenue in the interval of dollars [13108.31, 18391.45]
            (based on the mean and standard deviation)
         - Landscaping costs are somewhat spread out relative to the average expense
         - At least one store does not have a caf&eacute;, or at least they spent
            no money on their caf&eacute;.
         """)

quiz_card(
    question_title="1.1 Checkpoint",
    question_text="In dollars rounded to the nearest cent, what is the average weekly revenue among the stores which have reported their data?",
    options=['$3952.12', "$196.22", "$15,749.88", "$15,749.00"],
    key="check11",
    correct_answer="$15,749.88",
    correct_feedback="""Correct! The average is displayed in the row labeled 'mean', and all amounts are in hundreds of dollars.""",
    incorrect_feedback="Sorry, that's incorrect. Remember we're looking at the mean revenue and all values are in hundreds of dollars."
)

st.markdown("""#### 1.2 Visuals""")
st.write("""Our main objective is to determine where store owners should focus their
         attention and/or funds. To that end, we should plot revenue on the vertical
         axis as an output of each category on the horizontal axis. Click on the expanders
         below to view each plot.
         """)

with st.expander("Patrons"):
    fig, ax = plt.subplots()
    ax.scatter(data_first_tenth['Patrons'], data_first_tenth['Revenue'], marker='o', color='maroon')
    ax.set_xlabel("Weekly Patrons")
    ax.set_ylabel("Revenue (Hundreds of Dollars)")
    ax.set_title("Revenue vs. Patrons")
    st.pyplot(fig)
with st.expander("Book Pricing"):
    fig1, ax1 = plt.subplots()
    ax1.scatter(data_first_tenth['Average Book Price'], data_first_tenth['Revenue'], marker='o', color='black')
    ax1.set_xlabel("Average Book Price ($)")
    ax1.set_ylabel("Revenue (Hundreds of Dollars)")
    ax1.set_title("Revenue vs. Average Book Price")
    st.pyplot(fig1)


with st.expander("Cafe Spending"):
    fig3, ax3 = plt.subplots()
    ax3.scatter(data_first_tenth['Cafe Costs'], data_first_tenth['Revenue'], marker='o', color='orange')
    ax3.set_xlabel("Cafe Spending (Hundreds of Dollars)")
    ax3.set_ylabel("Revenue (Hundreds of Dollars)")
    ax3.set_title("Revenue vs. Cafe Spending")
    st.pyplot(fig3)

with st.expander("Landscaping Spending"):
    fig4, ax4 = plt.subplots()
    ax4.scatter(data_first_tenth['Landscaping Costs'], data_first_tenth['Revenue'], marker='o', color='green')
    ax4.set_xlabel("Landscaping Spending (Hundreds of Dollars)")
    ax4.set_ylabel("Revenue (Hundreds of Dollars)")
    ax4.set_title("Revenue vs. Landscaping Spending")
    st.pyplot(fig4)

st.markdown("""
            We can describe the plots as follows:
            
            - As the number of patrons increases, the revenue increases in what appears
            to be a linear fashion.

            - Revenue increases with book price until around $35, then decreases. 
            This relationship is a bit less clear than the relationship between patrons 
            and revenue.
            
            - As caf&eacute; spending increases, revenue may increase somewhat but is
            relatively flat (except some potential outliers).
            - The scatter plot of revenue vs landscaping expenditure looks similar to 
            the scatter plot of revenue vs caf&eacute; spending.

            Our next goal is to provide a quantitative evaluation of what we intuitively
            described above through modeling. To do this, we need a concrete way to 
            determine if a model curve fits the data well.
            """)

quiz_card(
    question_title="1.2 Checkpoint",
    question_text="What type of function would be most suitable for modeling revenue as a function of book pricing?",
    options=['Linear', "Quadratic", "Exponential", "Other"],
    key="check12",
    correct_answer="Quadratic",
    correct_feedback="""Correct! The data appears to follow a downward-facing quadratic curve, 
    i.e. a quadratic with negative leading coefficient.
    """,
    incorrect_feedback="Sorry, that's incorrect. Refer to the image and description above."
)
st.markdown("---")

st.markdown("""### 2. Determining Best Fit: Linear Model""")
st.markdown(f"""
            #### 2.1 Error and Cost
            The plots below show two different attempts at using a linear function to
            model the relationship between patrons per week and revenue. The 
            equation of the line in **{highlight("Linear Model 1 (LM1)")}** is
            """, unsafe_allow_html=True)
st.latex(r"R_1(x)=\frac{1}{80}x + 100") 

st.markdown(f"""
            while the equation of the line in **{highlight("Linear Model 2 (LM2)")}** is
            """, unsafe_allow_html=True)
st.latex(r"R_2(x)=\frac{13}{80}x - 10,")

st.markdown("""
            both of which take in $x =$ number of patrons and output a prediction of
            the revenue. 
            """)

patrons_range = np.arange(start=data_first_tenth["Patrons"].min(), stop=data_first_tenth['Patrons'].max())
revenue_est_b = 100 + (10/800)*patrons_range
revenue_est_g = (130/800)*patrons_range - 10
revenue_best = 0.2353*patrons_range -79.755
col_b, col_g = st.columns(2)

with col_b:
    fig_b, ax_b = plt.subplots()
    ax_b.scatter(data_first_tenth["Patrons"], data_first_tenth['Revenue'], color="maroon")
    ax_b.plot(patrons_range, revenue_est_b, linestyle='--', color='blue', linewidth=3)
    ax_b.set_xlabel("Patrons")
    ax_b.set_ylabel("Revenue (Hundreds of Dollars)")
    ax_b.set_title("Linear Model 1")

    st.pyplot(fig_b)

with col_g:
    fig_b, ax_b = plt.subplots()
    ax_b.scatter(data_first_tenth["Patrons"], data_first_tenth['Revenue'], color="maroon")
    ax_b.plot(patrons_range, revenue_est_g, linestyle='--', color='blue', linewidth=3)
    ax_b.set_xlabel("Patrons")
    ax_b.set_ylabel("Revenue (Hundreds of Dollars)")
    ax_b.set_title("Linear Model 2")

    st.pyplot(fig_b)

st.markdown(f"""
            Neither one of the lines pictured above fits the data perfectly, but visually
            we know Linear Model 2 gives a better description of the data; 
            how can we make this precise? We can **{highlight("measure the difference between" 
            " the predicted value and the actual value for each data point")}**.
            """, unsafe_allow_html=True)
st.markdown("---")
st.markdown(f""" 
            ##### Example: Calculating Model Error and Cost for 1 Point
            The first data point which we can see in the first table
            on this page has $x =$ 1027 patrons and an actual revenue of $R =$ $13,811.
            Each model tries to {highlight("predict the revenue")} based on the number of patrons:
            """, unsafe_allow_html=True)
with st.expander(f"Calculate Model Predictions"):
    st.latex(r"R_1(1027) = \frac{1027}{80}+100 = 112.84")
    st.latex(r"R_2(1027) = \frac{13\cdot 1027}{80} - 10 = 156.89")

st.markdown(f"""
            This means Linear Model 1 thinks the revenue will be \$11,284 while 
            Linear Model 2 thinks the revenue will be \$15,689. Associated with each
            prediction is some **{highlight("error, i.e. deviation from the actual revenue")}** of
            138.11 units:
            """, unsafe_allow_html=True)
with st.expander("Calculate Prediction Errors"):
    st.latex(r"err^{(1)}_1 = R_1(1027) - 138.11 = -25.27")
    st.latex(r"err^{(2)}_1 = R_2(1027) - 138.11 = 18.78")

st.markdown(f"""
            Here we denote the error from Linear Model 1 *on the first data point*
            by $err^{(1)}_1$ and the 
            error from Linear Model 2 *on the first data point* by $err^{(2)}$. 
            We don't care about the sign
            of the error, only how far our prediction is from reality, so we 
            **{highlight("compute the squared error or cost")}** which removes the sign:
            """, unsafe_allow_html=True)

with st.expander("Calculate Model Costs (Squared Error)"):
    st.latex(r"cost^{(1)}_1 = \left(err^{(1)}_1\right)^2 = 638.57")
    st.latex(r"cost^{(2)}_1 = \left(err^{(2)}_1\right)^2=352.69")
st.markdown("---")

st.markdown(f"""
            #### 2.2 Average Cost

            In the above example, we computed the error and cost produced by two
            models on a single data point. To measure *__overall model performance__*
            on the entire dataset, we **{highlight("take the average cost over all data points")}**.
            Denote the average cost by $J\left(\\frac{1}{80}, 100\\right)$ for Linear 
            Model 1 and $$J\left(\\frac{13}{80}, -10\\right)$$ for Linear Model 2.
            """, unsafe_allow_html=True)
st.latex(r"J\left(\frac{1}{80}, 100\right) = \frac{cost_1^{(1)}+cost_2^{(1)} + \cdots + cost_{50}^{(1)}}{50}")
st.latex(r"J\left(\frac{13}{80}, -10\right) = \frac{cost_1^{(2)}+cost_2^{(2)} + \cdots + cost_{50}^{(2)}}{50}")

st.markdown("""
            We can calculate these costs for our linear models and find:
            """)
cost_1 = calculate_average_cost(X=data_first_tenth['Patrons'], y=data_first_tenth['Revenue'], model=line_b)
cost_2 = calculate_average_cost(X=data_first_tenth['Patrons'], y=data_first_tenth['Revenue'], model=line_g)
st.write(f"""Linear Model 1 has $J\left(1/80, 100\\right)=$ {cost_1:0.2f}; Linear Model 2 has $J\left(13/80, -10\\right)=$ {cost_2:0.2f}. Thus
         Linear Model 1 has a much higher average cost compared to Linear Model 2.""")

quiz_card(
    question_title="2.1 Checkpoint",
    question_text="""Does it inherently matter if the error associated with the prediction
    is positive or negative?
    """,
    options=['Yes, we always want positive error.', "Yes, we always want negative error.", 
             "No, the *distance* from the actual value is all that matters.", 
             "Idk? Show me the answer."],
    key="check21",
    correct_answer="No, the *distance* from the actual value is all that matters.",
    correct_feedback="""Correct! The sign of the error only tells us if our prediction
    is above or below the actual value; we only care about the distance between the values,
    i.e. the magnitude of their difference.
    """,
    incorrect_feedback="Sorry, guess again. (If you asked for the answer...)"
)
quiz_card(
    question_title="2.2 Checkpoint",
    question_text="""Do we care about the *total* error added up across all data points,
    or the *average error per data point*?
    """,
    options=['The average error per data point.', 
             "The total error.", 
             "Both.", 
             "Neither."],
    key="check22",
    correct_answer="The average error per data point.",
    correct_feedback="""Correct! Imagine we have a dataset with 1000000 data points;
     if our model has an error of 0.01 on every data point, the total error is 10000!
     This is a pretty big number, but only because the dataset is large! Average error
     is more appropriate.
    """,
    incorrect_feedback="Sorry, guess again."
)

quiz_card(
    question_title="2.3 Checkpoint",
    question_text="""How should we justify the assertion that Linear Model 2 is better than
    Linear Model 1?
    """,
    options=['The line from Linear Model 2 looks better. Just look at it.', 
             "Linear Model 2 has a lower average squared error, a.k.a. average cost.", 
             "Linear Model 1 is better, not Linear Model 2."],
    key="check23",
    correct_answer="Linear Model 2 has a lower average squared error, a.k.a. average cost.",
    correct_feedback="""Correct! Smaller average cost means LM2 was generally closer
    to matching the true revenue of each data point than LM1.
    """,
    incorrect_feedback="Try again. Think of what the cost or error tells us about the "
    "model performance."
)
st.markdown("---")
st.markdown(f"""
#### 2.3 Minimizing Cost
            
From the previous discussion, we found that Linear Model 2 is better than 
Linear Model 1 because it has a smaller average squared error. However, that does
not mean Linear Model 2 is the best model! The {highlight("best *linear* model")} is given by 
$R(x)=m_0x+b_0$ where $(m_0,b_0)$ {highlight("minimizes the average squared error $J(m,b)$")} over
*all possible slopes and intercepts*. This is done using Python or R
packages which won't be discussed here, but the line of best fit is
shown below. It has equation 
""", unsafe_allow_html=True)
st.latex(r"R_{L}(x) = 0.2353x-79.755")


with st.expander("Linear Regression Code and Model Output"):
    code =f"""
def lin_reg(X, y):
    '''Fit a weighted logistic regression model with 
    feature data X and label data y.'''
    if not isinstance(y, pd.Series):
        y = pd.Series(y, index=X.index)
    else:
        y = y.reindex(X.index)

    

    model = sm.OLS(
        y,
        sm.add_constant(X),
        family=sm.families.Binomial(),
    )

    results = model.fit()
    return model, results

model, results = lin_reg(X=data_first_tenth['Patrons'], 
                 y=data_first_tenth['Revenue'])
"""
    model, results = lin_reg(X=data_first_tenth['Patrons'], y=data_first_tenth['Revenue'])
    st.code(code)
    with st.expander("Model Output"):
        st.write(results.summary())




fig_best, ax_best = plt.subplots()
ax_best.scatter(data_first_tenth["Patrons"], data_first_tenth['Revenue'], color="maroon")
ax_best.plot(patrons_range, revenue_best, linestyle='--', color='blue', linewidth=3)
ax_best.set_xlabel("Patrons")
ax_best.set_ylabel("Revenue (Hundreds of Dollars)")
ax_best.set_title("Line of Best Fit")

st.pyplot(fig_best)
st.markdown("---")

st.markdown("""
            ### 3. Determining Best Fit: Quadratic Model
            We saw in 1.2 Visuals that revenue is likely not
            modeled well as a linear function of average book price.
            Instead, we should use a second-order or **quadratic** 
            polynomial. Two options are depicted below along with
            their equations.
            """)

book_price_range = np.arange(start=np.min(data_first_tenth['Average Book Price']),
                             stop=np.max(data_first_tenth['Average Book Price']))
quad_b = quadratic(a=-1/20, b=4,c=105,x=book_price_range)
quad_g = quadratic(a=-1/16, b=4.25,c=105,x=book_price_range)

col_quad_1, col_quad_2 = st.columns(2)

with col_quad_1:
    fig_quad_b, ax_quad_b = plt.subplots()
    ax_quad_b.scatter(data_first_tenth["Average Book Price"], data_first_tenth['Revenue'], color="black")
    ax_quad_b.plot(book_price_range, quad_b, linestyle='--', color='blue', linewidth=3)
    ax_quad_b.set_xlabel("Patrons")
    ax_quad_b.set_ylabel("Revenue (Hundreds of Dollars)")
    ax_quad_b.set_title("Line of Best Fit")
    st.pyplot(fig_quad_b)
    st.caption("$R_3(x) = -\\frac{1}{20}x^2+4x+105$")

with col_quad_2:
    fig_quad_b, ax_quad_b = plt.subplots()
    ax_quad_b.scatter(data_first_tenth["Average Book Price"], data_first_tenth['Revenue'], color="black")
    ax_quad_b.plot(book_price_range, quad_g, linestyle='--', color='blue', linewidth=3)
    ax_quad_b.set_xlabel("Patrons")
    ax_quad_b.set_ylabel("Revenue (Hundreds of Dollars)")
    ax_quad_b.set_title("Line of Best Fit")
    st.pyplot(fig_quad_b)
    st.caption("$R_4(x) = -\\frac{1}{16}x^2+\\frac{17}{4}x+105$")

st.markdown("""Of the two, $R_4(x)$ seems to follow the data more
            closely. Let $J_3=J\left(-\\frac{1}{20}, 4,105\\right)$ be 
            the average cost for the model $R_3$ and let $J_4=J\left(-\\frac{1}{16}, \\frac{17}{4},105\\right)$
            be the average cost for the model $R_4$.""")
quiz_card(
    question_title="3.1 Checkpoint",
    question_text="""Which do we expect to be true?  
    """,
    options=['J_3=J_4', 
             "J_3<J_4", 
             "J_3>J_4",
             "J_3=0"],
    key="check31",
    correct_answer="J_3>J_4",
    correct_feedback="""Correct! Since $R_4$ fits the data better than $R_3$,
    we expect the cost $J_4$ corresponding to $R_4$ to be smaller
    than $J_3$, the cost associated with $R_3$.
    """,
    incorrect_feedback="Try again. If one model is better than another," \
    " how should their costs compare?"
)

cost_3 = calculate_average_cost(X=data_first_tenth['Average Book Price'], 
                                y=data_first_tenth['Revenue'], model=quad_3)
cost_4 = calculate_average_cost(X=data_first_tenth['Average Book Price'], 
                                y=data_first_tenth['Revenue'], model=quad_4)
st.markdown(f"""
            Through the same calculation as in 2.2, we
            find $J_3={cost_3:0.2f}$ and $J_4={cost_4:0.2f}$, meaning
            $R_4$ is a demonstrably better model. 
            
            We can also use regression techniques once again to 
            find the best model; in this case we **{highlight("minimize the cost" 
            "$J(a,b,c)$ incurred by any quadratic model $R(x)=ax^2+bx+c$")}.** The
            equation of the best quadratic model is:""", unsafe_allow_html=True)
st.latex(r"R_Q(x) = -0.0523x^2+3.7057x+106.8631")


data_first_tenth['Avg Book Price^2']=data_first_tenth['Average Book Price']**2
model_quad, results_quad = lin_reg(X=data_first_tenth[['Average Book Price', 'Avg Book Price^2']], y=data_first_tenth['Revenue'])

with st.expander("Code for Quadratic Model of Best Fit"):
    st.code(code)
    with st.expander("Model Output"):
        st.write(results_quad.summary())

quad_best = quadratic(a=-0.0523, b=3.7057,c=106.8631,x=book_price_range)
fig_best_quad, ax_best_quad = plt.subplots()
ax_best_quad.scatter(data_first_tenth["Average Book Price"], data_first_tenth['Revenue'], color="black")
ax_best_quad.plot(book_price_range, quad_best, linestyle='--', color='blue', linewidth=3)
ax_best_quad.set_xlabel("Average Book Price")
ax_best_quad.set_ylabel("Revenue (Hundreds of Dollars)")
ax_best_quad.set_title("Best Quadratic Model")
st.pyplot(fig_best_quad)

st.markdown("---")

st.markdown("""
            ### 4. Takeaways for Business Operations
            Based only on the analysis above, we are able to make 
            at least three important conclusions about the operation
            of the 50 stores which reported their data.

            1. Increasing patrons per week will generally lead
            to a *linear* revenue increase. While we obviously expect
            more revenue with more customers, the fact that it doesn't
            taper off but instead follows a linear trajectory is valuable.
            2. Revenue, as a function of book pricing, peaks at an average
            book price of (using vertex formula)
            """)
st.latex(r"-\frac{3.7057}{2(-0.0523)} = \$ 35.43")
st.markdown("""This tells us that the average book price which
            we see in the table in 1.1 of \$ 39.52 is too high!
            This might have gone unnoticed without proper data analysis.\\
            3. Though we didn't fit a models to it, we can say
            that the relationship between landscaping expenses and
            revenue appears to be roughly **constant**. This means that
            spending more money than necessary on landscaping is 
            wasteful, and you won't see a return on your investment.
            """)
st.markdown("---")

st.markdown("""
            ### 5. Adding New Data
            As a preview of some concepts we will see in the future, suppose now
            that all 500 stores have sent in their data to be analyzed.
            Below is the new table of descriptive statistics, followed by
            the updated scatter plots.
            """)
data = generate_bookstore_data(reveal_tenth=False)
description_full = data.describe()

st.write(description_full)
with st.expander("Patrons"):
    fig, ax = plt.subplots()
    ax.scatter(data['Patrons'], data['Revenue'], marker='o', color='maroon')
    ax.set_xlabel("Weekly Patrons")
    ax.set_ylabel("Revenue (Hundreds of Dollars)")
    ax.set_title("Revenue vs. Patrons")
    st.pyplot(fig)
with st.expander("Book Pricing"):
    fig1, ax1 = plt.subplots()
    ax1.scatter(data['Average Book Price'], data['Revenue'], marker='o', color='black')
    ax1.set_xlabel("Average Book Price ($)")
    ax1.set_ylabel("Revenue (Hundreds of Dollars)")
    ax1.set_title("Revenue vs. Average Book Price")
    st.pyplot(fig1)


with st.expander("Cafe Spending"):
    fig3, ax3 = plt.subplots()
    ax3.scatter(data['Cafe Costs'], data['Revenue'], marker='o', color='orange')
    ax3.set_xlabel("Cafe Spending (Hundreds of Dollars)")
    ax3.set_ylabel("Revenue (Hundreds of Dollars)")
    ax3.set_title("Revenue vs. Cafe Spending")
    st.pyplot(fig3)

with st.expander("Landscaping Spending"):
    fig4, ax4 = plt.subplots()
    ax4.scatter(data['Landscaping Costs'], data['Revenue'], marker='o', color='green')
    ax4.set_xlabel("Landscaping Spending (Hundreds of Dollars)")
    ax4.set_ylabel("Revenue (Hundreds of Dollars)")
    ax4.set_title("Revenue vs. Landscaping Spending")
    st.pyplot(fig4)

st.markdown("""
            #### 5.1 Questions to Ponder
            - Does the relationship between patrons and revenue
            still appear to be linear, or is a more complex curve needed
            to model?
            - What has changed about the relationship between book
            pricing and revenue? Is a quadratic model still appropriate?
            - Does cafe spending still appear to have limited impact
            on revenue?
            - Overall: is it better to have *more* data or *less* data?
            """)
st.markdown("---")
st.markdown("### Looking Forward")
st.markdown(
    """
Thank you for reading this demo! We saw how business data
can be used to inform decisions. Specifically, we used the 
idea of *__cost minimization__* to determine the best model of 
revenue as a function of average book price and number of 
weekly patrons.

In the next demo(s) we will explore look at *model complexity*,
*underfitting*, *overfitting*, and the *bias-variance tradeoff*.


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