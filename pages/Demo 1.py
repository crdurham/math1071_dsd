import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from pathlib import Path
from datetime import datetime
from db import save_comment

ROOT = Path(__file__).resolve().parent.parent
HOUSING_DATA_FILE = ROOT / "data" / "Housing.csv"

housing = pd.read_csv(HOUSING_DATA_FILE)



COMMENTS_FILE = Path("comments.csv")


def contains_negative(L):
    nonpos_sizes = 0
    for i in range(len(size_list_input_parsed)):
        if size_list_input_parsed[i] <= 0:
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


st.title("Demo 1: Introduction to Data Storage and Visualization", width="content")

st.subheader("Goals:")
st.markdown(
    """
    1. Become comfortable seeing data presented (a) in a table, or (b) graphically. 
    2. Be able to describe the difference(s) between *deterministic* functions and *random* functions.
    3. Use basic functions as models to describe the relationship between the input and output quantities of a dataset.
    4. Visually understand 'goodness of fit'.

    **Note:** Any questions posed within Checkpoints are only for you to monitor your understanding and will not be graded.
    The graded component of each walkthrough is the associated worksheet, which will ask questions based directly on the content
    shown here.
    """
)

quiz_card(
    question_title="0.0 Checkpoint",
    question_text="What will be graded for this demo?",
    options=['(A) Responses to questions on this site.', "(B) Nothing, it's all optional.", "(C) Responses to questions in associated worksheet.", "(D) Both (A) and (C)."],
    key="check00",
    correct_answer="(C) Responses to questions in associated worksheet.",
    correct_feedback="Correct! Glad you're at least sort of paying attention.",
    incorrect_feedback="...Not quite, but a valiant attempt. No responses entered outside the Comments form will be recorded, and only the associated worksheet will be graded."
)

st.markdown("---")

st.markdown(" ### 0. Motivating Example")
st.write("""
         Throughout this introduction to data, we will use **housing data** as a running example. In particular, 
         we should have in mind the notion that the price of a house should be dependent upon one or more factors 
         in a predictable way. In particular, we will focus on the (overly) simple case where only the *size* of the house is taken
         into account.
""")

st.markdown("### 1. Housing Price as a Function of Size")
st.write("""
        Suppose for the moment that there is a function $p(x)$ which takes as input the size $x$ of a house in square feet,
        and outputs the price of the house in dollars. Specifically, let's assume
""")
st.latex(r"p(x) = 150x + 50")

quiz_card(question_title="1.1 **Checkpoint**",
          question_text="According to the given setup, how much does a 2000 square foot home cost?",
          options=[300050, 150000, 3000500, 250000],
          correct_answer=300050,
          key="check11",
          correct_feedback="Correct! $p(2000)=(150)(2000)+50 = 300,050$.",
          incorrect_feedback="Incorrect, please try again.")

st.write("""We've seen in class that information about a function is typically represented in three ways: as an **equation**,
         in a **table**, or visually through a **graph**. Let's take the above equation for $p(x)$ and use it to produce
         a table and graph.
         """)

st.markdown("#### 1.1 Defining a Function")

st.write("""
        We can define the price function using `def price(x):` (or whatever label you wish to give the
         function) and expressing the desired formula. 

""")
with st.expander("Code"):
    code = """
def price(x):
    price = 150*x+50
    return price
"""
    st.code(code, language="python")

def price(x):
    price = 150*x + 50
    return price

st.write("With this function, we can compute the price of a house of whatever size we want.")

col1, col2 = st.columns(2)

with col1:
    size = st.text_input(label="Size of house (numeric):", width=150, value=1000)
    enter = st.button("Compute Price")
with col2:
    st.write("Price:")
    if enter:
        try:
            size_numeric = float(size)
            if size_numeric <=0:
                st.warning("For better or worse, negative square footage doesn't make sense. Please enter a positive number.")
            else:
                p = price(size_numeric)
                st.write(f"${p:,.02f}")
        except ValueError:
            st.warning("Please enter a valid (positive) number in numeric format.")

st.markdown("#### 1.2 Storing and Plotting")
st.write("""
         The most typical structures in Python used for storing data in a table are `Numpy arrays` and `Pandas dataframes`. Pandas
         dataframes are useful for a variety of reasons, not the least of which is they allow for column names. Below, you can enter a
         list of *comma-separated* house sizes and/or generate a random list of house sizes.

         The visual difference between the arrays and the dataframes is slight (especially with the streamlit display options), but 
         being able to select a particular column and know what you're looking at is nice.
         """)

st.markdown("#### 1.2.1 Manually Create Dataset")
size_list_input = st.text_input("Enter list of house sizes (comma-separated):", value="1000, 2000, 3000")

try:
    size_list_input_parsed = [float(x.strip()) for x in size_list_input.split(",")]
    if contains_negative(size_list_input_parsed)>0:
        size_list_input_parsed=[]
        st.warning("Please enter valid (positive) sizes, separated by commas.")
    
except ValueError:
    size_list_input_parsed = []
    st.warning("Please enter valid (positive) sizes, separated by commas.")

if size_list_input_parsed:
    sizes_array = np.array(size_list_input_parsed)
    prices_array = price(sizes_array)
    data_array = np.column_stack((sizes_array, prices_array))

    with st.expander("View manual entry array"):
        st.write(data_array)

    with st.expander("View manual entry dataframe"):
        df = pd.DataFrame(data_array, columns=["Size", "Price"])
        st.dataframe(df)
    
    with st.expander("View plotted data"):
        fig, ax = plt.subplots()
        ax.plot(df['Size'], df['Price'], marker='o')
        ax.set_xlabel("Size (sq. ft.)")
        ax.set_ylabel("Price ($)")
        ax.set_title("House Price as Function of Size")
        st.pyplot(fig)

    with st.expander("Code for manual entry", expanded=False):
        code = f"""
#Create array from entered sizes:
sizes_array = np.array({size_list_input_parsed})
sizes_array ={np.array(size_list_input_parsed)}

#Calculate prices by applying price() function
#to sizes_array
prices_array = np.array({prices_array.tolist()})
prices_array = {price(sizes_array)}

#Combine the 1D arrays into a 2D array (i.e. a table)
#with 2 columns
data_array = np.column_stack((sizes_array, prices_array))

#Change the 'type' of the data_array to a DataFrame and
#name the columns appropriately
df = pd.DataFrame(data_array, columns=["Size", "Price"])
        """

        st.code(code, language="python")

    

st.markdown("#### 1.2.2 Generate Dataset")

st.write("""
        A couple of notes here:
         - The maximum number of houses to generate is set to 50.
         - The minimum house size $x_{min}$ is forced to satisfy $100\leq x_{min} \leq 2000$, with the maximum size $x_{max}$ then allowed 
         to be between $x_{min} + 1\leq x_{max} \leq x_{min} + 10000$.
         - To create a new data set, you must either press enter with your cursor in one of the editable boxes or click 'Generate'.
         - Each time 'Generate' is clicked, even if the parameters are the same, the generated points may be different. This is because
         the `random seed` has not been set.
""")
sizes_array_gen = np.array([])
prices_array_gen = np.array([])
data_array_gen = np.empty((0,2))

with st.form(key="gen_house_data"):
    number_to_generate = st.number_input("Number of houses:", min_value=1, max_value=50, step=1)
    min_size = st.number_input("Minimum possible size:", min_value=100, max_value=2000, step=50)
    max_size = st.number_input("Maximum possible size:", min_value=min_size+1, max_value=min_size+10000)
    generate_list = st.form_submit_button("Generate")

if generate_list:
    size_range = range(min_size, max_size+1)
    sizes_array_gen = np.random.choice(size_range, size=number_to_generate)
    prices_array_gen = price(sizes_array_gen)
    data_array_gen = np.column_stack((sizes_array_gen, prices_array_gen))

if data_array_gen.shape[0] > 0:
    with st.expander("View generated array"):
        st.write(data_array_gen)

    with st.expander("View generated dataframe"):
        df_gen = pd.DataFrame(data_array_gen, columns=["Size", "Price"])
        st.dataframe(df_gen)

    with st.expander("View plotted data"):
        fig, ax = plt.subplots()
        ax.plot(df_gen['Size'], df_gen['Price'], marker='o')
        ax.set_xlabel("Size (sq. ft.)")
        ax.set_ylabel("Price ($)")
        ax.set_title("House Price as Function of Size")
        st.pyplot(fig)

    with st.expander("Code for generating"):
        code = f"""
# Sample from viable range, number of samples determined
# by  input
size_range = range({min_size}, {max_size+1})
sizes_array_gen = np.random.choice({size_range}, size={number_to_generate})

# Apply the price() function to the array of sizes.
prices_array_gen = price(sizes_array_gen)
data_array_gen = np.column_stack((sizes_array_gen, prices_array_gen))

# Create a DataFrame from the 2D array. Label columns 'Size' and 'Price'
df_gen = pd.DataFrame(data_array_gen, columns=["Size", "Price"])

"""
        st.code(code, language="python")
else:
    st.info("Click 'Generate' to see the dataset.")

quiz_card(
          question_title="1.2.1 **Checkpoint**",
          question_text="""True or False: In this setup, based on the definition of $p(x)$ and the example graphs above,
                        it is possible for a 1500 square foot house to cost exactly \$225,000.""",
          options=["True", "False"],
          correct_answer="False",
          key="check121",
          correct_feedback="""Correct! There is no room for fluctuation with a function like $p(x)$; if we are
                              assuming houses are priced only by $p(x)$, then $p(1500)=$225,050$. Even a 50 dollar deviation
                              is not allowed.
                           """,
        incorrect_feedback="""
        Incorrect, please try again. The value of $p(1500)$ is \$225,050, and there is no way
         for it to match \$225,000 *exactly*.
        """
            )

quiz_card(
          question_title="1.2.2 **Checkpoint**",
          question_text="""Is it reasonable to expect every house in the world of a given size $x$ to cost the exact same amount?""",
          options=["Of course not!", "Idk, maybe?", "Yes, houses are priced with complete consistency."],
          correct_answer="Of course not!",
          key="check122",
          correct_feedback="""What astounding intuition you've demonstrated. Two houses of the same size $x$ may have drastically
                              different prices, not only because there are many other *tangible* factors (e.g. location, age, property) 
                              which we've yet to take into account, but also *intangible, random* factors (e.g. the mood of the realtor).
                           """,
        incorrect_feedback="""
        If you said 'Idk, maybe?', your conviction is lacking (but that's okay, you're forgiven). 
        If you said housing prices are consistent...visit https://www.zillow.com!
        """
            )

st.markdown("""
            ### 1.3 What's Wrong with Functions?
""")
st.write(
    """
    Based on the previous two Checkpoint questions, it's apparent that functions are **too rigid to match up with 
    real-world experience**. Let's give a definition to describe this.
    """)


with st.container(border=True):
    st.markdown("""<h5><u>Definition:</u></h5>""", unsafe_allow_html=True)
    st.markdown(
        """
        1. A function $f:\mathbb{R} \\to \mathbb{R}$ is **deterministic** if $f(x)$ is always the same value, i.e. 
        it is completely determined by its input.
        2. A function $f:\mathbb{R} \\to \mathbb{R}$ is **nondeterministic** if $f(x)$ may vary each time the function is called.
        """
    )
        

st.markdown(""" ##### Remarks:""")
st.markdown(
    """
    - In class, *all* functions we work with are deterministic. Per our definition of a function, $f(x)$ cannot be more than one value,
    hence a nondeterministic function is not really a function at all.
    - The reason nondeterministic functions are valuable is because they allow for *randomness* or *noise*.
    - An example of a nondeterministic function: drawing a card from a standard deck. The input is the state of the deck 
    (the arrangement of the cards); the output is whichever card is drawn. Even with the same setup (input), a different
    card can be drawn each time (different output).
    """
)

st.markdown("### 1.4 Real Housing Data")
st.markdown("""
         To see what housing prices might really look like when plotted against size, we can obtain a real dataset
         from Kaggle [here](https://www.kaggle.com/datasets/yasserh/housing-prices-dataset/data). Note this data is rather old,
        so if the prices seem unrealistic...chalk it up to inflation.
         """, unsafe_allow_html=True)

housing = pd.read_csv(HOUSING_DATA_FILE)

fig, ax = plt.subplots()
ax.scatter(housing['area'], housing['price'], color="purple")
ax.set_xlabel("Size (sq. ft.)")
ax.set_ylabel("Price ($)")
ax.set_title("House Price as Function of Size")

ax.yaxis.set_major_formatter(ScalarFormatter())
ax.ticklabel_format(style='plain', axis='y')

st.pyplot(fig)

st.write(
    """
    We can examine the data displayed above more closely. First, `value_counts()` can be used to display the number of times each
     value occurs in a given column. For instance, below we see that the dataset contains 24 houses of size 6000 square feet. The
     number of rows displayed is limited using `.head()`.
    """)
st.write(housing['area'].value_counts().head())

st.markdown(
    """
    To confirm our suspicions that the relationship between $x =$ house size and $p(x) =$ house price is *not* a well-defined function
    as defined in class, we can check the prices of all houses of size 6000 square feet and see they range from $2.87M - $9.68M.
    """)

housing_6k = housing[housing['area'] == 6000][['area', 'price']].reset_index(drop=True)
st.write(housing_6k)

with st.expander("Code for scatterplot, value_counts, and size restriction", expanded=False):
    code = f"""
#To use Housing.csv file stored locally: pd.read_csv()
housing = pd.read_csv("file_path/Housing.csv")

#To plot 'area' on x-axis, 'price' on y-axis
fig, ax = plt.subplots()
ax.scatter(housing['area'], housing['price'], color="purple")
ax.set_xlabel("Size (sq. ft.)")
ax.set_ylabel("Price ($)")
ax.set_title("House Price as Function of Size")

#Calculate the number of houses with given size; 
#.head() limits to 5 rows by default
housing['area'].value_counts().head()

#Restrict the housing DataFrame to only 'size'=6000, columns 'size' and 'price'
housing_6k = housing[housing['area'] == 6000][['area', 'price']].reset_index(drop=True)
            """
    st.code(code)

quiz_card(
    question_title="1.4.1 Checkpoint",
    question_text="Which of the following statements is correct?",
    options=["Since p(6000) takes on many different values, the data comes from a well-defined function.",
             "Since p(6000) takes on many different values, the data is faulty and must be cleaned so that p(6000) is only one value.",
             "Since p(6000) takes on many different values, price is a nondeterministic output of size."],
    correct_answer="Since p(6000) takes on many different values, price is a nondeterministic output of size.",
    key="check141",
    correct_feedback="Correct! The relationship between size and price cannot be described by a well-defined function.",
    incorrect_feedback="Incorrect, please try again."
)

st.markdown("### 1.5 The Role of Functions: Modeling")
st.markdown(r"""
Data from the real world is usually not described perfectly by a deterministic function; however, for
the purposes of **modeling** and/or **prediction**, deterministic functions are exactly what we want. Here is a couple
of reasons why:

1. **True relationship** or **signal**: We assume or believe that there is some underlying relationship between the input(s) and 
   output which is deterministic, and that is what we aim to describe. In the housing situation, this means 
   $p(x) = p_0(x) + N$, where  
   - $p(x)$ is the actual observed price given size $x$  
   - $p_0(x)$ is the deterministic/true relationship between size and price, sometimes called **signal**  
   - $N$ is noise or randomness associated with house pricing

2. **Reproducibility**: When we create a model of the price to be used for making predictions, we want 
   exactly one output which is determined by the size. 
""")

st.write(
    """
To visually represent a model, plotting the signal function $p_0(x)$ on the same axes as $p(x)$ is done whenever possible. Here,
use the parameters $m$ and $b$ to represent $p_0(x) = mx+b$. Please be patient, as it may take a moment for the plot to update.
""")

m = st.slider(label="Set $m$:", min_value=-500, max_value=2000, step=1, value=-500)
b = st.slider(label="Set $b$:", min_value=10000, max_value=5000000, step=100, value=10000)
size_range = np.arange(start=np.floor(housing["area"].min()), stop=np.ceil(housing["area"].max())+1, step=1)
p_0 = m*size_range + b

fig1, ax1 = plt.subplots()
ax1.scatter(housing['area'], housing['price'], color="purple")
ax1.plot(size_range, p_0, linestyle='--', color='red', linewidth=3)
ax1.set_xlabel("Size (sq. ft.)")
ax1.set_ylabel("Price ($)")
ax1.set_title("House Price as Function of Size")

ax1.yaxis.set_major_formatter(ScalarFormatter())
ax1.ticklabel_format(style='plain', axis='y')

st.pyplot(fig1)

quiz_card(
    question_title="1.5.1 Checkpoint",
    question_text="What does the parameter $m$ determine about the line?",
    options=["(A) The length of the line.", "(B) The y-intercept.", "(C) The slope of the line.", "(D) Both (B) and (C)."],
    key="check151",
    correct_answer="(C) The slope of the line.",
    correct_feedback="Correct! In slope-intercept form $y=mx+b$, $m$ is the slope while $b$ is the intercept.",
    incorrect_feedback="Incorrect, please try again. Think of slope-intercept form."
)


st.markdown("### Looking Forward")
st.markdown(
    """
Thank you for reading this introduction to data! If you have any questions or comments, please enter them using the form below.

In the next couple of demos, we will explore 'best fit' lines or curves in a quantitative sense in order to measure and compare performance of different
models. We will then apply these ideas in new contexts, not just to housing data.
    """
    )

with st.form("comment_form"):
    name = st.text_input("Name (optional)")
    comment = st.text_area("Comments, questions, or suggestions for future topics:")
    submit_comment = st.form_submit_button("Submit")
    if submit_comment and comment.strip():
        save_comment(name.strip(), comment.strip())
        st.success("Comment submitted!")
