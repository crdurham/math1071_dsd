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
from matplotlib import colormaps
from matplotlib.colors import ListedColormap
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression, LogisticRegression

rng = np.random.default_rng(1)


ROOT = Path(__file__).resolve().parent.parent
HOUSING_DATA_FILE = ROOT / "data" / "Housing.csv"
student_data_file = ROOT / "data" / "student_data.csv"
ff_data_file = ROOT / "data" / "fantasy_data.csv"

student_data = pd.read_csv(student_data_file)
ff_data = pd.read_csv(ff_data_file)

def draw_line(p1, p2, style="-k", linewidth=1.5):
    plt.plot([p1[0], p2[0]], [p1[1], p2[1]], style, linewidth=linewidth)

def plot_data_points(X, idx):
    """Plots data points, coloring them by cluster index."""
    K = len(np.unique(idx))
    base_cmap = colormaps.get_cmap('Dark2')
    cmap = ListedColormap(base_cmap(np.linspace(0, 1, K)))
    
    plt.scatter(X[:, 0], X[:, 1], c=idx, cmap=cmap, s=45)
    
def plot_progress_kMeans(X, centroids, previous_centroids, idx, K, i):
    # Plot the examples
    plot_data_points(X, idx)
    
    # Plot the centroids as black 'x's
    plt.scatter(centroids[:, 0], centroids[:, 1], marker='x', c='k', linewidths=1, s=76)
    
    # Plot history of the centroids with lines
    for j in range(centroids.shape[0]):
        draw_line(centroids[j, :], previous_centroids[j, :])
    
    plt.title("Iteration number %d" %i)

def find_nearest_centroid(X,centroids):
    """
    Args:
        X -- design matrix with num rows = num samples, num cols = num features
        centroids -- matrix of centroids, K rows and n cols
    Output:
        idx -- array/vector v of length m; v^(i) = index of centroid closest to ith data point
    """
    K = np.array(centroids).shape[0]
    idx = np.zeros(X.shape[0], dtype=int)

    for i in range(X.shape[0]):
        distances = np.zeros(K)
        for j in range(K):
            distances[j] = np.sum((X[i]-centroids[j])**2)
        idx[i] = np.argmin(distances)
    return idx

def find_new_centroids(X, idx, K):
    """
    Args:
        X -- as above
        idx -- as in output from above, indices of closest centroid to each point
        K -- number of centroids
    """
    m, n = X.shape
    new_centroids = np.zeros((K,n)) 

    for k in range(K):
        points_in_k = X[idx == k] #Points currently assigned to centroid k
        if len(points_in_k)>0:
            new_centroids[k] = points_in_k.mean(axis = 0)
        
    return new_centroids
    
def Kmeans(X, initial_centroids, max_iters = 5, plot_progress = False, xlabel="X", ylabel="Y"):
    m, n = X.shape
    K = np.array(initial_centroids).shape[0]
    centroids = initial_centroids
    previous_centroids = centroids
    idx = np.zeros(m)

    figs = []

    for i in range(max_iters):
        idx = find_nearest_centroid(X, centroids=centroids)
        if plot_progress:
            fig, ax = plt.subplots()
            plot_progress_kMeans(X, centroids, previous_centroids, idx, K, i+1)
            ax.set_xlabel(xlabel=xlabel)
            ax.set_ylabel(ylabel=ylabel)
            figs.append(fig)
            previous_centroids = centroids
        centroids = find_new_centroids(X, idx, K)
    plt.show()
    return centroids, idx, figs

def Kmeans_cost(X, max_iters=30, K_values=[2,3,4,5,6,7, 8, 9, 10]):
    """
    Compute cost for multiple K values.
    """
    m = X.shape[0]
    costs = []

    for K in K_values:
        initial_centroids = [[rng.uniform(student_data['Quiz'].min(),student_data["Quiz"].max()),
                              rng.uniform(student_data['WA'].min(),student_data["WA"].max())] for _ in range(K)]
        
        final_centroids, idx, _ = Kmeans(
            X=X,
            initial_centroids=initial_centroids,
            max_iters=max_iters,
            plot_progress=False
        )
        
        total_cost = 0
        for i in range(m):
            total_cost += np.sum((X[i] - final_centroids[idx[i]])**2)
        avg_cost = total_cost / m
        costs.append(avg_cost)

    return costs


def highlight(text, color="#00cc6633"):
    """Return text string wrapped in HTML span with background highlight."""
    return f'<span style="background-color:{color}; padding:2px 4px; border-radius:3px"><b>{text}</b></span>'

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

st.title("Demo 6: Unsupervised Learning and K-Means Clustering")

st.subheader("Goals:")
st.markdown(
    f"""
    1. Understand the heuristic differences between supervised and unsupervised learning.
    2. Visualize how the {highlight("$K$-means learning algorithm")} clusters data.
    3. Acknowledge that the 'right' collection of clusters may depend on business needs or preferences.

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
            ### 0. Motivating Example: Student Performance Segmentation

            In this demo we will explore a real data set, not a synthetic
            or manufactured one. Last semester I (Cole) was the instructor of
            record of MATH 2410 at UConn, in which two components of the
            course were WebAssign Homework (denoted {highlight("WA")}) and Quizzes (denoted {highlight("Quiz")}).
            When presented with the WA average and Quiz average for approximately **65 students**,
            we would like to understand any patterns or structure within the data.
            In particular, can the students be placed in groups or clusters based
            on their performance? For instance, 'Top Students', 'Quiz Aces',
            'WebAssign Warriors', etc.

            Importantly, we are not tasked with predicting an output in the
            usual sense because the data provided has no labels! For a given
            student, their WA average and Quiz average represent inputs, and
            there is no output for us to predict.
            """, unsafe_allow_html=True)
st.markdown("---") 
st.markdown("""
            ### 1. Unsupervised Learning

            In previous demos, all data presented came in the form of
            input-output pairs $(x^{(1)},y^{(1)}),\\ldots,(x^{(m)},y^{(m)})$. Recall, for instance:
            - Demo 1: Input square footage, output house price
            - Demo 2: Input store information (patrons, expenditures), output
            total revenue
            - Demo 3: Input employee training time, output productivity.

            For each scenario, certain pieces of information are utilized to
            predict an output. This (roughly) defines **supervised learning**.

            Here, there is no output given to us (e.g. we are not trying to predict
            final exam score based on WA and Quiz). In such a scenario where
            our goal is to find structure within the data, it is called an
            **unsupervised learning** problem.""", unsafe_allow_html=True)

st.markdown(f"""
            #### 1.1 $K$-Means Clustering

            When presented with a collection of objects or people in a given
            category, it is natural to want to identify subcategories based on their
            measurable properties or traits. In a business setting this is frequently
            applied in {highlight("customer segmentation")}, whereby a company
            can identify different types of consumers among their customers and then
            cater to those groups. 

            When evaluating student performance, we will group students into {highlight("clusters")}
            based on performance in different components of the course. This can be
            achieved by an algorithm known as {highlight("$K$-means clustering")}, where
            $K$ is the number of groups. If we assume for now that $K=3$, the algorithm
            can be described in the following series of steps.
            1. Randomly initialize $K=3$ {highlight("centroids")} or group centers $c_1,c_2,c_3$,
            where each centroid is a point in the plane of the data, i.e. $c_1=(x_1,y_1)$, $c_2=(x_2,y_2)$,
            and $c_3=(x_3,y_3)$ where the $x$-coordinate is a quiz average and the $y$-coordinate is a
            WA average. {highlight("VISUAL")}
            2. For each data point, identify which centroid it is closest to. By doing this,
            each observed data point is placed in a group based on which centroid it is
            linked with.
            3. Update the centroid by finding the *average location of data points assigned
            to it*. This is done simply by taking the average within each coordinate,
            e.g. the average of $(1,2)$ and $(3,4)$ is $\\left(\\frac{{1+3}}{{2}}, \\frac{{2+4}}{{2}} \\right)=(2,3)$.
            4. Repeat steps 2 and 3 for a desired number of iterations or 
            until centroids no longer shift a significant amount when updating.

            Each of these steps will be visualized in the context of student
            performance on quizzes and WA.
            """, unsafe_allow_html=True)
st.markdown("---") 

st.markdown(f"""
            ### 2. Student Performance Clusters

            Below is a scatter plot with student quiz average on the
            horizontal axis and WA average on the vertical axis. Any
            point which is close to the top right corner indicates
            a student who performed well in each of the measured class 
            components. Clearly this class had a lot of strong performers,
            since the bulk of the data lands in the upper right!

            """, unsafe_allow_html=True)


fig4, ax4 = plt.subplots()
scatter = ax4.scatter(student_data['Quiz'], student_data['WA'], c='maroon', s=45)
ax4.set_xlabel("Quiz Average")
ax4.set_ylabel("WA Average")
ax4.set_title("Student Performance")
st.pyplot(fig4)

with st.expander("Student Performance Data"):
    st.write(student_data[['Quiz', 'WA']])

st.markdown(f"""
            We are truly presented with a cloud of data, nothing more,
            and that's all unsupervised learning requires. The point is
            to highlight data structure whether it is obvious to our eyes,
            our intuition, or not!

            Below, select a value of $K$ for the number of clusters and
            a value of $N$ for the number of iterations. Since the amount
            of data is small (only 68 students), the number of clusters
            and the number of iterations are restricted. For the plot of
            each iteration:
            - An 'x' indicates a centroid at that stage. 
            - The line attached to each centroid shows how the centroid
            has been updated since the previous iteration. For iteration 1
            there is no line because the centroids have only just been initialized.
            - Data points assigned to a given cluster are depicted in a common color.
            (If anyone has difficulty seeing the cluster colors for any reason, please
            let me know and I will try a different method.)
            - If at an iteration strictly after 1 any centroid does not have
            a line attached, it means that it remained the same after the update step.
            """, unsafe_allow_html=True)

K = st.slider("$K$:", min_value=2, max_value=6, step=1, value=3)
N = st.slider("$N$:", min_value=3, max_value=12, step=1, value=6)



initial_centers = [[rng.uniform(student_data["Quiz"].min(), student_data["Quiz"].max()),
                    rng.uniform(student_data["WA"].min(), student_data["WA"].max())] for _ in range(K)]

centroids, idx, figs = Kmeans(
    X=np.array(student_data[['Quiz', 'WA']]),
    initial_centroids=np.array(initial_centers),
    max_iters=N,
    plot_progress=True,
    xlabel="Quiz",
    ylabel="WA"
)
with st.expander("Student Performance Clusters: Plots of Iterations"):
    for fig in figs:
        st.pyplot(fig)

st.markdown("---") 
st.markdown(
    f"""
    ### 3. Choosing $K$

    An important question we must address: **what value of $K$ should we choose,
    and what should this decision be based on?** There are two adequate responses:
    - Choose a value of $K$ which suits the needs of your situation, or matches
    your *domain knowledge*/*intuition* about the problem.
    - Choose the value of $K$ at which the plot of the cost function
    $J$ (discussed below) exhibits an 'elbow'.

    Here, one natural value of $K$ appears to be 3. If the sliders
    above are set to $K=3$ and $N\geq 6$, we see that the algorithm
    identifies three groups: 
    1. The Top Students -- a cluster of data points in the top right
    corner, with all observed Quiz values above ~75, and all WA
    values above ~85.
    2. WebAssign Warriors -- students who generally performed better
    on WA than on Quiz, lying more to the top left of the dataset.
    3. Quiz Aces -- students who generally performed better on 
     Quiz than on WA, lying toward the bottom right of the dataset.

    This is a visually and intuitively reasonable conclusion. Depending
    on the demands of the situation being analyzed, this is sufficient.

    #### 3.1 Cost Function Elbow
     
    The cost function for the $K$-means clustering algorithm is given
    by the average distance from each point to the centroid it is assigned
    to. After running the algorithm for a given number of clusters $K$,
    the resulting configuration has an associated cost; we can plot the
    cost as a function of the number of clusters $K$. A general rule of thumb:
    pick the $K$ value at which the cost function *stops decreasing steeply,
    and begins to level off*. This is sometimes called the **cost function elbow**.
    
    For our scenario, the cost function is shown below.
    """, unsafe_allow_html=True
    )
costs = Kmeans_cost(X=np.array(student_data[['Quiz', 'WA']]),
    max_iters=N)

fig10, ax10 = plt.subplots()
ax10.plot([2,3,4,5,6,7, 8, 9, 10], costs, lw=2, color='k')
ax10.set_title("Clustering Cost Function")
ax10.set_xlabel("Number of Clusters K")
ax10.set_ylabel("Cost J(k)")
st.pyplot(fig10)

st.markdown(f""" 
            The cost function appears to level off at $K=6$, not the
            $K=3$ value we had determined previously. Returning to
            the sliders above, we can see why this happens if we set $K=6$:
            the outliers at $(30.4545,88.7318)$ and $(80.9091, 22.7162)$ in the data get 
            assigned to their own groups! Indeed, the Quiz Aces/Top Students
            are split into more refined classes as well.
            
            So which option is best, $K=3$ or $K=6$? There is no single right
            answer, and valid justification can be provided for either one. 
            """)

st.markdown("---") 
st.markdown("""
            ### 4. Important Caveat
            One technical detail regarding $K$-means clustering which
            will only be mentioned here in passing is that **the clusters
            obtained depend on the randomly initialized centroids** in step 1.
            That is to say, if we run the algorithm multiple times and begin
            with different centroids, the final clusters we end up with might
            be distinct (different centroids *and* different assignments).
            Typically, data scientists will run the algorithm many, many times
            and choose the clusters which yielded the lowest cost.

            In the above visuals, the initial centroids were randomly chosen,
            but are the same every time this page is loaded. Below is a different
            dataset on which we run $K=5$-means algorithm but with initial centroids
            chosen randomly each time the page is reloaded. See if by refreshing the page
            you get different clusters!

            (The dataset: rushing and receiving yards by NFL running backs with
            at least 75 rushing attempts and 30 targets in a season from 2015-2024.)
            """)

ff_rb_2020s = ff_data[(ff_data['FantPos']=='RB') &
                      (ff_data['Year'] >= 2015) &
                      (ff_data['Rushing_Att']>=75) &
                      (ff_data['Receiving_Tgt']>=30)][['Rushing_Yds', 'Receiving_Yds']]
fig_rb, ax_rb = plt.subplots()
ax_rb.scatter(ff_rb_2020s['Rushing_Yds'], ff_rb_2020s['Receiving_Yds'])
ax_rb.set_xlabel("Rushing Yards")
ax_rb.set_ylabel("Receiving Yards")
ax_rb.set_title("NFL RB Stats by Season 2015-2024")
st.pyplot(fig_rb)

K_rb = 5
N_rb = 100



initial_centers_rb = [[random.uniform(ff_rb_2020s["Rushing_Yds"].min(), ff_rb_2020s["Rushing_Yds"].max()),
                    random.uniform(ff_rb_2020s["Receiving_Yds"].min(), ff_rb_2020s["Receiving_Yds"].max())] for _ in range(K_rb)]

centroids_rb, idx_rb, figs_rb = Kmeans(
    X=np.array(ff_rb_2020s[['Rushing_Yds', 'Receiving_Yds']]),
    initial_centroids=np.array(initial_centers_rb),
    max_iters=N_rb,
    plot_progress=True,
    xlabel="Rushing Yards",
    ylabel="Receiving Yards"
)
with st.expander("RB Clusters: Final Groupings"):
    st.pyplot(figs_rb[N_rb-1])


st.markdown("---") 
st.markdown("### Looking Forward")
st.markdown(
    f"""
Thank you for reading this demo! 


In the next demo we will explore a 


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