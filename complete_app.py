import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import calendar
import random
import base64
import json
import os

# Set page configuration
st.set_page_config(
    page_title="TrackWise - Student Subscription Manager",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# CSS to customize the app appearance
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #4CAF50;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.8rem;
        color: #2196F3;
        margin-bottom: 1rem;
    }
    .card {
        background-color: #1E1E1E;
        border-radius: 10px;
        padding: 20px;
        margin-bottom: 20px;
    }
    .metric-card {
        background-color: #2C2C2C;
        border-radius: 10px;
        padding: 15px;
        text-align: center;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
    }
    .keep {
        color: #4CAF50;
        font-weight: bold;
    }
    .pause {
        color: #FFC107;
        font-weight: bold;
    }
    .cancel {
        color: #F44336;
        font-weight: bold;
    }
    .low-usage {
        color: #F44336;
    }
    .medium-usage {
        color: #FFC107;
    }
    .high-usage {
        color: #4CAF50;
    }
</style>
""", unsafe_allow_html=True)

# Import visualization libraries with fallbacks
try:
    import matplotlib.pyplot as plt
except ImportError:
    pass

try:
    import plotly.express as px
    import plotly.graph_objects as go
except ImportError:
    pass

try:
    from sklearn.linear_model import LinearRegression
except ImportError:
    # Create a simple replacement for LinearRegression
    class LinearRegression:
        def fit(self, X, y):
            # Simple implementation
            n = len(X)
            x_mean = sum(X[:, 0]) / n
            y_mean = sum(y) / n
            
            numerator = sum((X[i, 0] - x_mean) * (y[i] - y_mean) for i in range(n))
            denominator = sum((X[i, 0] - x_mean) ** 2 for i in range(n))
            
            self.coef_ = [numerator / denominator if denominator != 0 else 0]
            self.intercept_ = y_mean - self.coef_[0] * x_mean
            return self
            
        def predict(self, X):
            return [self.intercept_ + self.coef_[0] * x[0] for x in X]
            
        def score(self, X, y):
            y_pred = self.predict(X)
            y_mean = sum(y) / len(y)
            ss_total = sum((y_i - y_mean) ** 2 for y_i in y)
            ss_res = sum((y_i - y_pred_i) ** 2 for y_i, y_pred_i in zip(y, y_pred))
            return 1 - (ss_res / ss_total) if ss_total != 0 else 0

try:
    import statsmodels.api as sm
except ImportError:
    # Create a simple module for prediction
    class SM:
        def add_constant(self, X):
            # Add a column of ones to the front of X
            return [[1] + list(x) for x in X]
            
        class OLS:
            def __init__(self, y, X):
                self.y = y
                self.X = X
            
            def fit(self):
                # Just use the first feature for a simple prediction
                return SimpleFit()
                
    class SimpleFit:
        def predict(self, features):
            # Simple fallback prediction
            return [1.5]  # Default to "Maintain" recommendation
            
    sm = SM()

# Helper functions
def create_sample_data():
    """Create sample data for student profile, subscriptions, and usage"""
    # Student profile
    profile = {
        'name': 'Liam O'Dogherty',
        'email': 'liam.odoherty5@mail.dcu.ie',
        'budget': 75.00,
        'avatar': 'https://avatars.dicebear.com/api/avataaars/alexj.svg'
    }
    
    # Subscription services with variety of usage patterns
    services = [
        {
            'name': 'Netflix',
            'cost': 15.99,
            'category': 'Entertainment',
            'billing_day': 5,
            'usage': [4.5, 3.8, 3.2, 2.5],  # declining usage
            'url': 'https://www.netflix.com',
            'logo': '🎬'
        },
        {
            'name': 'Spotify',
            'cost': 9.99,
            'category': 'Music',
            'billing_day': 12,
            'usage': [7.2, 8.1, 7.5, 8.4],  # stable high usage
            'url': 'https://www.spotify.com',
            'logo': '🎵'
        },
        {
            'name': 'YouTube Premium',
            'cost': 11.99,
            'category': 'Entertainment',
            'billing_day': 18,
            'usage': [5.5, 5.2, 4.8, 4.2],  # slightly declining
            'url': 'https://www.youtube.com',
            'logo': '▶️'
        },
        {
            'name': 'ChatGPT Plus',
            'cost': 20.00,
            'category': 'Productivity',
            'billing_day': 24,
            'usage': [2.2, 1.8, 1.2, 0.7],  # strongly declining
            'url': 'https://chat.openai.com',
            'logo': '🤖'
        },
        {
            'name': 'Amazon Prime',
            'cost': 14.99,
            'category': 'Shopping',
            'billing_day': 3,
            'usage': [1.2, 2.5, 3.8, 4.5],  # increasing usage
            'url': 'https://www.amazon.com',
            'logo': '📦'
        },
         {
            'name': 'Quill Bot',
            'cost': 19.99,
            'category': 'Productivity',
            'billing_day': 15,
            'usage': [5.8, 5.7, 5.9, 5.7],
            'url': 'https://www.quillbot.com',
            'logo': '🤖'
        },
        {
            'name': 'Disney+',
            'cost': 7.99,
            'category': 'Entertainment',
            'billing_day': 8,
            'usage': [3.5, 1.8, 0.5, 0.2],  # sharp decline
            'url': 'https://www.disneyplus.com',
            'logo': '✨'
        }
    ]
    
    # Create more detailed weekly usage data
    # Each service gets 16 weeks of data (4 months) to allow for better trend analysis
    extended_services = []
    for service in services:
        # Create base trend from the 4 weekly data points
        base_trend = service['usage']
        # Generate additional 12 weeks based on the trend direction 
        if service['name'] == 'Netflix':  # declining
            additional_weeks = [max(0.5, base_trend[-1] - 0.2 * i) for i in range(1, 13)]
        elif service['name'] == 'Spotify':  # stable high
            additional_weeks = [base_trend[-1] + random.uniform(-0.3, 0.3) for _ in range(12)]
        elif service['name'] == 'YouTube Premium':  # slightly declining
            additional_weeks = [max(0.5, base_trend[-1] - 0.1 * i) for i in range(1, 13)]
        elif service['name'] == 'ChatGPT Plus':  # strongly declining
            additional_weeks = [max(0.1, base_trend[-1] - 0.15 * i) for i in range(1, 13)]
        elif service['name'] == 'Amazon Prime':  # increasing
            additional_weeks = [min(10, base_trend[-1] + 0.2 * i) for i in range(1, 13)]
        elif service['name'] == 'Adobe Creative Cloud':  # stable
            additional_weeks = [base_trend[-1] + random.uniform(-0.2, 0.2) for _ in range(12)]
        elif service['name'] == 'Disney+':  # sharp decline
            additional_weeks = [max(0.1, base_trend[-1] - 0.1 * i) for i in range(1, 13)]
            
        # Add noise to make data more realistic
        extended_usage = [max(0, x + random.uniform(-0.3, 0.3)) for x in base_trend + additional_weeks]
        
        # Create extended service entry
        extended_service = service.copy()
        extended_service['extended_usage'] = extended_usage
        extended_services.append(extended_service)
    
    return profile, extended_services

def predict_usage_trend(usage_data):
    """Use linear regression to predict future usage trend"""
    if len(usage_data) < 2:
        return None, None, 0, "Insufficient data"
    
    try:
        weeks = np.array(range(len(usage_data))).reshape(-1, 1)
        usage = np.array(usage_data)
        
        # Simple fallback if LinearRegression fails
        try:
            model = LinearRegression()
            model.fit(weeks, usage)
            
            slope = model.coef_[0]
            
            # Predict next 4 weeks
            future_weeks = np.array(range(len(usage_data), len(usage_data) + 4)).reshape(-1, 1)
            predicted_usage = model.predict(future_weeks)
            
            # Calculate R-squared
            r_squared = model.score(weeks, usage)
        except Exception:
            # Fallback to simple trend calculation if scikit-learn fails
            # Calculate slope manually (simple linear regression)
            x_mean = np.mean(range(len(usage_data)))
            y_mean = np.mean(usage_data)
            
            numerator = sum([(i - x_mean) * (y - y_mean) for i, y in enumerate(usage_data)])
            denominator = sum([(i - x_mean)**2 for i in range(len(usage_data))])
            
            if denominator == 0:
                slope = 0
            else:
                slope = numerator / denominator
            
            intercept = y_mean - slope * x_mean
            
            # Predict manually
            predicted_usage = [slope * (len(usage_data) + i) + intercept for i in range(4)]
            
            # Simple R-squared calculation
            y_pred = [slope * i + intercept for i in range(len(usage_data))]
            ss_total = sum([(y - y_mean)**2 for y in usage_data])
            ss_residual = sum([(y - y_pred[i])**2 for i, y in enumerate(usage_data)])
            
            if ss_total == 0:
                r_squared = 0
            else:
                r_squared = 1 - (ss_residual / ss_total)
        
        # Determine trend direction
        if slope < -0.2:
            trend = "Strong Decline"
        elif slope < -0.05:
            trend = "Slight Decline"
        elif slope > 0.2:
            trend = "Strong Increase"
        elif slope > 0.05:
            trend = "Slight Increase"
        else:
            trend = "Stable"
        
        return slope, predicted_usage, r_squared, trend
    
    except Exception:
        # Absolute fallback if everything fails
        # Return neutral values
        return 0, [usage_data[-1]] * 4 if usage_data else [0] * 4, 0, "Neutral"

def recommend_action(service, usage_data, cost):
    """Recommend whether to keep, pause, or cancel a subscription"""
    # Use the provided regression model approach
    slope, predicted_usage, r_squared, trend = predict_usage_trend(usage_data)
    
    if slope is None:
        return "Insufficient Data"
    
    # Calculate average weekly usage over the last month
    recent_usage = usage_data[-4:] if len(usage_data) >= 4 else usage_data
    avg_weekly_usage = sum(recent_usage) / len(recent_usage)
    
    # Calculate cost per hour
    cost_per_hour = cost / avg_weekly_usage if avg_weekly_usage > 0 else float('inf')
    
    # Calculate predicted usage for next month
    next_month_prediction = predicted_usage[0] if len(predicted_usage) > 0 else 0
    
    # Prepare features for the model
    features = [1, avg_weekly_usage * 4, cost, cost_per_hour]  # Convert weekly to monthly hours
    
    # Calculate predicted score using coefficients (simplified from the provided model)
    # These coefficients are approximated based on the provided model code
    coefficients = [1.0, 0.15, -0.05, -0.5]  # [const, hours, cost, cost_per_hour]
    score = sum(c * f for c, f in zip(coefficients, features))
    
    # Convert score to recommendation
    if avg_weekly_usage < 1 or next_month_prediction < 0.5:
        return "Cancel ❌"
    elif cost_per_hour > 5 or (slope < -0.2 and next_month_prediction < 2):
        return "Pause ⏸"
    else:
        return "Keep ✅"

def get_usage_change(service):
    """Calculate the percentage change in usage over time"""
    usage = service['extended_usage']
    if len(usage) < 8:  # Need at least 8 weeks to calculate meaningful change
        return 0
    
    # Compare average of last 4 weeks to previous 4 weeks
    recent_avg = sum(usage[-4:]) / 4
    previous_avg = sum(usage[-8:-4]) / 4
    
    if previous_avg == 0:
        return 0
    
    return ((recent_avg - previous_avg) / previous_avg) * 100

def calculate_potential_savings(services):
    """Calculate potential savings from canceling low-usage subscriptions"""
    savings = 0
    for service in services:
        recommendation = recommend_action(
            service, 
            service['extended_usage'], 
            service['cost']
        )
        if recommendation == "Cancel ❌":
            savings += service['cost']
        elif recommendation == "Pause ⏸":
            savings += service['cost'] / 2  # Assume 50% savings from pausing
    
    return savings

def get_upcoming_billing_dates(services):
    """Get upcoming billing dates for the next 30 days"""
    today = datetime.now()
    upcoming_bills = []
    
    for service in services:
        billing_day = service['billing_day']
        
        # Calculate next billing date
        if billing_day >= today.day:
            # Billing date is in current month
            next_billing = datetime(today.year, today.month, billing_day)
        else:
            # Billing date is in next month
            if today.month == 12:
                next_billing = datetime(today.year + 1, 1, billing_day)
            else:
                next_billing = datetime(today.year, today.month + 1, billing_day)
        
        # Check if within the next 30 days
        days_until_billing = (next_billing - today).days
        if 0 <= days_until_billing <= 30:
            upcoming_bills.append({
                'service': service['name'],
                'logo': service['logo'],
                'date': next_billing.strftime("%B %d"),
                'cost': service['cost'],
                'days_left': days_until_billing
            })
    
    # Sort by days left
    upcoming_bills.sort(key=lambda x: x['days_left'])
    return upcoming_bills

def get_usage_status_color(hours):
    """Return color code based on usage"""
    if hours < 2:
        return "low-usage"
    elif hours < 5:
        return "medium-usage"
    else:
        return "high-usage"

def load_data():
    """Load data or create sample data if not exists"""
    if 'profile' not in st.session_state:
        profile, services = create_sample_data()
        st.session_state['profile'] = profile
        st.session_state['services'] = services
    
    return st.session_state['profile'], st.session_state['services']

# Navigation sidebar
def render_sidebar():
    with st.sidebar:
        st.image("https://avatars.dicebear.com/api/avataaars/trackwise.svg", width=100)
        st.title("TrackWise")
        st.subheader("Subscription Manager")
        
        # Navigation
        page = st.radio(
            "Navigation",
            ["🏠 Home", "📦 Subscriptions", "📉 Usage Trends", "🗓 Calendar", "⚙️ Settings"]
        )
        
        # Show total subscriptions and costs at bottom of sidebar
        st.markdown("---")
        profile, services = load_data()
        total_cost = sum(service['cost'] for service in services)
        st.markdown(f"**Total Subscriptions:** {len(services)}")
        st.markdown(f"**Monthly Cost:** ${total_cost:.2f}")
        st.markdown(f"**Budget:** ${profile['budget']:.2f}")
        
        # Budget progress bar
        budget_percentage = min(100, (total_cost / profile['budget']) * 100)
        st.progress(budget_percentage / 100)
        if budget_percentage >= 100:
            st.warning("⚠️ Over budget!")
        
        st.markdown("---")
        st.markdown("© 2025 TrackWise")
    
    return page

# Page content functions
def render_home():
    profile, services = load_data()
    
    # Header with profile info
    col1, col2 = st.columns([1, 3])
    with col1:
        st.image("https://avatars.dicebear.com/api/avataaars/alexj.svg", width=150)
    with col2:
        st.markdown(f"# Welcome, {profile['name']}!")
        st.markdown(f"📧 {profile['email']}")
        
    st.markdown("---")
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    total_cost = sum(service['cost'] for service in services)
    total_services = len(services)
    potential_savings = calculate_potential_savings(services)
    
    with col1:
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        st.markdown("### Total Monthly Cost")
        st.markdown(f"<div class='metric-value'>${total_cost:.2f}</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        st.markdown("### Active Subscriptions")
        st.markdown(f"<div class='metric-value'>{total_services}</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col3:
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        st.markdown("### Monthly Budget")
        st.markdown(f"<div class='metric-value'>${profile['budget']:.2f}</div>", unsafe_allow_html=True)
        budget_status = "Under Budget" if total_cost <= profile['budget'] else "Over Budget"
        budget_color = "keep" if total_cost <= profile['budget'] else "cancel"
        st.markdown(f"<div class='{budget_color}'>{budget_status}</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col4:
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        st.markdown("### Potential Savings")
        st.markdown(f"<div class='metric-value'>${potential_savings:.2f}</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Subscriptions by category donut chart
    col1, col2 = st.columns([3, 2])
    
    with col1:
        st.subheader("Subscription Spending by Category")
        
        # Prepare data for the pie chart
        categories = {}
        for service in services:
            category = service['category']
            if category in categories:
                categories[category] += service['cost']
            else:
                categories[category] = service['cost']
        
        category_df = pd.DataFrame({
            'Category': list(categories.keys()),
            'Cost': list(categories.values())
        })
        
        try:
            if 'px' in globals():  # Check if plotly express is available
                fig = px.pie(
                    category_df, 
                    values='Cost', 
                    names='Category',
                    hole=0.4,
                    color_discrete_sequence=px.colors.qualitative.Bold
                )
                fig.update_layout(
                    margin=dict(t=0, b=0, l=0, r=0),
                    showlegend=True,
                    legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5)
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                # Fallback to a simple table if plotly is unavailable
                st.dataframe(category_df)
        except Exception:
            # Fallback if visualization fails
            st.dataframe(category_df)
    
    with col2:
        st.subheader("Recent Usage Trends")
        
        # Calculate usage trends
        trends = []
        for service in services:
            slope, predicted, r_squared, trend_desc = predict_usage_trend(service['extended_usage'])
            if slope is not None:
                trends.append({
                    'name': service['name'],
                    'logo': service['logo'],
                    'trend': trend_desc,
                    'slope': slope
                })
        
        # Sort by most concerning (largest negative slope)
        trends.sort(key=lambda x: x['slope'])
        
        for trend in trends[:5]:  # Show top 5 most concerning trends
            trend_color = "cancel" if trend['slope'] < -0.1 else "pause" if trend['slope'] < 0 else "keep"
            st.markdown(f"{trend['logo']} **{trend['name']}:** <span class='{trend_color}'>{trend['trend']}</span>", unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Upcoming billing dates
    st.subheader("Upcoming Billing (Next 7 Days)")
    
    upcoming_bills = get_upcoming_billing_dates(services)
    upcoming_soon = [bill for bill in upcoming_bills if bill['days_left'] <= 7]
    
    if upcoming_soon:
        col_headers = st.columns([0.5, 2, 1.5, 1])
        col_headers[0].markdown("**Logo**")
        col_headers[1].markdown("**Service**")
        col_headers[2].markdown("**Date**")
        col_headers[3].markdown("**Amount**")
        
        for bill in upcoming_soon:
            cols = st.columns([0.5, 2, 1.5, 1])
            cols[0].markdown(f"{bill['logo']}")
            cols[1].markdown(f"{bill['service']}")
            cols[2].markdown(f"{bill['date']} ({bill['days_left']} days)")
            cols[3].markdown(f"${bill['cost']:.2f}")
    else:
        st.info("No bills due in the next 7 days.")

def render_subscriptions():
    profile, services = load_data()
    
    st.markdown("<div class='main-header'>Subscription Management</div>", unsafe_allow_html=True)
    st.markdown("Track and evaluate all your active subscriptions")
    
    total_cost = sum(service['cost'] for service in services)
    
    # Summary metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Monthly Cost", f"${total_cost:.2f}")
    with col2:
        st.metric("Number of Services", len(services))
    with col3:
        budget_diff = profile['budget'] - total_cost
        st.metric("Budget Remaining", f"${budget_diff:.2f}", 
                 delta=f"{budget_diff:.2f}", 
                 delta_color="normal" if budget_diff >= 0 else "inverse")
    
    st.markdown("---")
    
    # Subscription list with analysis
    st.subheader("Your Subscriptions")
    
    # Sort subscriptions by cost (highest first)
    sorted_services = sorted(services, key=lambda x: x['cost'], reverse=True)
    
    for service in sorted_services:
        with st.expander(f"{service['logo']} {service['name']} - ${service['cost']:.2f}/month"):
            col1, col2 = st.columns([3, 2])
            
            with col1:
                # Get recent usage data (last 4 weeks)
                recent_usage = service['extended_usage'][-4:]
                avg_weekly_usage = sum(recent_usage) / len(recent_usage)
                
                # Calculate cost per hour
                cost_per_hour = service['cost'] / avg_weekly_usage if avg_weekly_usage > 0 else float('inf')
                
                st.markdown(f"**Category:** {service['category']}")
                st.markdown(f"**Monthly Cost:** ${service['cost']:.2f}")
                st.markdown(f"**Average Weekly Usage:** {avg_weekly_usage:.1f} hours")
                st.markdown(f"**Cost per Hour:** ${cost_per_hour:.2f}")
                
                # Usage change
                usage_change = get_usage_change(service)
                change_color = "cancel" if usage_change < -10 else "pause" if usage_change < 0 else "keep"
                st.markdown(f"**Usage Trend:** <span class='{change_color}'>{usage_change:.1f}%</span>", unsafe_allow_html=True)
                
                # Recommendation
                recommendation = recommend_action(service, service['extended_usage'], service['cost'])
                rec_color = "keep" if recommendation == "Keep ✅" else "pause" if recommendation == "Pause ⏸" else "cancel"
                st.markdown(f"**Recommendation:** <span class='{rec_color}'>{recommendation}</span>", unsafe_allow_html=True)
                
                if recommendation != "Keep ✅":
                    savings = service['cost'] if recommendation == "Cancel ❌" else service['cost'] / 2
                    st.markdown(f"**Potential Savings:** ${savings:.2f}/month")
            
            with col2:
                # Mini usage chart
                weeks = list(range(1, len(service['extended_usage']) + 1))
                usage_data = service['extended_usage']
                
                try:
                    if 'px' in globals():
                        fig = px.line(
                            x=weeks[-8:],  # Last 8 weeks
                            y=usage_data[-8:],  # Last 8 weeks
                            markers=True,
                            title="Recent Weekly Usage (Hours)"
                        )
                        fig.update_layout(
                            xaxis_title="Week",
                            yaxis_title="Hours",
                            margin=dict(l=0, r=10, t=30, b=0),
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        # Fallback to a simple table
                        usage_df = pd.DataFrame({
                            'Week': weeks[-8:],
                            'Hours': usage_data[-8:]
                        })
                        st.dataframe(usage_df)
                except Exception:
                    # Show data in table format if visualization fails
                    st.write("Recent weekly usage (hours):")
                    for i, usage in enumerate(usage_data[-8:]):
                        st.write(f"Week {weeks[-8:][i]}: {usage:.1f} hours")
    
    st.markdown("---")
    
    # Subscription Value Matrix (Cost vs. Usage)
    st.subheader("Subscription Value Matrix")
    
    # Prepare data for scatter plot
    scatter_data = []
    for service in services:
        recent_usage = service['extended_usage'][-4:]
        avg_weekly_usage = sum(recent_usage) / len(recent_usage)
        scatter_data.append({
            'name': service['name'],
            'logo': service['logo'],
            'cost': service['cost'],
            'usage': avg_weekly_usage,
            'category': service['category'],
            'recommendation': recommend_action(service, service['extended_usage'], service['cost'])
        })
    
    scatter_df = pd.DataFrame(scatter_data)
    
    # Create quadrant lines for the plot
    avg_cost = scatter_df['cost'].mean()
    avg_usage = scatter_df['usage'].mean()
    
    try:
        if 'px' in globals():
            # Create the scatter plot
            fig = px.scatter(
                scatter_df,
                x='usage',
                y='cost',
                color='category',
                size='cost',
                hover_name='name',
                text='name',
                title="Cost vs. Usage Analysis",
                labels={'usage': 'Weekly Usage (Hours)', 'cost': 'Monthly Cost ($)'}
            )
            
            # Add quadrant lines
            fig.add_shape(
                type="line", line=dict(dash="dash", color="gray"),
                x0=avg_usage, y0=0, x1=avg_usage, y1=max(scatter_df['cost']) * 1.1
            )
            fig.add_shape(
                type="line", line=dict(dash="dash", color="gray"),
                x0=0, y0=avg_cost, x1=max(scatter_df['usage']) * 1.1, y1=avg_cost
            )
            
            # Update layout
            fig.update_layout(
                margin=dict(l=0, r=0, t=50, b=0),
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            # Fallback to table view
            st.dataframe(scatter_df[['name', 'cost', 'usage', 'category', 'recommendation']])
    except Exception:
        # Fallback if visualization fails
        st.write("Subscription value analysis:")
        st.dataframe(scatter_df[['name', 'cost', 'usage', 'category', 'recommendation']])
    
def render_usage_trends():
    profile, services = load_data()
    
    st.markdown("<div class='main-header'>Usage Trends & Forecasts</div>", unsafe_allow_html=True)
    st.markdown("Analyze your subscription usage patterns over time")
    
    # Service selector
    service_names = [f"{service['logo']} {service['name']}" for service in services]
    selected_service_name = st.selectbox("Select a subscription to analyze:", service_names)
    
    # Get selected service
    selected_service = next(
        (service for service in services 
         if f"{service['logo']} {service['name']}" == selected_service_name),
        services[0]  # Default to first service if none found
    )
    
    st.markdown("---")
    
    # Display service details
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader(f"{selected_service['logo']} {selected_service['name']}")
        st.markdown(f"**Category:** {selected_service['category']}")
        st.markdown(f"**Monthly Cost:** ${selected_service['cost']:.2f}")
        
        # Calculate metrics
        recent_usage = selected_service['extended_usage'][-4:]
        avg_weekly_usage = sum(recent_usage) / len(recent_usage)
        cost_per_hour = selected_service['cost'] / avg_weekly_usage if avg_weekly_usage > 0 else float('inf')
        
        st.markdown(f"**Average Weekly Usage:** {avg_weekly_usage:.1f} hours")
        st.markdown(f"**Cost per Hour:** ${cost_per_hour:.2f}")
        
        # Usage change
        usage_change = get_usage_change(selected_service)
        change_color = "cancel" if usage_change < -10 else "pause" if usage_change < 0 else "keep"
        st.markdown(f"**Monthly Usage Change:** <span class='{change_color}'>{usage_change:.1f}%</span>", unsafe_allow_html=True)
        
        # Recommendation
        recommendation = recommend_action(selected_service, selected_service['extended_usage'], selected_service['cost'])
        rec_color = "keep" if recommendation == "Keep ✅" else "pause" if recommendation == "Pause ⏸" else "cancel"
        st.markdown(f"**Recommendation:** <span class='{rec_color}'>{recommendation}</span>", unsafe_allow_html=True)
    
    with col2:
        # Detailed usage trend chart with forecast
        weeks = list(range(1, len(selected_service['extended_usage']) + 1))
        usage_data = selected_service['extended_usage']
        
        # Predict future usage
        slope, predicted_usage, r_squared, trend = predict_usage_trend(usage_data)
        
        try:
            if slope is not None and 'px' in globals():
                # Create forecast data
                future_weeks = list(range(len(usage_data) + 1, len(usage_data) + 5))  # 4 weeks into future
                
                # Create the plot
                fig = go.Figure()
                
                # Add historical data
                fig.add_trace(go.Scatter(
                    x=weeks,
                    y=usage_data,
                    mode='lines+markers',
                    name='Historical Usage',
                    line=dict(color='#2196F3'),
                    marker=dict(size=8)
                ))
                
                # Add forecast data
                fig.add_trace(go.Scatter(
                    x=future_weeks,
                    y=predicted_usage,
                    mode='lines+markers',
                    name='Forecasted Usage',
                    line=dict(color='#FF9800', dash='dash'),
                    marker=dict(size=8)
                ))
                
                # Add trend line
                all_weeks = weeks + future_weeks
                trend_line = [(slope * (w - 1)) + usage_data[0] for w in all_weeks]
                fig.add_trace(go.Scatter(
                    x=all_weeks,
                    y=trend_line,
                    mode='lines',
                    name='Trend Line',
                    line=dict(color='#F44336', dash='dot'),
                    opacity=0.5
                ))
                
                fig.update_layout(
                    title="Weekly Usage with 4-Week Forecast",
                    xaxis_title="Week",
                    yaxis_title="Hours Used",
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
                    margin=dict(l=0, r=0, t=50, b=0),
                    hovermode="x unified"
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Show trend analysis
                trend_color = "cancel" if slope < -0.2 else "pause" if slope < 0 else "keep"
                st.markdown(f"**Trend Analysis:** <span class='{trend_color}'>{trend}</span> (Slope: {slope:.3f})", unsafe_allow_html=True)
                st.markdown(f"**Forecast Confidence:** {r_squared:.2f} R²")
                
                if trend in ["Strong Decline", "Slight Decline"]:
                    if predicted_usage[-1] < 1:
                        st.warning("⚠️ Usage is predicted to drop below 1 hour per week. Consider canceling.")
                    elif predicted_usage[-1] < 2:
                        st.info("ℹ️ Usage is predicted to drop below 2 hours per week. Consider pausing.")
            else:
                # Fallback to a simple data table
                data = []
                for i, usage in enumerate(usage_data):
                    data.append({"Week": i+1, "Usage (hours)": usage})
                st.table(pd.DataFrame(data).tail(8))
                
                if slope is not None:
                    st.write(f"Trend: {trend} (Slope: {slope:.3f})")
                    st.write(f"Predicted usage for next 4 weeks: {', '.join([f'{u:.1f}' for u in predicted_usage])}")
        except Exception:
            # Simple table fallback
            st.write("Usage data:")
            data = []
            for i, usage in enumerate(usage_data):
                data.append({"Week": i+1, "Usage (hours)": usage})
            st.dataframe(pd.DataFrame(data).tail(8))
    
    st.markdown("---")
    
    # Usage patterns comparison
    st.subheader("Comparative Usage Analysis")
    
    # Create a DataFrame with all services' usage data
    comparison_data = []
    for service in services:
        # Get only the last 8 weeks of data for clarity
        recent_weeks = min(8, len(service['extended_usage']))
        for i in range(recent_weeks):
            week_idx = len(service['extended_usage']) - recent_weeks + i
            comparison_data.append({
                'Service': f"{service['logo']} {service['name']}",
                'Week': f"Week {week_idx+1}",
                'Usage': service['extended_usage'][week_idx],
                'WeekNum': week_idx+1
            })
    
    comp_df = pd.DataFrame(comparison_data)
    
    try:
        if 'px' in globals():
            # Create the comparison chart
            fig = px.line(
                comp_df,
                x='WeekNum',
                y='Usage',
                color='Service',
                markers=True,
                labels={'Usage': 'Hours Used', 'WeekNum': 'Week'},
                title="Recent Usage Comparison Across Services"
            )
            
            fig.update_layout(
                xaxis_title="Week",
                yaxis_title="Hours Used",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                margin=dict(l=0, r=0, t=50, b=0),
                hovermode="x unified"
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            # Pivot table fallback
            pivot_df = comp_df.pivot(index='WeekNum', columns='Service', values='Usage')
            st.dataframe(pivot_df)
    except Exception:
        # Simple fallback
        st.write("Comparative usage data:")
        pivot_df = comp_df.pivot(index='WeekNum', columns='Service', values='Usage')
        st.dataframe(pivot_df)
    
    # Weekly usage breakdown for selected service
    st.subheader(f"Weekly Usage Breakdown: {selected_service['name']}")
    
    # Create the weekly usage bar chart
    recent_usage = selected_service['extended_usage'][-8:]  # Last 8 weeks
    weeks = [f"Week {len(selected_service['extended_usage']) - 8 + i + 1}" for i in range(len(recent_usage))]
    
    try:
        if 'px' in globals():
            fig = px.bar(
                x=weeks,
                y=recent_usage,
                text=[f"{val:.1f}" for val in recent_usage],
                labels={'x': 'Week', 'y': 'Hours Used'},
                title=f"Weekly Usage Hours: {selected_service['name']}"
            )
            
            fig.update_layout(
                xaxis_title="Week",
                yaxis_title="Hours Used",
                margin=dict(l=0, r=0, t=50, b=0),
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            # Table fallback
            usage_df = pd.DataFrame({
                'Week': weeks,
                'Hours Used': recent_usage
            })
            st.dataframe(usage_df)
    except Exception:
        # Simple fallback
        for i, week in enumerate(weeks):
            st.write(f"{week}: {recent_usage[i]:.1f} hours")

def render_calendar():
    profile, services = load_data()
    
    st.markdown("<div class='main-header'>Billing Calendar</div>", unsafe_allow_html=True)
    st.markdown("Track upcoming subscription bills and payment dates")
    
    # Current month and year
    now = datetime.now()
    current_month = now.month
    current_year = now.year
    
    # Month selector
    months = ["January", "February", "March", "April", "May", "June", 
              "July", "August", "September", "October", "November", "December"]
    selected_month = st.selectbox("Month", months, index=current_month-1)
    selected_month_num = months.index(selected_month) + 1
    
    # Get days in selected month
    days_in_month = calendar.monthrange(current_year, selected_month_num)[1]
    
    st.markdown("---")
    
    # Calendar view
    st.subheader(f"Billing Calendar for {selected_month} {current_year}")
    
    # Create a mapping of day to services that bill on that day
    billing_days = {}
    for service in services:
        day = service['billing_day']
        if day not in billing_days:
            billing_days[day] = []
        billing_days[day].append(service)
    
    # Display calendar
    # Create a 7-column layout for the days of the week
    week_days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    cols = st.columns(7)
    for i, day_name in enumerate(week_days):
        cols[i].markdown(f"**{day_name}**")
    
    # Get the day of the week for the first day of the month
    first_day = datetime(current_year, selected_month_num, 1).weekday()  # Monday is 0
    
    # Create calendar grid
    day_counter = 1
    for week in range(6):  # Maximum 6 weeks in a month
        cols = st.columns(7)
        
        for i in range(7):
            if (week == 0 and i < first_day) or day_counter > days_in_month:
                # Empty cell
                cols[i].markdown("&nbsp;", unsafe_allow_html=True)
            else:
                # Day cell
                is_today = day_counter == now.day and selected_month_num == now.month
                day_style = "background-color: #E0E0E0; padding: 5px; border-radius: 5px;" if is_today else ""
                
                cols[i].markdown(f"<div style='{day_style}'>{day_counter}</div>", unsafe_allow_html=True)
                
                # Add billing services for this day
                if day_counter in billing_days:
                    for service in billing_days[day_counter]:
                        cols[i].markdown(
                            f"<div style='background-color: #87CEEB; color: white; padding: 5px; margin-top: 2px; border-radius: 3px;'>"
                            f"{service['logo']} {service['name']}<br/>"
                            f"<b>${service['cost']:.2f}</b></div>", 
                            unsafe_allow_html=True
                        )
                
                day_counter += 1
        
        # Break if we've displayed all days
        if day_counter > days_in_month:
            break
    
    st.markdown("---")
    
    # Upcoming bills
    st.subheader("Upcoming Bills (Next 30 Days)")
    
    upcoming_bills = get_upcoming_billing_dates(services)
    
    if upcoming_bills:
        col_headers = st.columns([0.5, 2, 1.5, 1, 1])
        col_headers[0].markdown("**Logo**")
        col_headers[1].markdown("**Service**")
        col_headers[2].markdown("**Date**")
        col_headers[3].markdown("**Amount**")
        col_headers[4].markdown("**Days Left**")
        
        # Add separator line
        st.markdown("<hr style='margin-top: 0; margin-bottom: 10px'>", unsafe_allow_html=True)
        
        total_upcoming = 0
        
        for bill in upcoming_bills:
            cols = st.columns([0.5, 2, 1.5, 1, 1])
            cols[0].markdown(f"{bill['logo']}")
            cols[1].markdown(f"{bill['service']}")
            cols[2].markdown(f"{bill['date']}")
            cols[3].markdown(f"${bill['cost']:.2f}")
            
            days_style = "color: red;" if bill['days_left'] <= 3 else "color: orange;" if bill['days_left'] <= 7 else ""
            cols[4].markdown(f"<span style='{days_style}'>{bill['days_left']} days</span>", unsafe_allow_html=True)
            
            total_upcoming += bill['cost']
        
        st.markdown("<hr style='margin-top: 10px;'>", unsafe_allow_html=True)
        st.markdown(f"**Total upcoming billing:** ${total_upcoming:.2f}")
    else:
        st.info("No bills due in the next 30 days.")
    
    st.markdown("---")
    
    # Monthly billing breakdown
    st.subheader("Monthly Billing Distribution")
    
    # Create a DataFrame for the monthly distribution
    days = list(range(1, 32))
    day_counts = [sum(1 for s in services if s['billing_day'] == day) for day in days]
    day_costs = [sum(s['cost'] for s in services if s['billing_day'] == day) for day in days]
    
    billing_df = pd.DataFrame({
        'Day': days,
        'Count': day_counts,
        'Cost': day_costs
    })
    
    try:
        if 'px' in globals():
            # Create the bar chart
            fig = px.bar(
                billing_df,
                x='Day',
                y='Cost',
                hover_data=['Count'],
                labels={'Cost': 'Billing Amount ($)', 'Day': 'Day of Month', 'Count': 'Number of Services'},
                title="Billing Distribution by Day of Month"
            )
            
            fig.update_layout(
                xaxis=dict(tickmode='linear', dtick=5),
                margin=dict(l=0, r=0, t=50, b=0),
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            # Table fallback
            st.dataframe(billing_df[billing_df['Count'] > 0])
    except Exception:
        # Simple fallback
        st.dataframe(billing_df[billing_df['Count'] > 0])

def render_settings():
    profile, services = load_data()
    
    st.markdown("<div class='main-header'>Settings</div>", unsafe_allow_html=True)
    st.markdown("Manage your profile, budget and notification preferences")
    
    # Profile settings
    st.subheader("Profile Information")
    
    col1, col2 = st.columns([1, 3])
    
    with col1:
        try:
            st.image("https://avatars.dicebear.com/api/avataaars/alexj.svg", width=150)
        except:
            st.write("👤 Profile Image")
    
    with col2:
        new_name = st.text_input("Name", profile['name'])
        new_email = st.text_input("Email", profile['email'])
    
    # Budget settings
    st.subheader("Budget Settings")
    
    current_budget = profile['budget']
    new_budget = st.slider("Monthly Subscription Budget ($)", 
                         min_value=0.0, 
                         max_value=200.0, 
                         value=float(current_budget),
                         step=5.0)
    
    # Calculate budget status
    total_cost = sum(service['cost'] for service in services)
    budget_diff = new_budget - total_cost
    budget_status = "Under Budget" if budget_diff >= 0 else "Over Budget"
    
    st.markdown(f"**Status:** {budget_status} (${abs(budget_diff):.2f} {'remaining' if budget_diff >= 0 else 'over'})")
    st.progress(min(1.0, total_cost / new_budget))
    
    # Notification preferences
    st.subheader("Notification Preferences")
    
    notify_upcoming = st.checkbox("Notify me about upcoming bills", value=True)
    notify_days = st.slider("Days in advance", 1, 7, 3, disabled=not notify_upcoming)
    
    notify_low_usage = st.checkbox("Notify me about low usage subscriptions", value=True)
    notify_budget = st.checkbox("Notify me when I exceed my budget", value=True)
    
    # Export data
    st.subheader("Data Management")
    
    col1, col2 = st.columns(2)
    with col1:
        st.button("Export Subscription Data")
    with col2:
        st.button("Reset All Data")
    
    # Save settings button
    if st.button("Save Settings"):
        # Update profile in session state
        profile['name'] = new_name
        profile['email'] = new_email
        profile['budget'] = new_budget
        
        st.session_state['profile'] = profile
        st.success("Settings saved successfully!")

# Main function
def main():
    try:
        # Render sidebar navigation
        page = render_sidebar()
        
        # Render selected page
        if page == "🏠 Home":
            render_home()
        elif page == "📦 Subscriptions":
            render_subscriptions()
        elif page == "📉 Usage Trends":
            render_usage_trends()
        elif page == "🗓 Calendar":
            render_calendar()
        elif page == "⚙️ Settings":
            render_settings()
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.info("Please try refreshing the page or contact support if the issue persists.")

if __name__ == "__main__":
    main()
