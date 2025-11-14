import streamlit as st
import math

# Simple appliance database
APPLIANCES = {
    "Refrigerator": 150,
    "Air Conditioner": 1200,
    "Ceiling Fan": 75,
    "LED Bulb": 10,
    "Television": 60,
    "Laptop": 65,
    "Washing Machine": 500,
    "Microwave": 1000,
    "Iron": 1200,
    "Water Pump": 750
}

st.title("Solar Calculator and Watt Recommender System")

# Initialize
if 'appliances' not in st.session_state:
    st.session_state.appliances = []

# Sidebar
st.sidebar.header("Add Appliances")
appliance = st.sidebar.selectbox("Select Appliance", list(APPLIANCES.keys()))
watts = st.sidebar.number_input("Watts", value=APPLIANCES[appliance], min_value=1)
hours = st.sidebar.number_input("Hours per day", value=8.0, min_value=0.1, step=0.5)
qty = st.sidebar.number_input("Quantity", value=1, min_value=1)

if st.sidebar.button("Add Appliance"):
    st.session_state.appliances.append({
        'name': appliance,
        'watts': watts,
        'hours': hours,
        'qty': qty
    })
    st.sidebar.success("Added!")

if st.sidebar.button("Clear All"):
    st.session_state.appliances = []
    st.sidebar.success("Cleared!")

# Location selection
st.sidebar.header("Location")
locations = {
    "Lagos": 5.2,
    "Abuja": 5.8,
    "Kano": 6.2,
    "Port Harcourt": 4.8,
    "Ibadan": 5.1,
    "Kaduna": 5.9,
    "Jos": 5.7
}
location = st.sidebar.selectbox("Select Location", list(locations.keys()))
sun_hours = locations[location]
st.sidebar.write(f"Sun hours: {sun_hours}h/day")

# Panel selection
st.sidebar.header("Panel & System")
panel_watts = st.sidebar.selectbox("Panel Type", [400, 350, 300])
panel_price = st.sidebar.number_input("Panel Price (₦)", value=100000, step=1000)
budget = st.sidebar.number_input("Budget (₦)", value=500000, step=1000)
client_name = st.sidebar.text_input("Client Name", value="Client A")

# Main area
st.header("Your Appliances")

if st.session_state.appliances:
    total_watts = 0
    daily_kwh = 0
    
    for i, app in enumerate(st.session_state.appliances):
        col1, col2, col3, col4, col5, col6 = st.columns([2, 1, 1, 1, 1, 1])
        
        app_total_watts = app['watts'] * app['qty']
        app_daily_kwh = (app['watts'] * app['hours'] * app['qty']) / 1000
        
        total_watts += app_total_watts
        daily_kwh += app_daily_kwh
        
        with col1:
            st.write(f"**{app['name']}**")
        with col2:
            st.write(f"{app['watts']}W")
        with col3:
            st.write(f"{app['hours']}h")
        with col4:
            st.write(f"x{app['qty']}")
        with col5:
            st.write(f"{app_total_watts}W")
        with col6:
            if st.button("❌", key=f"del_{i}"):
                st.session_state.appliances.pop(i)
                st.rerun()
    
    st.write("---")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Watts", f"{total_watts:,} W")
    with col2:
        st.metric("Daily Energy", f"{daily_kwh:.1f} kWh")
    with col3:
        st.metric("Location", f"{location} ({sun_hours}h)")
    
    # Calculate system
    if st.button("Calculate Solar System", type="primary"):
        system_kw = (daily_kwh / sun_hours) * 1.3  # Use location sun hours, 30% margin
        panels_needed = math.ceil((system_kw * 1000) / panel_watts)
        panel_cost = panels_needed * panel_price
        battery_cost = math.ceil(daily_kwh * 2 / 5) * 200000  # 2 days backup, 5kWh batteries
        total_cost = panel_cost + battery_cost
        
        st.header("System Recommendation")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("System Size", f"{system_kw:.1f} kW")
        with col2:
            st.metric("Panels Needed", f"{panels_needed}")
        with col3:
            st.metric("Location", f"{location}")
        with col4:
            st.metric("Total Cost", f"₦{total_cost:,.0f}")
        
        if total_cost <= budget:
            st.success("✅ Within budget!")
        else:
            st.error(f"❌ Over budget by ₦{total_cost - budget:,.0f}")
        
        # Simple invoice
        st.subheader("Invoice")
        st.write(f"**Client:** {client_name}")
        st.write(f"**Location:** {location} ({sun_hours}h sun/day)")
        st.write(f"**{panel_watts}W Panels:** {panels_needed} x ₦{panel_price:,.0f} = ₦{panel_cost:,.0f}")
        st.write(f"**Batteries (5kWh):** {math.ceil(daily_kwh * 2 / 5)} x ₦200,000 = ₦{battery_cost:,.0f}")
        st.write(f"**TOTAL: ₦{total_cost:,.0f}**")

else:
    st.info("Add appliances using the sidebar")