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
st.sidebar.write(f"☀️ Sun hours: {sun_hours}h/day")

# System Configuration
st.sidebar.header("System Configuration")

# Voltage selection based on system size
voltage_options = {
    "12V (Small systems <1kW)": 12,
    "24V (Medium systems 1-3kW)": 24,
    "48V (Large systems >3kW)": 48
}
system_voltage_label = st.sidebar.selectbox("System Voltage", list(voltage_options.keys()))
system_voltage = voltage_options[system_voltage_label]
st.sidebar.info(f"Selected: {system_voltage}V - Higher voltage = better efficiency")

# Panel configuration
st.sidebar.subheader("Panel Configuration")
panel_config = st.sidebar.radio(
    "Wiring Configuration",
    ["Series (Boost Voltage)", "Parallel (Boost Current)", "Series-Parallel (Balanced)"]
)

# Panel selection
st.sidebar.header("Panel & Pricing")
panel_watts = st.sidebar.selectbox("Panel Type (W)", [400, 350, 300])
panel_voltage = st.sidebar.selectbox("Panel Voltage (V)", [12, 24, 36, 48])
panel_price = st.sidebar.number_input("Panel Price (₦)", value=100000, step=1000)
budget = st.sidebar.number_input("Budget (₦)", value=500000, step=1000)
client_name = st.sidebar.text_input("Client Name", value="Client A")

# Efficiency factors
st.sidebar.header("System Losses")
charge_controller_eff = st.sidebar.slider("Charge Controller Efficiency (%)", 85, 98, 95) / 100
inverter_eff = st.sidebar.slider("Inverter Efficiency (%)", 85, 98, 90) / 100
battery_eff = st.sidebar.slider("Battery Efficiency (%)", 80, 95, 85) / 100
wiring_loss = st.sidebar.slider("Wiring Loss (%)", 1, 10, 5) / 100

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
        # Account for all system losses
        total_system_efficiency = charge_controller_eff * inverter_eff * battery_eff * (1 - wiring_loss)
        
        # Adjusted daily energy accounting for losses
        adjusted_daily_kwh = daily_kwh / total_system_efficiency
        
        # System sizing with proper derating
        system_kw = (adjusted_daily_kwh / sun_hours) * 1.25  # 25% safety margin
        
        # Panel calculations based on configuration
        panels_needed = math.ceil((system_kw * 1000) / panel_watts)
        
        # Adjust panel count based on voltage and configuration
        if "Series" in panel_config:
            # Series: panels_in_series = system_voltage / panel_voltage
            panels_in_series = math.ceil(system_voltage / panel_voltage)
            strings_needed = math.ceil(panels_needed / panels_in_series)
            panels_needed = panels_in_series * strings_needed
        elif "Parallel" in panel_config:
            # Parallel: all panels at same voltage
            panels_needed = math.ceil(panels_needed)
        else:  # Series-Parallel
            panels_in_series = math.ceil(system_voltage / panel_voltage)
            strings_needed = math.ceil(panels_needed / panels_in_series)
            panels_needed = panels_in_series * strings_needed
        
        panel_cost = panels_needed * panel_price
        
        # Battery sizing based on voltage and autonomy
        battery_capacity_ah = (adjusted_daily_kwh * 1000 * 2) / system_voltage  # 2 days autonomy
        battery_capacity_kwh = (battery_capacity_ah * system_voltage) / 1000
        batteries_needed = math.ceil(battery_capacity_kwh / 5)  # 5kWh batteries
        battery_cost = batteries_needed * 200000
        
        # Charge controller sizing
        panel_current = (panels_needed * panel_watts) / system_voltage
        charge_controller_rating = math.ceil(panel_current * 1.25)  # 25% safety margin
        
        # Inverter sizing
        inverter_rating = math.ceil(total_watts * 1.3)  # 30% surge capacity
        
        total_cost = panel_cost + battery_cost
        
        st.header("System Recommendation")
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("System Size", f"{system_kw:.2f} kW")
        with col2:
            st.metric("System Voltage", f"{system_voltage}V")
        with col3:
            st.metric("Location", f"{location}")
        with col4:
            st.metric("Total Cost", f"₦{total_cost:,.0f}")
        
        # Component details
        st.subheader("System Components")
        
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Solar Panels (Module)**")
            st.write(f"• Panels: {panels_needed} x {panel_watts}W")
            st.write(f"• Configuration: {panel_config}")
            st.write(f"• Total Power: {panels_needed * panel_watts:,}W")
            st.write(f"• Cost: ₦{panel_cost:,.0f}")
            
            st.write("**Charge Controller**")
            st.write(f"• Rating: {charge_controller_rating}A")
            st.write(f"• Voltage: {system_voltage}V")
            st.write(f"• Efficiency: {charge_controller_eff*100:.0f}%")
        
        with col2:
            st.write("**Battery Bank**")
            st.write(f"• Capacity: {battery_capacity_kwh:.1f} kWh")
            st.write(f"• Batteries: {batteries_needed} x 5kWh")
            st.write(f"• Voltage: {system_voltage}V")
            st.write(f"• Autonomy: 2 days")
            st.write(f"• Cost: ₦{battery_cost:,.0f}")
            
            st.write("**Inverter**")
            st.write(f"• Rating: {inverter_rating}W")
            st.write(f"• Voltage: {system_voltage}V DC to 230V AC")
            st.write(f"• Efficiency: {inverter_eff*100:.0f}%")
        
        # System efficiency
        st.subheader("System Efficiency Analysis")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Overall Efficiency", f"{total_system_efficiency*100:.1f}%")
        with col2:
            st.metric("Daily Energy Loss", f"{(1-total_system_efficiency)*daily_kwh:.2f} kWh")
        with col3:
            st.metric("Effective Output", f"{daily_kwh:.2f} kWh")
        
        # Budget analysis with component costs
        total_with_components = total_cost + 200000  # Add controller and inverter
        if total_with_components <= budget:
            st.success(f"✅ Complete system within budget! Remaining: ₦{budget - total_with_components:,.0f}")
        else:
            st.error(f"❌ Over budget by ₦{total_with_components - budget:,.0f}")
            st.warning("💡 Consider: Lower voltage system, fewer panels, or reduced battery capacity")
        
        # Simple invoice
        st.subheader("Professional Invoice")
        st.write(f"**Client:** {client_name}")
        st.write(f"**Location:** {location} ({sun_hours}h sun/day)")
        st.write(f"**System Voltage:** {system_voltage}V")
        st.write(f"**Configuration:** {panel_config}")
        st.write("---")
        st.write(f"**Solar Panels ({panel_watts}W):** {panels_needed} x ₦{panel_price:,.0f} = ₦{panel_cost:,.0f}")
        st.write(f"**Batteries (5kWh, {system_voltage}V):** {batteries_needed} x ₦200,000 = ₦{battery_cost:,.0f}")
        st.write(f"**Charge Controller ({charge_controller_rating}A):** 1 x ₦50,000 = ₦50,000")
        st.write(f"**Inverter ({inverter_rating}W):** 1 x ₦150,000 = ₦150,000")
        st.write("---")
        st.write(f"**TOTAL SYSTEM COST: ₦{total_cost + 200000:,.0f}**")
        st.caption(f"System Efficiency: {total_system_efficiency*100:.1f}% | Daily Output: {daily_kwh:.2f} kWh")

else:
    st.info("Add appliances using the sidebar")

# Educational section
with st.expander("ℹ️ Solar System Design Guide"):
    st.write("**Four Key Components:**")
    st.write("1. **Solar Module (Panels):** Converts sunlight to DC electricity")
    st.write("2. **Charge Controller:** Regulates charging, prevents overcharge")
    st.write("3. **Battery Bank:** Stores energy for nighttime/cloudy days")
    st.write("4. **Inverter:** Converts DC to AC for household appliances")
    st.write("")
    st.write("**Voltage Selection:**")
    st.write("• 12V: Small systems (<1kW), RVs, boats")
    st.write("• 24V: Medium systems (1-3kW), homes")
    st.write("• 48V: Large systems (>3kW), better efficiency")
    st.write("")
    st.write("**Panel Configuration:**")
    st.write("• Series: Increases voltage (V₁ + V₂ + V₃...)")
    st.write("• Parallel: Increases current (I₁ + I₂ + I₃...)")
    st.write("• Series-Parallel: Balanced voltage and current")
    st.write("")
    st.write("**System Losses Accounted:**")
    st.write("• Charge controller efficiency (85-98%)")
    st.write("• Inverter efficiency (85-98%)")
    st.write("• Battery round-trip efficiency (80-95%)")
    st.write("• Wiring losses (1-10%)")