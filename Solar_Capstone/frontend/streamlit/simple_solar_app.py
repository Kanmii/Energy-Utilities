import streamlit as st
import math

# Appliance database
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

# Sidebar - Input Method Selection
st.sidebar.header("Input Method")
input_method = st.sidebar.radio(
    "Choose calculation method:",
    ["Historical Consumption (Most Accurate)", "Appliance-Based Load (Backup & New Builds)"]
)

# A. Historical Consumption Method
if input_method == "Historical Consumption (Most Accurate)":
    st.sidebar.subheader("Historical Consumption")
    st.sidebar.caption("Most accurate - uses your actual electricity bills")
    
    consumption_period = st.sidebar.selectbox("Period", ["Monthly", "Annual"])
    
    if consumption_period == "Monthly":
        monthly_kwh = st.sidebar.number_input("Monthly Consumption (kWh)", value=300.0, min_value=0.0, step=10.0, help="Check your electricity bill for kWh consumed")
        daily_kwh = monthly_kwh / 30
        st.sidebar.success(f"Daily: {daily_kwh:.1f} kWh/day")
        st.sidebar.info(f"Yearly: {monthly_kwh * 12:,.0f} kWh/year")
    else:
        annual_kwh = st.sidebar.number_input("Annual Consumption (kWh)", value=3600.0, min_value=0.0, step=100.0, help="Total kWh over 12 months")
        daily_kwh = annual_kwh / 365
        st.sidebar.success(f"Daily: {daily_kwh:.1f} kWh/day")
        st.sidebar.info(f"Monthly: {annual_kwh / 12:.0f} kWh/month")
    
    total_watts = daily_kwh * 1000 / 24  # Approximate average watts

# B. Appliance-Based Load Method
else:
    st.sidebar.subheader("Add Appliances")
    appliance = st.sidebar.selectbox("Select Appliance", list(APPLIANCES.keys()))
    watts = st.sidebar.number_input("Watts", value=APPLIANCES[appliance], min_value=1)
    hours = st.sidebar.number_input("Daily Usage Hours", value=8.0, min_value=0.1, step=0.5)
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
    
    # Calculate from appliances
    if st.session_state.appliances:
        total_watts = sum(app['watts'] * app['qty'] for app in st.session_state.appliances)
        daily_kwh = sum((app['watts'] * app['hours'] * app['qty']) / 1000 for app in st.session_state.appliances)
    else:
        total_watts = 0
        daily_kwh = 0

# C. User Requirements & Constraints
st.sidebar.header("User Requirements")

# Desired Bill Offset
bill_offset = st.sidebar.slider("Desired Bill Offset (%)", 0, 100, 100, 5)
st.sidebar.info(f"System will cover {bill_offset}% of your energy needs")

# Backup Duration
backup_duration = st.sidebar.selectbox(
    "Backup Duration",
    ["No Backup (Grid-Tied)", "8 Hours (Critical Load)", "1 Day", "2 Days", "3 Days"]
)

# Convert backup duration to days
backup_days_map = {
    "No Backup (Grid-Tied)": 0,
    "8 Hours (Critical Load)": 0.33,
    "1 Day": 1,
    "2 Days": 2,
    "3 Days": 3
}
backup_days = backup_days_map[backup_duration]

# Location selection with coordinates
st.sidebar.header("Location & Solar Data")

locations = {
    "Lagos": {"sun_hours": 5.2, "lat": 6.5244, "lon": 3.3792},
    "Abuja": {"sun_hours": 5.8, "lat": 9.0765, "lon": 7.3986},
    "Kano": {"sun_hours": 6.2, "lat": 12.0022, "lon": 8.5920},
    "Port Harcourt": {"sun_hours": 4.8, "lat": 4.8156, "lon": 7.0498},
    "Ibadan": {"sun_hours": 5.1, "lat": 7.3775, "lon": 3.9470},
    "Kaduna": {"sun_hours": 5.9, "lat": 10.5105, "lon": 7.4165},
    "Jos": {"sun_hours": 5.7, "lat": 9.8965, "lon": 8.8583}
}

location_method = st.sidebar.radio("Location Input", ["Select City", "Enter Coordinates"])

if location_method == "Select City":
    location = st.sidebar.selectbox("Select Location", list(locations.keys()))
    sun_hours = locations[location]["sun_hours"]
    latitude = locations[location]["lat"]
    longitude = locations[location]["lon"]
    st.sidebar.success(f"Coordinates: {latitude:.4f}°N, {longitude:.4f}°E")
else:
    location = "Custom Location"
    latitude = st.sidebar.number_input("Latitude (°)", value=9.0765, min_value=-90.0, max_value=90.0, step=0.0001, format="%.4f")
    longitude = st.sidebar.number_input("Longitude (°)", value=7.3986, min_value=-180.0, max_value=180.0, step=0.0001, format="%.4f")
    sun_hours = 5.5 + (latitude - 9) * 0.15
    sun_hours = max(4.5, min(6.5, sun_hours))
    st.sidebar.success(f"Location: {latitude:.4f}°N, {longitude:.4f}°E")

st.sidebar.info(f"Sun hours: {sun_hours:.1f}h/day")

# Environmental Factors
st.sidebar.subheader("Environmental Factors")

# Temperature input
st.sidebar.write("**Ambient Temperature**")
temp_max = st.sidebar.number_input("Max Temperature (°C)", value=35.0, min_value=0.0, max_value=50.0, step=0.5)
temp_min = st.sidebar.number_input("Min Temperature (°C)", value=22.0, min_value=0.0, max_value=50.0, step=0.5)
temp_avg = (temp_max + temp_min) / 2
st.sidebar.info(f"Average: {temp_avg:.1f}°C")

# Temperature derating calculation
if temp_avg > 25:
    temp_derating = 1 - ((temp_avg - 25) * 0.004)
else:
    temp_derating = 1.0
temp_derating = max(0.85, min(1.0, temp_derating))
st.sidebar.caption(f"Temp derating: {temp_derating:.1%} (panels lose ~0.4%/°C above 25°C)")

# Weather patterns
st.sidebar.write("**Weather Patterns**")
cloud_cover = st.sidebar.slider("Cloud Cover (%)", 0, 100, 30, 5, help="Average annual cloud cover")
dust_level = st.sidebar.selectbox("Dust/Pollution Level", ["Low", "Medium", "High"])
humidity = st.sidebar.slider("Humidity (%)", 0, 100, 60, 5)

# Calculate environmental factor
cloud_factor = 1 - (cloud_cover / 200)
dust_factor = {"Low": 0.98, "Medium": 0.95, "High": 0.90}[dust_level]
humidity_factor = 1 - (max(0, humidity - 60) / 400)
environmental_factor = cloud_factor * dust_factor * humidity_factor * temp_derating

st.sidebar.write("**System Loss Factors:**")
st.sidebar.write(f"• Temperature: {temp_derating:.1%}")
st.sidebar.write(f"• Cloud cover: {cloud_factor:.1%}")
st.sidebar.write(f"• Dust/pollution: {dust_factor:.1%}")
st.sidebar.write(f"• Humidity: {humidity_factor:.1%}")
st.sidebar.success(f"Overall efficiency: {environmental_factor:.1%}")

# Physical Site Features
st.sidebar.header("Physical Site Features")

# Roof area
st.sidebar.write("**Usable Roof Area**")
area_unit = st.sidebar.radio("Unit", ["m²", "ft²"])
if area_unit == "m²":
    roof_area = st.sidebar.number_input("Roof Area (m²)", value=50.0, min_value=1.0, step=1.0)
    roof_area_m2 = roof_area
else:
    roof_area = st.sidebar.number_input("Roof Area (ft²)", value=538.0, min_value=1.0, step=1.0)
    roof_area_m2 = roof_area * 0.092903

st.sidebar.info(f"Usable area: {roof_area_m2:.1f} m²")

# Roof tilt/pitch
st.sidebar.write("**Roof Tilt/Pitch**")
roof_tilt = st.sidebar.slider("Tilt Angle (°)", 0, 90, 15, 1, help="0° = flat, 90° = vertical")

# Calculate tilt factor (optimal is typically latitude angle)
if location_method == "Select City":
    optimal_tilt = latitude
else:
    optimal_tilt = latitude

tilt_difference = abs(roof_tilt - optimal_tilt)
if tilt_difference <= 15:
    tilt_factor = 1.0
else:
    tilt_factor = 1 - (tilt_difference - 15) * 0.01
tilt_factor = max(0.85, min(1.0, tilt_factor))

st.sidebar.caption(f"Optimal tilt: {optimal_tilt:.0f}° | Efficiency: {tilt_factor:.1%}")

# Roof orientation (azimuth)
st.sidebar.write("**Roof Orientation (Azimuth)**")
azimuth_direction = st.sidebar.selectbox(
    "Direction",
    ["South (180°)", "South-East (135°)", "South-West (225°)", "East (90°)", "West (270°)", "North (0°)", "Custom"]
)

azimuth_map = {
    "South (180°)": 180,
    "South-East (135°)": 135,
    "South-West (225°)": 225,
    "East (90°)": 90,
    "West (270°)": 270,
    "North (0°)": 0
}

if azimuth_direction == "Custom":
    azimuth = st.sidebar.number_input("Azimuth Angle (°)", value=180, min_value=0, max_value=360, step=1)
else:
    azimuth = azimuth_map[azimuth_direction]

# Calculate azimuth factor (180° South is optimal in Northern hemisphere)
azimuth_deviation = abs(azimuth - 180)
if azimuth_deviation <= 45:
    azimuth_factor = 1.0
else:
    azimuth_factor = 1 - (azimuth_deviation - 45) * 0.005
azimuth_factor = max(0.70, min(1.0, azimuth_factor))

st.sidebar.caption(f"Azimuth: {azimuth}° | Efficiency: {azimuth_factor:.1%}")

# Shading analysis
st.sidebar.write("**Shading Obstacles**")
shading_level = st.sidebar.select_slider(
    "Shading Level",
    options=["None", "Minimal (<10%)", "Light (10-25%)", "Moderate (25-50%)", "Heavy (>50%)"],
    value="Minimal (<10%)"
)

shading_factors = {
    "None": 1.0,
    "Minimal (<10%)": 0.95,
    "Light (10-25%)": 0.85,
    "Moderate (25-50%)": 0.65,
    "Heavy (>50%)": 0.40
}
shading_factor = shading_factors[shading_level]

st.sidebar.write("**Shading Sources:**")
has_trees = st.sidebar.checkbox("Trees", value=False)
has_buildings = st.sidebar.checkbox("Nearby buildings", value=False)
has_chimneys = st.sidebar.checkbox("Chimneys/vents", value=False)

shading_sources = []
if has_trees:
    shading_sources.append("Trees")
if has_buildings:
    shading_sources.append("Buildings")
if has_chimneys:
    shading_sources.append("Chimneys")

if shading_sources:
    st.sidebar.warning(f"Obstacles: {', '.join(shading_sources)}")

st.sidebar.caption(f"Shading factor: {shading_factor:.1%}")

# Combined site factor
site_factor = tilt_factor * azimuth_factor * shading_factor
st.sidebar.success(f"Overall site efficiency: {site_factor:.1%}")

# Inverter Type Selection
st.sidebar.header("Inverter Configuration")

# Intelligent inverter recommendation
if shading_factor < 0.85 or has_trees or has_buildings:
    recommended_inverter = "Microinverters"
    recommendation_reason = "Complex shading detected"
elif backup_days > 0:
    recommended_inverter = "Hybrid"
    recommendation_reason = "Battery storage required"
else:
    recommended_inverter = "String"
    recommendation_reason = "Simple, cost-effective"

st.sidebar.info(f"💡 Recommended: {recommended_inverter} ({recommendation_reason})")

inverter_type = st.sidebar.selectbox(
    "Select Inverter Type",
    ["String Inverter", "Microinverters", "Hybrid Inverter"],
    index=["String Inverter", "Microinverters", "Hybrid Inverter"].index(recommended_inverter if "Inverter" in recommended_inverter else recommended_inverter + " Inverter") if recommended_inverter != "Microinverters" else 1
)

# Inverter characteristics
inverter_specs = {
    "String Inverter": {
        "efficiency": 0.97,
        "cost_per_kw": 50000,
        "pros": ["Lowest cost", "Simple installation", "Proven technology"],
        "cons": ["Single point of failure", "Poor with shading", "No panel-level monitoring"],
        "best_for": "Unshaded roofs, grid-tied systems"
    },
    "Microinverters": {
        "efficiency": 0.96,
        "cost_per_kw": 80000,
        "pros": ["Panel-level optimization", "Excellent with shading", "No single point of failure", "Panel-level monitoring"],
        "cons": ["Higher cost", "More components to maintain"],
        "best_for": "Complex shading, maximum production"
    },
    "Hybrid Inverter": {
        "efficiency": 0.95,
        "cost_per_kw": 100000,
        "pros": ["Battery integration", "Grid + battery backup", "Smart energy management"],
        "cons": ["Highest cost", "More complex"],
        "best_for": "Off-grid, battery backup systems"
    }
}

inverter_info = inverter_specs[inverter_type]
inverter_efficiency = inverter_info["efficiency"]

with st.sidebar.expander(f"ℹ️ {inverter_type} Details"):
    st.write(f"**Efficiency:** {inverter_efficiency:.1%}")
    st.write(f"**Cost:** ₦{inverter_info['cost_per_kw']:,.0f}/kW")
    st.write("**Pros:**")
    for pro in inverter_info["pros"]:
        st.write(f"• {pro}")
    st.write("**Cons:**")
    for con in inverter_info["cons"]:
        st.write(f"• {con}")
    st.write(f"**Best for:** {inverter_info['best_for']}")

# Panel & Budget
st.sidebar.header("Panel & Budget")
panel_watts = st.sidebar.selectbox("Panel Type (W)", [400, 350, 300])
panel_price = st.sidebar.number_input("Panel Price (₦)", value=100000, step=1000)
budget = st.sidebar.number_input("Budget (₦)", value=500000, step=1000)
client_name = st.sidebar.text_input("Client Name", value="Client A")

# Main Display Area
st.header("Energy Requirements")

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Total Load", f"{total_watts:,.0f} W")
with col2:
    st.metric("Daily Energy", f"{daily_kwh:.1f} kWh")
with col3:
    st.metric("Bill Offset", f"{bill_offset}%")

# Show appliances if using appliance-based method
if input_method == "Appliance-Based Load (Backup & New Builds)" and st.session_state.appliances:
    st.subheader("Your Appliances")
    for i, app in enumerate(st.session_state.appliances):
        col1, col2, col3, col4, col5, col6 = st.columns([2, 1, 1, 1, 1, 1])
        
        app_total_watts = app['watts'] * app['qty']
        app_daily_kwh = (app['watts'] * app['hours'] * app['qty']) / 1000
        
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

# Calculate System Button
if st.button("Calculate Solar System", type="primary", use_container_width=True):
    if daily_kwh == 0:
        st.error("Please enter consumption data or add appliances first.")
    else:
        # Apply bill offset
        adjusted_daily_kwh = daily_kwh * (bill_offset / 100)
        
        # System sizing with environmental and site derating
        combined_efficiency = environmental_factor * site_factor
        system_kw = (adjusted_daily_kwh / sun_hours) * 1.3 / combined_efficiency  # 30% safety margin
        panels_needed = math.ceil((system_kw * 1000) / panel_watts)
        panel_cost = panels_needed * panel_price
        
        # Battery sizing based on backup duration
        if backup_days > 0:
            battery_capacity_kwh = adjusted_daily_kwh * backup_days * 1.2  # 20% safety margin
            batteries_needed = math.ceil(battery_capacity_kwh / 5)  # 5kWh batteries
            battery_cost = batteries_needed * 200000
        else:
            battery_capacity_kwh = 0
            batteries_needed = 0
            battery_cost = 0
        
        # Inverter cost
        inverter_cost = system_kw * inverter_info["cost_per_kw"]
        
        total_cost = panel_cost + battery_cost + inverter_cost
        
        st.header("System Recommendation")
        
        # Check if panels fit on roof
        panel_area = 2.0  # Typical panel area in m² (400W panel ~2m²)
        required_area = panels_needed * panel_area
        area_sufficient = required_area <= roof_area_m2
        
        # System Overview
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("System Size", f"{system_kw:.2f} kW")
        with col2:
            st.metric("Panels Needed", f"{panels_needed}")
        with col3:
            st.metric("Battery Capacity", f"{battery_capacity_kwh:.1f} kWh" if backup_days > 0 else "Grid-Tied")
        with col4:
            st.metric("Total Cost", f"₦{total_cost:,.0f}")
        
        # Roof space check
        if not area_sufficient:
            st.error(f"⚠️ Insufficient roof space! Need {required_area:.1f} m², have {roof_area_m2:.1f} m²")
            st.info(f"💡 Reduce to {int(roof_area_m2 / panel_area)} panels or use higher wattage panels")
        else:
            st.success(f"✅ Roof space sufficient: {required_area:.1f} m² used of {roof_area_m2:.1f} m² available")
        
        # Detailed Breakdown
        st.subheader("System Details")
        
        # Environmental impact
        st.subheader("Environmental Analysis")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Avg Temperature", f"{temp_avg:.1f}°C")
        with col2:
            st.metric("Temp Derating", f"{temp_derating:.1%}")
        with col3:
            st.metric("Cloud Cover", f"{cloud_cover}%")
        with col4:
            st.metric("Overall Efficiency", f"{environmental_factor:.1%}")
        
        st.write("**Environmental Factors Applied:**")
        st.write(f"• Temperature range: {temp_min}°C - {temp_max}°C (avg {temp_avg:.1f}°C)")
        st.write(f"• Panel efficiency loss from heat: {(1-temp_derating)*100:.1f}%")
        st.write(f"• Cloud cover impact: {cloud_cover}% reduces output by {(1-cloud_factor)*100:.1f}%")
        st.write(f"• Dust/pollution level: {dust_level} ({dust_factor:.1%} efficiency)")
        st.write(f"• Humidity: {humidity}% ({humidity_factor:.1%} efficiency)")
        st.write(f"• Combined environmental factor: {environmental_factor:.1%}")
        st.info(f"🌡️ System sized with {(1-environmental_factor)*100:.1f}% environmental derating")
        
        # Site analysis
        st.subheader("Site Suitability Analysis")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Roof Area", f"{roof_area_m2:.1f} m²")
        with col2:
            st.metric("Tilt Efficiency", f"{tilt_factor:.1%}")
        with col3:
            st.metric("Orientation", f"{azimuth}°")
        with col4:
            st.metric("Shading Impact", f"{shading_factor:.1%}")
        
        st.write("**Physical Site Factors:**")
        st.write(f"• Roof area: {roof_area_m2:.1f} m² ({roof_area:.1f} {area_unit})")
        st.write(f"• Required area: {required_area:.1f} m² for {panels_needed} panels")
        st.write(f"• Roof tilt: {roof_tilt}° (optimal: {optimal_tilt:.0f}°) - {tilt_factor:.1%} efficiency")
        st.write(f"• Orientation: {azimuth}° azimuth - {azimuth_factor:.1%} efficiency")
        st.write(f"• Shading: {shading_level} - {shading_factor:.1%} efficiency")
        if shading_sources:
            st.write(f"• Obstacles: {', '.join(shading_sources)}")
        st.write(f"• Combined site factor: {site_factor:.1%}")
        st.info(f"🏗️ System oversized by {(1/site_factor - 1)*100:.1f}% to compensate for site conditions")
        
        st.subheader("System Components")
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Energy Analysis**")
            st.write(f"• Total Daily Load: {daily_kwh:.1f} kWh")
            st.write(f"• System Coverage: {bill_offset}% ({adjusted_daily_kwh:.1f} kWh)")
            st.write(f"• Grid Dependency: {100-bill_offset}% ({daily_kwh - adjusted_daily_kwh:.1f} kWh)")
            st.write(f"• Location: {location} ({sun_hours}h sun/day)")
            
            st.write("")
            st.write("**Solar Panels**")
            st.write(f"• Type: {panel_watts}W panels")
            st.write(f"• Quantity: {panels_needed} panels")
            st.write(f"• Total Capacity: {panels_needed * panel_watts:,}W")
            st.write(f"• Cost: ₦{panel_cost:,.0f}")
            
            st.write("")
            st.write(f"**{inverter_type}**")
            st.write(f"• Capacity: {system_kw:.2f} kW")
            st.write(f"• Efficiency: {inverter_efficiency:.1%}")
            st.write(f"• Cost: ₦{inverter_cost:,.0f}")
        
        with col2:
            st.write("**Backup Configuration**")
            st.write(f"• Duration: {backup_duration}")
            if backup_days > 0:
                st.write(f"• Battery Capacity: {battery_capacity_kwh:.1f} kWh")
                st.write(f"• Batteries: {batteries_needed} x 5kWh")
                st.write(f"• Cost: ₦{battery_cost:,.0f}")
            else:
                st.write("• Type: Grid-Tied (No Battery)")
                st.write("• Note: System requires grid connection")
            
            st.write("")
            st.write("**Financial Summary**")
            st.write(f"• Budget: ₦{budget:,.0f}")
            st.write(f"• System Cost: ₦{total_cost:,.0f}")
            if total_cost <= budget:
                st.write(f"• Remaining: ₦{budget - total_cost:,.0f} ✅")
            else:
                st.write(f"• Over Budget: ₦{total_cost - budget:,.0f} ❌")
        
        # Budget Status
        if total_cost <= budget:
            st.success(f"✅ System is within your budget! Remaining: ₦{budget - total_cost:,.0f}")
        else:
            st.error(f"❌ System exceeds budget by ₦{total_cost - budget:,.0f}")
            st.info("💡 Consider: Reducing bill offset, shorter backup duration, or smaller panels")
        
        # Invoice
        st.subheader("Professional Invoice")
        st.write(f"**Client:** {client_name}")
        st.write(f"**Location:** {location}")
        st.write(f"**Solar Data:** {sun_hours:.1f}h sun/day")
        st.write(f"**Temperature:** {temp_min}°C - {temp_max}°C (avg {temp_avg:.1f}°C)")
        st.write(f"**Weather:** {cloud_cover}% cloud, {dust_level} dust, {humidity}% humidity")
        st.write(f"**Environmental Factor:** {environmental_factor:.1%}")
        st.write(f"**Site Suitability:**")
        st.write(f"  - Roof: {roof_area_m2:.1f} m², Tilt: {roof_tilt}°, Azimuth: {azimuth}°")
        st.write(f"  - Shading: {shading_level}")
        if shading_sources:
            st.write(f"  - Obstacles: {', '.join(shading_sources)}")
        st.write(f"  - Site Factor: {site_factor:.1%}")
        st.write(f"**Combined Efficiency:** {combined_efficiency:.1%}")
        st.write(f"**System Type:** {'Off-Grid' if backup_days > 0 else 'Grid-Tied'}")
        st.write(f"**Coverage:** {bill_offset}% of {daily_kwh:.1f} kWh/day")
        st.write("---")
        st.write(f"**Solar Panels ({panel_watts}W):** {panels_needed} x ₦{panel_price:,.0f} = ₦{panel_cost:,.0f}")
        st.write(f"**{inverter_type}:** {system_kw:.2f} kW x ₦{inverter_info['cost_per_kw']:,.0f}/kW = ₦{inverter_cost:,.0f}")
        if backup_days > 0:
            st.write(f"**Batteries (5kWh):** {batteries_needed} x ₦200,000 = ₦{battery_cost:,.0f}")
            st.write(f"**Backup Duration:** {backup_duration}")
        else:
            st.write("**Batteries:** Not Required (Grid-Tied System)")
        st.write("---")
        st.write(f"**TOTAL SYSTEM COST: ₦{total_cost:,.0f}**")

# Information Section
with st.expander("🏗️ Physical Site Features"):
    st.write("**Rooftop Suitability Analysis:**")
    st.write("")
    st.write("**1. Usable Roof Area:**")
    st.write("• Determines maximum number of panels")
    st.write("• Typical 400W panel: ~2 m² (21.5 ft²)")
    st.write("• Account for walkways, vents, chimneys")
    st.write("")
    st.write("**2. Roof Tilt/Pitch:**")
    st.write("• Optimal tilt ≈ latitude angle")
    st.write(f"• Your location: {optimal_tilt:.0f}° optimal")
    st.write("• Flat roofs (0°): Use tilted mounting")
    st.write("• Steep roofs (>45°): Reduced efficiency")
    st.write("")
    st.write("**3. Roof Orientation (Azimuth):**")
    st.write("• South (180°): Best in Northern hemisphere")
    st.write("• SE/SW (135°/225°): 95-98% efficiency")
    st.write("• East/West (90°/270°): 80-85% efficiency")
    st.write("• North (0°): Not recommended (<70%)")
    st.write("")
    st.write("**4. Shading Analysis:**")
    st.write("• Critical factor - even 10% shade = 50% loss")
    st.write("• Sources: Trees, buildings, chimneys, vents")
    st.write("• Use satellite/drone imagery for accuracy")
    st.write("• Consider seasonal sun path changes")
    st.write("")
    st.write(f"**Current Site Efficiency: {site_factor:.1%}**")

with st.expander("🌡️ Temperature & Weather Impact"):
    st.write("**Why Temperature Matters:**")
    st.write("• Solar panels lose ~0.4-0.5% efficiency per °C above 25°C")
    st.write("• Hot climates require larger systems to compensate")
    st.write("• Temperature coefficient varies by panel type")
    st.write("")
    st.write("**Weather Pattern Effects:**")
    st.write("• Cloud Cover: Reduces direct sunlight, lowers output")
    st.write("• Dust/Pollution: Blocks light, requires frequent cleaning")
    st.write("• Humidity: Can reduce efficiency and cause corrosion")
    st.write("• Fog: Significantly reduces morning production")
    st.write("")
    st.write("**System Loss Factor:**")
    st.write("Combines all environmental losses into one derating factor:")
    st.write(f"• Current factor: {environmental_factor:.1%}")
    st.write(f"• System oversized by: {(1/environmental_factor - 1)*100:.1f}%")
    st.write("• Ensures real-world output meets your needs")

with st.expander("ℹ️ Calculation Methods Explained"):
    st.write("**A. Historical Consumption (Most Accurate)**")
    st.write("• Uses your actual electricity bills")
    st.write("• Best for existing homes with utility data")
    st.write("• Provides most accurate system sizing")
    st.write("")
    st.write("**B. Appliance-Based Load (Backup & New Builds)**")
    st.write("• Calculates load from individual appliances")
    st.write("• Best for new builds or backup systems")
    st.write("• Allows critical load selection")
    st.write("")
    st.write("**C. User Requirements**")
    st.write("• Bill Offset: How much of your bill to cover (50-100%)")
    st.write("• Backup Duration: How long system runs without grid/sun")
    st.write("• Grid-Tied: No battery, cheaper, requires grid connection")
    st.write("• Off-Grid: With battery, more expensive, full independence")