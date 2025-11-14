"""
Solar Calculator and Watt Recommender System - Fast & Simple
"""

import streamlit as st
import math
import io
from datetime import datetime
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet

# Appliance database
APPLIANCE_DATABASE = {
    "Refrigerator": 150,
    "Air Conditioner (1.5HP)": 1200,
    "Air Conditioner (1HP)": 800,
    "Ceiling Fan": 75,
    "LED Bulb (10W)": 10,
    "LED Bulb (20W)": 20,
    "Television (LED 32\")": 60,
    "Television (LED 55\")": 120,
    "Laptop": 65,
    "Desktop Computer": 200,
    "Washing Machine": 500,
    "Microwave": 1000,
    "Electric Kettle": 1500,
    "Iron": 1200,
    "Water Pump": 750,
    "Freezer": 200,
    "Custom Appliance": 0
}

# Panel options
PANEL_OPTIONS = {
    "400W Monocrystalline": {"watts": 400, "default_price": 100000},
    "350W Monocrystalline": {"watts": 350, "default_price": 90000},
    "300W Monocrystalline": {"watts": 300, "default_price": 80000},
    "450W Bifacial": {"watts": 450, "default_price": 120000},
    "320W Polycrystalline": {"watts": 320, "default_price": 75000}
}

st.set_page_config(page_title="Solar Calculator and Watt Recommender System", layout="wide")

# Initialize session state
if 'appliances' not in st.session_state:
    st.session_state.appliances = []

def create_pdf_invoice(client_name, location, items, total_cost):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    
    story = []
    story.append(Paragraph("Solar System Invoice", styles['h1']))
    story.append(Spacer(1, 12))
    story.append(Paragraph(f"<b>Date:</b> {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC", styles['Normal']))
    story.append(Paragraph(f"<b>Client:</b> {client_name}", styles['Normal']))
    story.append(Paragraph(f"<b>Location:</b> {location}", styles['Normal']))
    story.append(Spacer(1, 24))

    table_data = [["Item", "Quantity", "Unit Price", "Line Total"]]
    
    for item in items:
        table_data.append([
            item['name'],
            str(item['quantity']),
            f"₦{item['unit_price']:,.2f}",
            f"₦{item['line_total']:,.2f}"
        ])
        
    table_data.append(["", "", "<b>Total</b>", f"<b>₦{total_cost:,.2f}</b>"])

    invoice_table = Table(table_data, colWidths=[280, 80, 100, 100])
    invoice_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -2), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('FONTNAME', (0, -1), (-1, -1), 'Helvetica-Bold'),
        ('BACKGROUND', (0, -1), (-1, -1), colors.lightgrey),
    ]))
    
    story.append(invoice_table)
    story.append(Spacer(1, 24))
    story.append(Paragraph("Thank you for your business!", styles['Normal']))

    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()

st.markdown("# Solar Calculator and Watt Recommender System")

# Sidebar Configuration
st.sidebar.header("Configuration")

# Appliance Selection
st.sidebar.subheader("Add Appliances")
selected_appliance = st.sidebar.selectbox("Select Appliance", list(APPLIANCE_DATABASE.keys()))

# Auto-fill watts
default_watts = APPLIANCE_DATABASE[selected_appliance]

if selected_appliance == "Custom Appliance":
    appliance_name = st.sidebar.text_input("Custom Appliance Name", placeholder="e.g., Gaming Console")
    appliance_watts = st.sidebar.number_input("Power (Watts)", min_value=1, value=100, step=10)
else:
    appliance_name = selected_appliance
    appliance_watts = st.sidebar.number_input("Power (Watts)", min_value=1, value=default_watts, step=10)

appliance_hours = st.sidebar.number_input("Hours per day", min_value=0.1, value=8.0, step=0.5)
appliance_qty = st.sidebar.number_input("Quantity", min_value=1, value=1, step=1)

col1, col2 = st.sidebar.columns(2)
with col1:
    if st.button("Add"):
        if appliance_name and appliance_name != "Custom Appliance":
            st.session_state.appliances.append({
                'name': appliance_name,
                'watts': appliance_watts,
                'hours': appliance_hours,
                'quantity': appliance_qty
            })
            st.success(f"Added {appliance_name}")
            st.rerun()

with col2:
    if st.button("Clear All"):
        st.session_state.appliances = []
        st.success("Cleared all")
        st.rerun()

# System Configuration
st.sidebar.subheader("System Configuration")
sun_hours = st.sidebar.number_input("Peak sun hours per day", min_value=1.0, max_value=12.0, value=5.5, step=0.1)
budget = st.sidebar.number_input("Budget (₦)", min_value=0.0, value=500000.0, step=1000.0)
battery_type = st.sidebar.selectbox("Battery Type", ["Lithium-Ion", "Lead-Acid (AGM)", "Lead-Acid (Gel)", "LiFePO4"])
autonomy_days = st.sidebar.slider("Backup Days", 1, 7, 2)
client_name = st.sidebar.text_input("Client Name", value="Client A")

# Panel Selection
st.sidebar.subheader("Panel Selection")
selected_panel = st.sidebar.selectbox("Choose Panel Type", list(PANEL_OPTIONS.keys()))
default_price = PANEL_OPTIONS[selected_panel]["default_price"]
panel_price = st.sidebar.number_input(f"{selected_panel} Price (₦)", min_value=0.0, value=float(default_price), step=1000.0)

# Main Content
st.subheader("Your Selected Appliances")

if st.session_state.appliances:
    # Calculate totals
    total_watts = 0
    daily_kwh = 0
    
    st.write("**Appliance List:**")
    for i, app in enumerate(st.session_state.appliances):
        appliance_total_watts = app['watts'] * app['quantity']
        appliance_daily_kwh = (app['watts'] * app['hours'] * app['quantity']) / 1000
        
        total_watts += appliance_total_watts
        daily_kwh += appliance_daily_kwh
        
        col1, col2, col3, col4, col5, col6 = st.columns([3, 1, 1, 1, 1, 1])
        with col1:
            st.write(f"{i+1}. **{app['name']}**")
        with col2:
            st.write(f"{app['watts']}W")
        with col3:
            st.write(f"{app['hours']}h")
        with col4:
            st.write(f"Qty: {app['quantity']}")
        with col5:
            st.write(f"{appliance_total_watts}W total")
        with col6:
            if st.button("Remove", key=f"remove_{i}"):
                st.session_state.appliances.pop(i)
                st.rerun()
    
    # Summary
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total System Watts", f"{total_watts:,} W")
    with col2:
        st.metric("Daily Energy", f"{daily_kwh:.2f} kWh")
    with col3:
        st.metric("Number of Appliances", len(st.session_state.appliances))
        
else:
    st.info("No appliances added yet. Use the sidebar to add appliances.")
    total_watts = 0
    daily_kwh = 0

# Calculate System
if st.button("Calculate Solar System", type="primary", use_container_width=True):
    if not st.session_state.appliances:
        st.error("Please add at least one appliance first.")
    else:
        # System calculations
        system_kw = (daily_kwh / sun_hours) * 1.3  # 30% safety margin
        
        # Battery calculations
        battery_capacity = daily_kwh * autonomy_days * 1.2
        depth_of_discharge = 0.8 if "Lithium" in battery_type else 0.5
        battery_capacity_kwh = battery_capacity / depth_of_discharge
        
        # Panel calculations
        panel_info = PANEL_OPTIONS[selected_panel]
        panel_watts = panel_info["watts"]
        total_watts_needed = math.ceil(system_kw * 1000)
        panel_count = math.ceil(total_watts_needed / panel_watts)
        panel_cost = panel_count * panel_price
        
        # Battery calculations
        battery_count = math.ceil(battery_capacity_kwh / 5)  # 5kWh batteries
        battery_cost = battery_count * 200000
        total_system_cost = panel_cost + battery_cost
        
        st.header("System Recommendations")
        
        # System overview
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("System Size", f"{system_kw:.2f} kW")
        with col2:
            st.metric("Panels Needed", f"{panel_count} panels")
        with col3:
            st.metric("Battery Capacity", f"{battery_capacity_kwh:.1f} kWh")
        with col4:
            st.metric("Total Cost", f"₦{total_system_cost:,.2f}")
        
        # Budget analysis
        if total_system_cost <= budget:
            st.success(f"✅ System is within your budget of ₦{budget:,.2f}")
        else:
            over_budget = total_system_cost - budget
            st.error(f"❌ System exceeds budget by ₦{over_budget:,.2f}")
        
        # Invoice Generation
        st.subheader("Generate Invoice")
        
        # Invoice preview
        st.write("**Invoice Preview:**")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.write("**Item**")
            st.write(selected_panel)
            st.write(f"{battery_type} Battery (5kWh)")
            st.write("**TOTAL**")
        with col2:
            st.write("**Quantity**")
            st.write(panel_count)
            st.write(battery_count)
            st.write("")
        with col3:
            st.write("**Unit Price**")
            st.write(f"₦{panel_price:,.2f}")
            st.write("₦200,000")
            st.write("")
        with col4:
            st.write("**Total**")
            st.write(f"₦{panel_cost:,.2f}")
            st.write(f"₦{battery_cost:,.2f}")
            st.write(f"**₦{total_system_cost:,.2f}**")
        
        if st.button("Generate PDF Invoice", type="primary"):
            try:
                invoice_items = [
                    {
                        'name': selected_panel,
                        'quantity': panel_count,
                        'unit_price': panel_price,
                        'line_total': panel_cost
                    },
                    {
                        'name': f'{battery_type} Battery (5kWh)',
                        'quantity': battery_count,
                        'unit_price': 200000,
                        'line_total': battery_cost
                    }
                ]
                
                invoice_bytes = create_pdf_invoice(client_name, "Nigeria", invoice_items, total_system_cost)
                
                st.download_button(
                    label="Download Invoice PDF",
                    data=invoice_bytes,
                    file_name=f"solar_invoice_{client_name.replace(' ', '_')}.pdf",
                    mime="application/pdf"
                )
                
                st.success("Invoice generated! Click download button above.")
                
            except Exception as e:
                st.error(f"Error generating invoice: {str(e)}")

st.markdown("---")
st.markdown("**Solar Calculator and Watt Recommender System** - Fast and accurate solar system design")