"""
Solar Calculator - Taipy GUI
"""

from taipy.gui import Gui, notify
import math
import io
import os
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

# Nigerian locations with sun hours
NIGERIAN_LOCATIONS = {
    "Lagos": {"sun_hours": 5.2, "region": "South West"},
    "Abuja": {"sun_hours": 5.8, "region": "North Central"},
    "Kano": {"sun_hours": 6.2, "region": "North West"},
    "Port Harcourt": {"sun_hours": 4.8, "region": "South South"},
    "Ibadan": {"sun_hours": 5.1, "region": "South West"},
    "Kaduna": {"sun_hours": 5.9, "region": "North West"},
    "Enugu": {"sun_hours": 5.0, "region": "South East"},
    "Jos": {"sun_hours": 5.7, "region": "North Central"},
    "Maiduguri": {"sun_hours": 6.5, "region": "North East"},
    "Calabar": {"sun_hours": 4.9, "region": "South South"},
    "Custom Location": {"sun_hours": 5.5, "region": "Custom"}
}

# State variables
selected_appliance = "Refrigerator"
appliance_watts = 150
appliance_hours = 8.0
appliance_qty = 1
appliances = []
daily_kwh = 12.0
selected_location = "Lagos"
sun_hours = 5.2
budget = 500000.0
battery_type = "Lithium-Ion"
autonomy_days = 2
client_name = "Client A"
location = "Lagos"
system_kw = 0.0
battery_capacity_kwh = 0.0
results_visible = False
panel_400_price = 100000.0
panel_350_price = 90000.0
panel_300_price = 80000.0
recommendations = ""
invoice_items = []
total_cost = 0.0
show_invoice = False
best_panel_recommendation = ""

def calculate_requirement_from_daily_kwh(daily_kwh, sun_hours=5.5, derating=1.3):
    if sun_hours <= 0:
        sun_hours = 5.5
    return (daily_kwh / sun_hours) * derating

def on_appliance_change(state):
    state.appliance_watts = APPLIANCE_DATABASE[state.selected_appliance]

def on_location_change(state):
    if state.selected_location in NIGERIAN_LOCATIONS:
        state.sun_hours = NIGERIAN_LOCATIONS[state.selected_location]["sun_hours"]
        state.location = state.selected_location
        notify(state, "info", f"Updated sun hours to {state.sun_hours} for {state.selected_location}")

def add_appliance(state):
    if state.selected_appliance != "Custom Appliance" or state.appliance_watts > 0:
        state.appliances.append({
            'name': state.selected_appliance,
            'watts': state.appliance_watts,
            'hours': state.appliance_hours,
            'quantity': state.appliance_qty
        })
        state.daily_kwh = sum(app['watts'] * app['hours'] * app['quantity'] / 1000 for app in state.appliances)
        notify(state, "success", f"Added {state.selected_appliance}")

def clear_appliances(state):
    state.appliances = []
    state.daily_kwh = 12.0
    notify(state, "info", "Cleared all appliances")

def remove_appliance(state, var_name, payload):
    index = payload['index']
    if 0 <= index < len(state.appliances):
        removed = state.appliances.pop(index)
        state.daily_kwh = sum(app['watts'] * app['hours'] * app['quantity'] / 1000 for app in state.appliances) if state.appliances else 12.0
        notify(state, "info", f"Removed {removed['name']}")

def create_pdf_invoice(client_name, location, items, total_cost, currency="N"):
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
            f"{currency}{item['unit_price']:,.2f}",
            f"{currency}{item['line_total']:,.2f}"
        ])
        
    table_data.append(["", "", "<b>Total</b>", f"<b>{currency}{total_cost:,.2f}</b>"])

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

def calculate_system(state):
    state.system_kw = calculate_requirement_from_daily_kwh(state.daily_kwh, state.sun_hours)
    
    # Battery calculations
    battery_capacity = state.daily_kwh * state.autonomy_days * 1.2
    depth_of_discharge = 0.8 if "Lithium" in state.battery_type or "LiFePO4" in state.battery_type else 0.5
    state.battery_capacity_kwh = battery_capacity / depth_of_discharge
    
    # Panel calculations
    total_watts = math.ceil(state.system_kw * 1000)
    panel_400_count = math.ceil(total_watts / 400)
    panel_350_count = math.ceil(total_watts / 350)
    panel_300_count = math.ceil(total_watts / 300)
    
    cost_400 = panel_400_count * state.panel_400_price
    cost_350 = panel_350_count * state.panel_350_price
    cost_300 = panel_300_count * state.panel_300_price
    
    # Determine best panel recommendation
    panel_options = [
        {'type': '400W', 'count': panel_400_count, 'cost': cost_400, 'efficiency': 0.95},
        {'type': '350W', 'count': panel_350_count, 'cost': cost_350, 'efficiency': 0.90},
        {'type': '300W', 'count': panel_300_count, 'cost': cost_300, 'efficiency': 0.85}
    ]
    
    # Find best option within budget
    affordable_options = [opt for opt in panel_options if opt['cost'] <= state.budget]
    
    if affordable_options:
        best_option = max(affordable_options, key=lambda x: x['efficiency'])
        state.best_panel_recommendation = f"RECOMMENDED: {best_option['type']} panels ({best_option['count']} units) - Most efficient within budget"
    else:
        best_option = min(panel_options, key=lambda x: x['cost'])
        state.best_panel_recommendation = f"BUDGET OPTION: {best_option['type']} panels ({best_option['count']} units) - Cheapest available"
    
    # Prepare invoice items
    state.invoice_items = [
        {'name': '400W Solar Panel', 'quantity': panel_400_count, 'unit_price': state.panel_400_price, 'line_total': cost_400},
        {'name': '350W Solar Panel', 'quantity': panel_350_count, 'unit_price': state.panel_350_price, 'line_total': cost_350},
        {'name': '300W Solar Panel', 'quantity': panel_300_count, 'unit_price': state.panel_300_price, 'line_total': cost_300},
        {'name': f'{state.battery_type} Battery (5kWh)', 'quantity': math.ceil(state.battery_capacity_kwh/5), 'unit_price': 200000, 'line_total': math.ceil(state.battery_capacity_kwh/5)*200000}
    ]
    
    state.total_cost = best_option['cost'] + state.invoice_items[3]['line_total']
    
    # Location info
    location_info = NIGERIAN_LOCATIONS.get(state.selected_location, {"region": "Unknown", "sun_hours": state.sun_hours})
    
    state.recommendations = f"""
### Location Analysis
- **Location:** {state.selected_location} ({location_info['region']} Region)
- **Peak Sun Hours:** {state.sun_hours} hours/day
- **Solar Potential:** {'Excellent' if state.sun_hours > 6 else 'Good' if state.sun_hours > 5 else 'Fair'}

### System Requirements
- **System Size:** {state.system_kw:.2f} kW
- **Battery Capacity:** {state.battery_capacity_kwh:.2f} kWh ({math.ceil(state.battery_capacity_kwh/5)} x 5kWh units)
- **Battery Type:** {state.battery_type}

### Panel Options Analysis
1. **400W Panels:** {panel_400_count} panels - N{cost_400:,.2f} {'(Affordable)' if cost_400 <= state.budget else '(Over budget)'}
2. **350W Panels:** {panel_350_count} panels - N{cost_350:,.2f} {'(Affordable)' if cost_350 <= state.budget else '(Over budget)'}
3. **300W Panels:** {panel_300_count} panels - N{cost_300:,.2f} {'(Affordable)' if cost_300 <= state.budget else '(Over budget)'}

### AI Recommendation
{state.best_panel_recommendation}

### Budget Summary
- **Your Budget:** N{state.budget:,.2f}
- **Recommended System Cost:** N{state.total_cost:,.2f}
- **Budget Status:** {'Within Budget' if state.total_cost <= state.budget else 'Over Budget by N' + f'{state.total_cost - state.budget:,.2f}'}
"""
    
    state.results_visible = True
    notify(state, "success", "System calculated successfully!")

def generate_invoice(state):
    if state.invoice_items:
        # Select best affordable option
        best_item = None
        for item in state.invoice_items[:3]:  # Only panel options
            if item['line_total'] <= state.budget:
                best_item = item
                break
        
        if not best_item:
            best_item = min(state.invoice_items[:3], key=lambda x: x['line_total'])
        
        selected_items = [best_item, state.invoice_items[3]]  # Panel + Battery
        total_invoice_cost = best_item['line_total'] + state.invoice_items[3]['line_total']
        
        try:
            invoice_bytes = create_pdf_invoice(state.client_name, state.location, selected_items, total_invoice_cost)
            
            filename = f"invoice_{state.client_name.replace(' ', '_').replace('/', '_')}.pdf"
            filepath = os.path.join(os.getcwd(), filename)
            
            with open(filepath, 'wb') as f:
                f.write(invoice_bytes)
            
            state.show_invoice = True
            notify(state, "success", f"Invoice saved to: {filepath}")
        except Exception as e:
            notify(state, "error", f"Error generating invoice: {str(e)}")
    else:
        notify(state, "warning", "Please calculate system first")

# Taipy page layout
page = """
<|toggle|theme|>

# Solar Calculator - Interactive System Designer

<|layout|columns=1 2|
<|part|class_name=sidebar|
## Configuration Panel

### Add Appliances
<|{selected_appliance}|selector|lov={list(APPLIANCE_DATABASE.keys())}|on_change=on_appliance_change|label=Select Appliance|class_name=fullwidth|>

<|layout|columns=2 1 1|
<|{appliance_watts}|number|label=Power (W)|class_name=fullwidth|>
<|{appliance_hours}|number|label=Hours/day|min=0.1|step=0.5|class_name=fullwidth|>
<|{appliance_qty}|number|label=Qty|min=1|class_name=fullwidth|>
|>

<|layout|columns=1 1|
<|Add Appliance|button|on_action=add_appliance|class_name=success fullwidth|>
<|Clear All|button|on_action=clear_appliances|class_name=secondary fullwidth|>
|>

---

### Location & System Configuration

<|{selected_location}|selector|lov={list(NIGERIAN_LOCATIONS.keys())}|on_change=on_location_change|label=Select Location|class_name=fullwidth|>

<|{sun_hours}|number|label=Peak sun hours/day|min=1.0|max=12.0|step=0.1|class_name=fullwidth|>

<|{budget}|number|label=Budget (N)|step=1000|format=%,.0f|class_name=fullwidth|>

<|{battery_type}|selector|lov={["Lithium-Ion", "Lead-Acid (AGM)", "Lead-Acid (Gel)", "LiFePO4"]}|label=Battery Type|class_name=fullwidth|>

<|{autonomy_days}|slider|label=Backup Days: {autonomy_days}|min=1|max=7|class_name=fullwidth|>

<|{client_name}|input|label=Client Name|class_name=fullwidth|>

---

### Panel Prices (N)

<|{panel_400_price}|number|label=400W Panel|step=1000|format=%,.0f|class_name=fullwidth|>

<|{panel_350_price}|number|label=350W Panel|step=1000|format=%,.0f|class_name=fullwidth|>

<|{panel_300_price}|number|label=300W Panel|step=1000|format=%,.0f|class_name=fullwidth|>

<|layout|columns=1 1|
<|Calculate System|button|on_action=calculate_system|class_name=primary fullwidth|>
<|Generate Invoice|button|on_action=generate_invoice|class_name=success fullwidth|active={results_visible}|>
|>

|>

<|part|
## System Overview

<|layout|columns=2 2 1|
<|card|
### Energy
**{daily_kwh:.2f}** kWh/day
|>
<|card|
### Appliances
**{len(appliances)}** items
|>
<|card|
### Location
**{selected_location}**
|>
|>

---

### Your Appliances
<|{appliances}|table|columns=name,watts,hours,quantity|show_all|on_action=remove_appliance|>

<|{len(appliances) == 0}|
*No appliances added yet. Use the panel on the left to add appliances.*
|>

---

<|{results_visible}|
## System Recommendations

{recommendations}

### Invoice Items
<|{invoice_items}|table|columns=name,quantity,unit_price,line_total|show_all|>

**Recommended System Cost: N{total_cost:,.2f}**
|>

<|{show_invoice}|
### Invoice Generated Successfully!
Invoice PDF has been saved to your current directory.
|>

|>
|>
"""

if __name__ == "__main__":
    gui = Gui(page)
    gui.run(title="Solar Calculator - Interactive System Designer", port="auto", dark_mode=False, debug=True)