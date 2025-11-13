
import streamlit as st
import math
import requests
import re
import io
import csv
from datetime import datetime
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet

st.set_page_config(page_title="Solar Calculator", layout="wide")

st.markdown("""
# Solar Calculator

This simplified app calculates the kW requirement for a building, recommends panels
based on budget, attempts to fetch a market price for the recommended component,
and produces a downloadable invoice.
""")

def fetch_market_price(component_query: str, timeout: int = 6):
    """Best-effort market price lookup by searching the web."""
    try:
        headers = {"User-Agent": "Mozilla/5.0 (compatible; SolarCalc/1.0)"}
        url = f"https://duckduckgo.com/html/?q={requests.utils.quote(component_query + ' price')}"
        r = requests.get(url, headers=headers, timeout=timeout)
        text = r.text
        matches = re.findall(r"(₦|\$|£|€)?\s?([\d,]+(?:\.\d+)?)", text)
        if not matches:
            return None
        symbol, number = matches[0]
        clean = number.replace(',', '')
        value = float(clean)
        return {"price": value, "currency": symbol if symbol else ""}
    except Exception:
        return None

def calculate_requirement_from_daily_kwh(daily_kwh: float, sun_hours: float = 5.5, derating: float = 1.3):
    """Return required system power in kW."""
    if sun_hours <= 0:
        sun_hours = 5.5
    required_kw = (daily_kwh / sun_hours) * derating
    return required_kw

def recommend_panels(system_kw: float, panel_watts_options=(400, 350, 300)):
    """Return recommended panel options."""
    results = []
    total_watts_needed = math.ceil(system_kw * 1000)
    for w in panel_watts_options:
        count = math.ceil(total_watts_needed / w)
        results.append({"panel_watt": w, "count": count})
    return results

def create_pdf_invoice(client_name: str, location: str, items: list, total_cost: float, currency: str):
    """Create a PDF invoice."""
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()

    story = []

    logo_style = ParagraphStyle(name='logo', parent=styles['h1'], alignment=2, fontSize=16, spaceAfter=20)
    story.append(Paragraph("Your Company Logo", logo_style))
    story.append(Paragraph("Invoice", styles['h1']))
    story.append(Spacer(1, 12))
    story.append(Paragraph(f"<b>Date:</b> {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC", styles['Normal']))
    story.append(Paragraph(f"<b>Client:</b> {client_name}", styles['Normal']))
    story.append(Paragraph(f"<b>Location:</b> {location}", styles['Normal']))
    story.append(Spacer(1, 24))

    table_data = [["Item", "Quantity", "Unit Price", "Line Total"]]
    for item in items:
        table_data.append([
            item['name'],
            item['quantity'],
            f"{currency}{item['unit_price']:.2f}",
            f"{currency}{item['line_total']:.2f}"
        ])
    table_data.append(["", "", "<b>Total</b>", f"<b>{currency}{total_cost:.2f}</b>"])

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

st.sidebar.header("Inputs")
input_mode = st.sidebar.radio("Input mode:", ["Full house (kWh/day)", "Rooms/Stories estimate"])

if input_mode == "Full house (kWh/day)":
    daily_kwh = st.sidebar.number_input("Total household energy (kWh per day)", min_value=0.0, value=12.0, step=0.1)
else:
    rooms = st.sidebar.number_input("Number of rooms (or equivalent units)", min_value=1, value=4, step=1)
    avg_kwh_per_room = st.sidebar.number_input("Estimated daily kWh per room", min_value=0.1, value=2.5, step=0.1)
    daily_kwh = rooms * avg_kwh_per_room

sun_hours = st.sidebar.number_input("Peak sun hours per day (estimate)", min_value=1.0, max_value=12.0, value=5.5, step=0.1)
budget = st.sidebar.number_input("Budget (local currency units)", min_value=0.0, value=500000.0, step=1000.0)
client_name = st.sidebar.text_input("Client / Site Name", value="Client A")
location = st.sidebar.text_input("Location (city / address)", value="Unknown")

st.header("Quick Summary")
st.write(f"Estimated daily energy: **{daily_kwh:.2f} kWh/day**")

if st.button("Calculate System"):
    with st.spinner("Calculating system requirement and recommendations..."):
        system_kw = calculate_requirement_from_daily_kwh(daily_kwh, sun_hours)
        st.metric("Required System Size", f"{system_kw:.2f} kW")

        panel_options = recommend_panels(system_kw)
        st.subheader("Panel Recommendations")
        st.write("The count below estimates number of panels required for each typical panel watt rating.")

        lookup_name = "400W solar panel"
        price_info = fetch_market_price(lookup_name)
        currency_symbol = ""
        if price_info and price_info["currency"]:
            currency_symbol = price_info["currency"]

        items_for_invoice = []
        best_cost_option = None
        best_fit_option = None

        for opt in panel_options:
            w = opt['panel_watt']
            count = opt['count']

            if price_info:
                approx_unit_price = price_info['price']
            else:
                approx_unit_price = None

            c1, c2, c3 = st.columns([2, 1, 2])
            with c1:
                st.write(f"• {w} W panels: **{count}** panels")
            with c2:
                if approx_unit_price is not None:
                    st.write(f"Unit price: {currency_symbol}{approx_unit_price:,.2f}")
                else:
                    st.write("Unit price: (unknown)")
            with c3:
                if approx_unit_price is None:
                    user_price = st.number_input(f"Enter unit price for {w}W panel", min_value=0.0, value=100000.0, key=f"price_{w}")
                    unit_price = user_price
                else:
                    unit_price = approx_unit_price

                total_cost = unit_price * count
                st.write(f"Estimated cost for this option: {total_cost:,.2f}")

            item = {
                'name': f"{w}W panel",
                'quantity': count,
                'unit_price': unit_price,
                'line_total': total_cost
            }
            items_for_invoice.append(item)

            if total_cost <= budget:
                if best_cost_option is None or total_cost < best_cost_option['line_total']:
                    best_cost_option = item
                if best_fit_option is None or total_cost > best_fit_option['line_total']:
                    best_fit_option = item

        st.subheader("Recommended Options")
        if best_cost_option:
            st.success(f"**Best Cost Option:** {best_cost_option['name']} ({best_cost_option['quantity']} panels) - Cost: {currency_symbol}{best_cost_option['line_total']:,.2f}")
        else:
            st.warning("No affordable options found for 'Best Cost'.")

        if best_fit_option:
            st.info(f"**Best Fit for Budget:** {best_fit_option['name']} ({best_fit_option['quantity']} panels) - Cost: {currency_symbol}{best_fit_option['line_total']:,.2f}")
        else:
            st.warning("No affordable options found for 'Best Fit for Budget'.")

        total_estimate = sum(it['line_total'] for it in items_for_invoice)

        if st.button("Generate PDF Invoice"):
            invoice_bytes = create_pdf_invoice(client_name, location, items_for_invoice, total_estimate, currency_symbol)
            st.download_button("Download Invoice (PDF)", data=invoice_bytes, file_name=f"invoice_{client_name.replace(' ','_')}.pdf")
