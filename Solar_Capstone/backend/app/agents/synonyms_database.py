# This file contains an AI agent that helps with solar system recommendations
SMART_SYNONYMS = {
    # ---------------------------
    # AGRICULTURAL EQUIPMENT
    # ---------------------------
    "Greenhouse Systems": [
        "greenhouse", "farming greenhouse", "nursery house"
    ],
    "Incubators": [
        "incubator", "egg incubator", "hatchery machine"
    ],
    "Irrigation Systems": [
        "irrigation", "drip irrigation", "sprinkler system"
    ],
    "Livestock Equipment": [
        "livestock equipment", "animal farming equipment"
    ],
    "Sprayers": [
        "sprayer", "farm sprayer", "knapsack sprayer"
    ],

    # ---------------------------
    # COMPUTERS & ACCESSORIES
    # ---------------------------
    "Desktop PC": [
        "desktop", "desktop computer", "system unit", "office computer"
    ],
    "Laptop": [
        "laptop", "notebook", "hp laptop", "dell laptop", "computer"
    ],
    "Monitors": [
        "monitor", "screen", "lcd", "led screen"
    ],
    "Networking Devices": [
        "router", "mifi", "modem", "wifi device", "hotspot"
    ],
    "Printers": [
        "printer", "printing machine", "hp printer", "epson printer"
    ],
    "Printers (Solar)": [
        "solar printer", "dc printer"
    ],
    "Projectors": [
        "projector", "video projector", "presentation projector"
    ],
    "Procjetors (Solar)": [
        "solar projector", "dc projector"
    ],
    "Scanners": [
        "scanner", "document scanner"
    ],
    "Servers/Workstations": [
        "server", "workstation", "server computer"
    ],

    # ---------------------------
    # COOLING DEVICES
    # ---------------------------
    "Air Conditioner - 1 HP (Small Room)": [
        "1hp ac", "small ac", "split ac small"
    ],
    "Air Conditioner - 1.5 HP (Medium Room)": [
        "1.5hp ac", "medium ac", "split unit"
    ],
    "Air Conditioner - 2 HP (Large Room)": [
        "2hp ac", "big ac", "large split ac"
    ],
    "Air Conditioner - Standing AC (2–3 HP)": [
        "standing ac", "floor ac", "cabinet ac"
    ],
    "Air Conditioner - Window AC": [
        "window ac", "box ac"
    ],
    "Air Conditioner - Portable AC": [
        "portable ac", "mobile ac"
    ],
    "Ceiling Fan": [
        "ceiling fan", "orifice fan"
    ],
    "Fan - Standing Fan": [
        "standing fan", "pedestal fan"
    ],
    "Fan - Table Fan": [
        "table fan", "small fan"
    ],
    "Fan - Rechargeable Fan": [
        "rechargeable fan", "dc fan"
    ],
    "Fan - Industrial Fan": [
        "industrial fan", "big fan"
    ],

    # ---------------------------
    # ENTERTAINMENT & AV
    # ---------------------------
    "Gaming Consoles": [
        "game console", "playstation", "xbox"
    ],
    "Home Theater Systems": [
        "home theater", "surround system", "sound bar"
    ],
    "Media Players": [
        "dvd player", "blu-ray player", "media player"
    ],
    "Musical Instruments": [
        "keyboard", "guitar", "drum set"
    ],
    "Other Electronics": [
        "electronics", "misc electronics"
    ],
    "Photography Equipment": [
        "camera", "dslr", "tripod"
    ],
    "Radios": [
        "radio", "fm radio", "am radio"
    ],
    "Recreational Lighting": [
        "party light", "disco light", "stage light"
    ],
    "Set-Top Box/Decoder": [
        "decoder", "dstv", "gotv", "startimes box"
    ],
    "Sound Systems": [
        "sound system", "speaker system", "amplifier"
    ],
    "Speakers": [
        "speaker", "bluetooth speaker", "big speaker"
    ],
    "Television - Small (24–32 inch)": [
        "small tv", "24 inch tv", "32 inch tv"
    ],
    "Television - Medium (40–50 inch)": [
        "medium tv", "40 inch tv", "50 inch tv"
    ],
    "Television - Large (55–75 inch)": [
        "big tv", "55 inch tv", "75 inch tv"
    ],
    "Television (Solar)": [
        "solar tv", "dc tv"
    ],

    # ---------------------------
    # HEALTH & MEDICAL DEVICES
    # ---------------------------
    "CPAP/BIPAP Machines": [
        "cpap", "bipap", "sleep apnea machine"
    ],
    "Dental Equipment - Portable Dental Chair": [
        "dental chair", "portable dentist chair"
    ],
    "Dental Equipment - Ultrasonic Scaler": [
        "ultrasonic scaler", "dental scaler"
    ],
    "Diagnostic Equipment - Ultrasound Machine (Portable)": [
        "ultrasound machine", "scan machine"
    ],
    "Diagnostic Equipment - X-ray Machine (Portable)": [
        "xray machine", "portable xray"
    ],
    "Medical Analyzers - Hematology Analyzer": [
        "hematology analyzer", "blood analyzer"
    ],
    "Medical Analyzers - Biochemistry Analyzer": [
        "biochemistry analyzer", "lab analyzer"
    ],
    "Microscope (Medical Use)": [
        "medical microscope", "lab microscope"
    ],
    "Patient Monitoring - ECG Machine": [
        "ecg machine", "heart monitor"
    ],
    "Patient Monitoring - Multiparameter Monitor": [
        "patient monitor", "hospital monitor"
    ],
    "Refrigeration - Vaccine Refrigerator (Medical)": [
        "vaccine fridge", "medical fridge"
    ],
    "Sterilization Equipment - Autoclave (Small)": [
        "autoclave", "sterilizer", "steam sterilizer"
    ],
    "Suction Machines": [
        "suction machine", "medical suction"
    ],
    "Theater Equipment - Operating Lamp": [
        "operating lamp", "surgical light"
    ],
    "Theater Equipment - Small Anesthesia Machine": [
        "anesthesia machine", "gas machine"
    ],

    # ---------------------------
    # LARGE COOKING APPLIANCES
    # ---------------------------
    "Cooker - Small Electric Cooker (Tabletop)": [
        "small electric cooker", "table top cooker", "mini cooker"
    ],
    "Cooker - Gas Cooker (2–4 Burners)": [
        "gas cooker", "4 burner gas cooker", "cooker with oven"
    ],
    "Cooker - Gas + Electric Cooker (Hybrid)": [
        "hybrid cooker", "gas electric cooker"
    ],
    "Cooker - Standing Gas Cooker (Oven + Grill)": [
        "standing gas cooker", "oven cooker", "full gas cooker"
    ],
    "Cooker - Hot Plate (Single/Double)": [
        "hot plate", "electric hot plate", "table hot plate"
    ],
    "Cooker (Solar) - DC Solar Cooker": [
        "solar cooker", "dc cooker"
    ],
    "Microwave Oven - Small (20–25L)": [
        "small microwave", "20l microwave"
    ],
    "Microwave Oven - Medium (30–35L)": [
        "medium microwave", "30l microwave"
    ],
    "Microwave Oven - Large (40L+)": [
        "large microwave", "40l microwave"
    ],
    "Microwave Oven (Solar) - DC Solar Microwave": [
        "solar microwave", "dc microwave"
    ],
    "Oven - Small Electric Oven": [
        "small oven", "table oven", "electric oven"
    ],
    "Oven - Medium Electric Oven": [
        "medium oven", "oven for baking"
    ],
    "Oven - Large Electric Oven": [
        "large oven", "industrial oven", "bakery oven"
    ],

    # ---------------------------
    # LAUNDRY & CLEANING
    # ---------------------------
    "Dryer - Small Clothes Dryer": [
        "small dryer", "mini dryer"
    ],
    "Dryer - Large Clothes Dryer": [
        "large dryer", "industrial dryer"
    ],
    "Iron - Basic Electric Pressing Iron": [
        "pressing iron", "iron", "binatone iron"
    ],
    "Iron - Steam Iron": [
        "steam iron", "philips steam iron"
    ],
    "Iron - Industrial Iron": [
        "industrial iron", "laundry iron"
    ],
    "Washing Machine - Small (5–6kg)": [
        "small washing machine", "5kg washer"
    ],
    "Washing Machine - Medium (7–8kg)": [
        "medium washing machine", "7kg washer"
    ],
    "Washing Machine - Large (9–12kg)": [
        "large washing machine", "9kg washer", "10kg washer"
    ],
    "Washing Machine - Industrial (15kg+)": [
        "industrial washing machine", "laundry machine"
    ],

    # ---------------------------
    # LIGHTING
    # ---------------------------
    "Lighting - CFL Bulb (Energy Saver)": [
        "cfl bulb", "energy saver", "small bulb"
    ],
    "Lighting - LED Bulb (5–12W)": [
        "led bulb", "low energy bulb"
    ],
    "Lighting - LED Tube Light (2ft/4ft)": [
        "led tube", "tube light", "strip tube"
    ],
    "Lighting - Floodlight (30–100W)": [
        "flood light", "outdoor light", "security light"
    ],
    "Lighting - Solar Lantern": [
        "solar lantern", "rechargeable lantern"
    ],
    "Lighting - Street Light (Solar)": [
        "solar street light", "dc street light"
    ],

    # ---------------------------
    # MOBILE & PERSONAL DEVICES
    # ---------------------------
    "Mobile Phones - Basic Phone": [
        "small phone", "button phone", "torchlight phone"
    ],
    "Mobile Phones - Smartphone (Android/iOS)": [
        "smartphone", "android phone", "iphone"
    ],
    "Mobile Phones - Tablet (iPad/Android)": [
        "tablet", "ipad", "android tab"
    ],
    "Power Banks - Small (5,000–10,000mAh)": [
        "small power bank", "5000mah power bank"
    ],
    "Power Banks - Medium (10,000–20,000mAh)": [
        "medium power bank", "20000mah power bank"
    ],
    "Power Banks - Large (30,000mAh+)": [
        "large power bank", "big power bank", "30000mah power bank"
    ],

    # ---------------------------
    # OFFICE EQUIPMENT
    # ---------------------------
    "Cash Register/POS - POS Terminal": [
        "pos", "pos machine", "pos terminal"
    ],
    "Cash Register/POS - Cash Register": [
        "cash register", "sales register"
    ],
    "Office Shredders - Small Paper Shredder": [
        "paper shredder", "mini shredder"
    ],
    "Office Shredders - Large Paper Shredder": [
        "industrial shredder", "big shredder"
    ],

    # ---------------------------
    # PERSONAL CARE
    # ---------------------------
    "Clippers - Small Hair Clipper": [
        "hair clipper", "rechargeable clipper"
    ],
    "Clippers - Professional Hair Clipper": [
        "barber clipper", "salon clipper"
    ],
    "Hair Dryers - Small Hair Dryer": [
        "hair dryer", "mini hair dryer"
    ],
    "Hair Dryers - Professional Hair Dryer": [
        "salon dryer", "barber hair dryer"
    ],
    "Shaving Machines - Electric Shaver": [
        "shaver", "electric shaver"
    ],

    # ---------------------------
    # REFRIGERATION & FREEZING
    # ---------------------------
    "Refrigerator - Small Single Door (50–100L)": [
        "small fridge", "table fridge"
    ],
    "Refrigerator - Medium Double Door (150–250L)": [
        "medium fridge", "double door fridge"
    ],
    "Refrigerator - Large Fridge (300L+)": [
        "large fridge", "big fridge", "standing fridge"
    ],
    "Refrigerator - Chest Freezer (200–500L)": [
        "chest freezer", "deep freezer", "lg freezer"
    ],
    "Refrigerator - Industrial Cold Room": [
        "cold room", "industrial freezer"
    ],
    "Refrigerator (Solar) - DC Solar Fridge (100–200L)": [
        "solar fridge", "dc fridge", "12v fridge"
    ],
    "Refrigerator (Solar) - DC Solar Freezer (200L+)": [
        "solar freezer", "dc freezer"
    ],

    # ---------------------------
    # SECURITY & SAFETY
    # ---------------------------
    "CCTV - 4 Camera DVR Kit": [
        "cctv kit", "dvr kit", "security camera"
    ],
    "CCTV - IP Camera (Wi-Fi)": [
        "ip camera", "wifi camera"
    ],
    "CCTV - Dome/Bullet Camera (Standalone)": [
        "dome camera", "bullet camera"
    ],
    "Security Alarm Systems - Intruder Alarm": [
        "alarm system", "burglar alarm"
    ],
    "Security Alarm Systems - Fire Alarm": [
        "fire alarm", "smoke alarm"
    ],
    "Video Door Phones - Smart Doorbell": [
        "video doorbell", "smart doorbell", "ring bell"
    ],

    # ---------------------------
    # SMALL KITCHEN APPLIANCES
    # ---------------------------
    "Blender - Small Blender (1–1.5L)": [
        "small blender", "mini blender"
    ],
    "Blender - Big Blender (2L+)": [
        "big blender", "large blender"
    ],
    "Blender - Rechargeable Blender": [
        "rechargeable blender", "portable blender"
    ],
    "Coffee Maker - Small Coffee Machine": [
        "coffee maker", "mini coffee maker"
    ],
    "Coffee Maker - Large Coffee Machine": [
        "big coffee maker", "espresso machine"
    ],
    "Electric Kettle - Small (1L)": [
        "small kettle", "mini kettle", "1 litre kettle"
    ],
    "Electric Kettle - Large (1.7–2L)": [
        "large kettle", "2 litre kettle"
    ],
    "Food Processor - Small Food Processor": [
        "small food processor", "mini food processor"
    ],
    "Food Processor - Large Food Processor": [
        "big food processor", "industrial food processor"
    ],
    "Toaster - 2 Slice Toaster": [
        "2 slice toaster", "small toaster"
    ],
    "Toaster - 4 Slice Toaster": [
        "4 slice toaster", "big toaster"
    ],
    "Toaster - Sandwich Maker": [
        "sandwich maker", "toast maker"
    ],

    # ---------------------------
    # TOOLS & WORKSHOP
    # ---------------------------
    "Drills - Small Electric Drill": [
        "electric drill", "hand drill"
    ],
    "Drills - Large Industrial Drill": [
        "industrial drill", "big drill"
    ],
    "Grinders - Angle Grinder": [
        "angle grinder", "cutting machine"
    ],
    "Grinders - Bench Grinder": [
        "bench grinder", "workshop grinder"
    ],
    "Sewing Machines - Manual Sewing Machine": [
        "manual sewing machine", "butterfly machine"
    ],
    "Sewing Machines - Electric Sewing Machine": [
        "electric sewing machine", "brother sewing"
    ],
    "Sewing Machines - Industrial Sewing Machine": [
        "industrial sewing machine", "tailoring machine"
    ],
    "Welding Machines - Small Welding Machine": [
        "small welding machine", "mini welding"
    ],
    "Welding Machines - Large Welding Machine": [
        "industrial welding machine", "arc welder"
    ],

    # ---------------------------
    # WATER SYSTEMS
    # ---------------------------
    "Water Dispensers - Tabletop Dispenser": [
        "table water dispenser", "small dispenser"
    ],
    "Water Dispensers - Standing Dispenser (Hot/Cold)": [
        "standing water dispenser", "hot and cold dispenser"
    ],
    "Water Pump - Small Water Pump (0.5–1HP)": [
        "small water pump", "0.5hp pump"
    ],
    "Water Pump - Medium Water Pump (1.5–2HP)": [
        "medium water pump", "2hp pump"
    ],
    "Water Pump - Large Industrial Pump (3HP+)": [
        "industrial pump", "big water pump"
    ],
    "Water Pump (Solar) - DC Solar Pump": [
        "solar pump", "dc pump", "borehole solar pump"
    ],
    "Water Purifier - Small Purifier": [
        "small water purifier", "filter machine"
    ],
    "Water Purifier - Large Purifier": [
        "big purifier", "industrial purifier"
    ]
}
