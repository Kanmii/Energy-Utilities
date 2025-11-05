# How to Run the Solar System Recommendation Platform

## 🚀 Quick Start Guide

This guide shows you how to run your complete AI-powered solar system recommendation platform with all 10 agents working together.

---

## **Method 1: Quick Start (Recommended)**

### **Step 1: Install Dependencies**
```bash
# Install all dependencies from the consolidated requirements.txt
pip install -r requirements.txt
```

### **Step 2: Start the FastAPI Backend**
```bash
# Navigate to backend directory
cd backend

# Start the FastAPI server
uvicorn app.main:app --reload
```
The backend will be available at: `http://localhost:8000`

### **Step 3: Start the Streamlit Frontend (New Terminal)**
```bash
# Run Streamlit from project root (recommended)
streamlit run frontend/streamlit/streamlit_app.py

# OR navigate to frontend directory
cd frontend/streamlit
streamlit run streamlit_app.py
```
The frontend will be available at: `http://localhost:8501`

---

## **Method 2: Docker Setup (Production)**

### **Using Docker Compose**
```bash
# Build and run with Docker Compose
docker-compose up --build
```

This will start both backend and frontend automatically.

---

## **Method 3: Environment Setup**

### **Create Environment Variables**
Create a `.env` file in the root directory:
```bash
# API Keys (get these from respective services)
OPENWEATHER_API_KEY=your_openweather_api_key_here
HERE_API_KEY=your_here_api_key_here
WEATHERBIT_API_KEY=your_weatherbit_api_key_here
NASA_POWER_API_KEY=your_nasa_power_api_key_here

# Application Settings
DEBUG=True
LOG_LEVEL=INFO
ENVIRONMENT=development
```

---

## **Method 4: Individual Component Testing**

### **Test Individual Agents**
```bash
# Test GeoAgent
python backend/app/agents/geo_agent.py

# Test SystemSizingAgent
python backend/app/agents/system_sizing_agent.py

# Test BrandIntelligenceAgent
python backend/app/agents/brand_intelligence_agent.py
```

### **Test Complete System**
```bash
# Run integration tests
python tests/integration/test_complete_system.py

# Run FastAPI tests
python tests/integration/test_fastapi_integration.py
```

---

## **Method 5: Data Processing**

### **Run Data Preparation**
```bash
# Clean and prepare all datasets
python scripts/data_prep_agent.py

# Show mapping results
python scripts/show_mapping_results.py
```

---

## **Access Points**

### **After Starting:**
- **Streamlit Frontend**: `http://localhost:8501` (User Interface)
- **FastAPI Backend**: `http://localhost:8000` (API)
- **API Documentation**: `http://localhost:8000/docs` (Interactive API docs)
- **API ReDoc**: `http://localhost:8000/redoc` (Alternative API docs)

---

## **Verification Steps**

### **Check if Everything is Working:**
1. **Backend Health**: Visit `http://localhost:8000/health`
2. **API Endpoints**: Visit `http://localhost:8000/docs`
3. **Frontend**: Visit `http://localhost:8501`
4. **Test Complete Flow**: Use the Streamlit interface to test the full system

---

## **Usage Flow**

### **Using the System:**
1. **Open Streamlit**: Go to `http://localhost:8501`
2. **Enter Your Information**:
   - Location (city, state, region)
   - Appliances (what you want to power)
   - Budget preferences
   - Autonomy days (backup power)
3. **Get Recommendations**: The system will calculate your solar system
4. **View Results**: See component recommendations, costs, and explanations

---

## **Troubleshooting**

### **Common Issues:**

#### **APIRouter Exception Handler Error:**
If you get this error:
```
AttributeError: 'APIRouter' object has no attribute 'exception_handler'
```

**Solution:** This has been fixed in the code. The exception handlers are now properly placed in `main.py` instead of `router.py`.

#### **Streamlit Import Error:**
If you get this error:
```
ModuleNotFoundError: No module named 'backend'
```

**Solution:** Run Streamlit from the project root directory:
```bash
# From project root (recommended)
streamlit run frontend/streamlit/streamlit_app.py

# NOT from frontend/streamlit directory
```

#### **Other Common Issues:**
```bash
# If you get import errors
pip install -r requirements.txt

# If Streamlit doesn't start
pip install streamlit

# If FastAPI doesn't start
pip install fastapi uvicorn

# If you get API key errors
# Make sure to set up your .env file with actual API keys
```

### **Check Dependencies:**
```bash
# Verify all packages are installed
pip list | grep -E "(fastapi|streamlit|pandas|numpy|scikit-learn)"
```

---

## **Quick Test**

### **Test the Complete System:**
1. Start both backend and frontend
2. Open `http://localhost:8501`
3. Enter a Nigerian city (e.g., "Lagos")
4. Add some appliances (e.g., "fridge", "TV", "lights")
5. Click "Calculate Solar System"
6. View your personalized recommendations!

---

## **System Architecture**

### **What's Running:**
- **9 Runtime AI Agents**: All working in coordination
- **1 Data Processing Agent**: Handles data preparation
- **FastAPI Backend**: RESTful API with all endpoints
- **Streamlit Frontend**: User-friendly interface
- **Complete Data Pipeline**: 17 datasets processed and ready

### **Agent Coordination:**
```
User Input → SuperAgent → [InputMapping, Geo, SystemSizing, BrandIntelligence, Educational, Chat, QA, Marketplace] → Results
```

---

## **Production Deployment**

### **Docker Production Setup:**
```bash
# Build production images
docker build -t solar-backend -f deployment/docker/Dockerfile.backend .
docker build -t solar-frontend -f deployment/docker/Dockerfile.frontend .

# Run with Docker Compose
docker-compose -f deployment/docker-compose.yml up -d
```

### **Kubernetes Deployment:**
```bash
# Apply Kubernetes configurations
kubectl apply -f deployment/kubernetes/
```

---

## **API Endpoints**

### **Main Endpoints:**
- `POST /api/v1/solar-system/calculate` - Complete solar system calculation
- `GET /api/v1/health` - Health check
- `GET /api/v1/agents/*` - Individual agent endpoints
- `GET /api/v1/appliances/categories` - Appliance categories
- `GET /api/v1/regions` - Available regions

### **Agent-Specific Endpoints:**
- `POST /api/v1/agents/input-mapping/process` - Appliance mapping
- `POST /api/v1/agents/geo/process` - Location intelligence
- `POST /api/v1/agents/system-sizing/calculate` - System sizing
- `POST /api/v1/agents/brand-intelligence/recommend` - Component recommendations
- `POST /api/v1/agents/educational/guidance` - Educational content
- `POST /api/v1/agents/chat/message` - Chat interface
- `POST /api/v1/agents/qa/ask` - Question answering
- `POST /api/v1/agents/marketplace/search` - Marketplace search

---

## **Development Mode**

### **Hot Reload Development:**
```bash
# Backend with auto-reload
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# Frontend with auto-reload
streamlit run streamlit_app.py --server.runOnSave true
```

### **Debug Mode:**
```bash
# Enable debug logging
export DEBUG=True
export LOG_LEVEL=DEBUG

# Run with debug output
python -m uvicorn app.main:app --reload --log-level debug
```

---

## **Performance Monitoring**

### **Check System Performance:**
```bash
# Monitor API performance
curl http://localhost:8000/health

# Check agent response times
curl -X POST http://localhost:8000/api/v1/solar-system/calculate \
  -H "Content-Type: application/json" \
  -d '{"appliances": [{"name": "fridge", "category": "kitchen", "usage_hours": 24, "quantity": 1}], "location": {"city": "Lagos", "state": "Lagos", "region": "South West"}, "preferences": {"autonomy_days": 3, "battery_chemistry": "LiFePO4", "budget_range": "medium", "quality_threshold": 0.8}}'
```

---

## **Support & Help**

### **If You Need Help:**
1. Check the logs in the terminal where you started the services
2. Visit the API documentation at `http://localhost:8000/docs`
3. Test individual components using the test scripts
4. Verify all dependencies are installed correctly

### **Common Solutions:**
- **Port conflicts**: Change ports in the startup commands
- **Import errors**: Reinstall requirements.txt
- **API key errors**: Set up your .env file properly
- **Data errors**: Run the data preparation scripts first

---

**Your AI-powered solar system recommendation platform is now running with all 10 agents working together! 🌟**

**Happy Solar System Planning! ☀️**
