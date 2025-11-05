# 📊 Solar Capstone Project - API Documentation

## Overview
This document provides a comprehensive analysis of all APIs used in the Solar Capstone project, including their requirements, costs, and setup instructions.

---

## 🔑 APIs That REQUIRE API Keys

### 🤖 LLM Providers (AI Functionality)

| API | Website | Free Tier | Status | Priority |
|-----|---------|-----------|--------|----------|
| **Groq** | [console.groq.com/keys](https://console.groq.com/keys) | 14,400 requests/day | ❌ Missing | 🔥 **CRITICAL** |
| **HuggingFace** | [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) | Rate limited | ❌ Missing | 🔥 **CRITICAL** |
| **Replicate** | [replicate.com/account/api-tokens](https://replicate.com/account/api-tokens) | Free tier available | ❌ Missing | 🟡 **MEDIUM** |
| **OpenRouter** | [openrouter.ai/keys](https://openrouter.ai/keys) | Free tier available | ❌ Missing | 🟡 **MEDIUM** |

### 🔍 Search APIs (Web Search)

| API | Website | Free Tier | Status | Priority |
|-----|---------|-----------|--------|----------|
| **Tavily** | [tavily.com](https://tavily.com/) | 1,000 searches/month | ❌ Missing | 🔥 **HIGH** |
| **SerpAPI** | [serpapi.com](https://serpapi.com/) | 100 searches/month | ❌ Missing | 🟡 **MEDIUM** |
| **Brave Search** | [brave.com/search/api](https://brave.com/search/api/) | Free tier available | ❌ Missing | 🟡 **MEDIUM** |

### 🌤️ Weather APIs

| API | Website | Free Tier | Status | Priority |
|-----|---------|-----------|--------|----------|
| **WeatherAPI** | [weatherapi.com](https://www.weatherapi.com/) | 1M calls/month | ✅ **HAVE** | ✅ **READY** |
| **Weatherbit** | [weatherbit.io](https://www.weatherbit.io/) | 500 calls/day | ✅ **HAVE** | ✅ **READY** |
| **OpenWeatherMap** | [openweathermap.org/api](https://openweathermap.org/api) | 1,000 calls/day | ❌ Missing | 🟡 **MEDIUM** |

### 🗺️ Geographic APIs

| API | Website | Free Tier | Status | Priority |
|-----|---------|-----------|--------|----------|
| **HERE Maps** | [developer.here.com](https://developer.here.com/) | 250K transactions/month | ✅ **HAVE** | ✅ **READY** |
| **Mapbox** | [account.mapbox.com](https://account.mapbox.com/) | 50K requests/month | ✅ **HAVE** | ✅ **READY** |
| **Google Maps** | [console.cloud.google.com](https://console.cloud.google.com/) | $200 credit/month | ❌ Missing | 🟡 **OPTIONAL** |

---

## 🆓 APIs That DON'T Require API Keys (Free)

### 🌤️ Weather & Solar APIs

| API | Website | Description | Status |
|-----|---------|-------------|--------|
| **Open-Meteo** | [open-meteo.com](https://open-meteo.com/) | Weather and solar data | ✅ **READY** |
| **NASA POWER** | [power.larc.nasa.gov](https://power.larc.nasa.gov/) | Solar irradiance data | ✅ **READY** |
| **PVGIS** | [re.jrc.ec.europa.eu/pvg_tools](https://re.jrc.ec.europa.eu/pvg_tools/en/) | Solar data and tools | ✅ **READY** |

### 🔍 Search APIs

| API | Website | Description | Status |
|-----|---------|-------------|--------|
| **DuckDuckGo** | Built-in | Web search (no key needed) | ✅ **READY** |
| **SearX** | Self-hosted | Meta search engine | ✅ **READY** |

### 🗺️ Geographic APIs

| API | Website | Description | Status |
|-----|---------|-------------|--------|
| **Nominatim/OpenStreetMap** | [nominatim.org](https://nominatim.org/) | Geocoding and mapping | ✅ **READY** |

---

## 🎯 Setup Priority

### Phase 1: Essential APIs (Get These First)
1. **Groq** - Fast AI responses
2. **HuggingFace** - Open source AI models
3. **Tavily** - Web search functionality

### Phase 2: Recommended APIs (Add These Next)
4. **OpenWeatherMap** - Additional weather data
5. **SerpAPI** - Alternative search option

### Phase 3: Optional APIs (Nice to Have)
6. **Replicate** - More AI model options
7. **Google Maps** - Advanced mapping features

---

## 💰 Cost Analysis

### Free APIs (No Cost)
- ✅ Open-Meteo (Unlimited)
- ✅ NASA POWER (Unlimited)
- ✅ PVGIS (Unlimited)
- ✅ DuckDuckGo (Unlimited)
- ✅ Nominatim/OpenStreetMap (Unlimited)

### Free Tier APIs (Limited Usage)
- 🔑 Groq (14,400 requests/day)
- 🔑 HuggingFace (Rate limited)
- 🔑 Tavily (1,000 searches/month)
- 🔑 WeatherAPI (1M calls/month) ✅ *Already have*
- 🔑 Weatherbit (500 calls/day) ✅ *Already have*
- 🔑 HERE Maps (250K transactions/month) ✅ *Already have*
- 🔑 Mapbox (50K requests/month) ✅ *Already have*

---

## 📋 Quick Setup Checklist

### Essential APIs
- [ ] Get Groq API key from https://console.groq.com/keys
- [ ] Get HuggingFace token from https://huggingface.co/settings/tokens
- [ ] Get Tavily API key from https://tavily.com/

### Recommended APIs
- [ ] Get OpenWeatherMap key from https://openweathermap.org/api
- [ ] Get SerpAPI key from https://serpapi.com/

### Optional APIs
- [ ] Get Replicate token from https://replicate.com/account/api-tokens
- [ ] Get Google Maps key from https://console.cloud.google.com/

---

## 🔧 Environment File Structure

```env
# ===========================================
# 🔑 LLM PROVIDERS (REQUIRE API KEYS)
# ===========================================
GROQ_API_KEY=your_groq_api_key_here
HUGGINGFACE_API_TOKEN=your_huggingface_token_here
REPLICATE_API_TOKEN=your_replicate_token_here
OPENROUTER_API_KEY=your_openrouter_key_here

# ===========================================
# 🔑 SEARCH APIs (REQUIRE API KEYS)
# ===========================================
TAVILY_API_KEY=your_tavily_api_key_here
SERP_API_KEY=your_serpapi_key_here
BRAVE_API_KEY=your_brave_key_here

# ===========================================
# 🔑 WEATHER APIs (REQUIRE API KEYS)
# ===========================================
WEATHERAPI_KEY=730fdaf7e6504a2598562147251706  # ✅ HAVE
WEATHERBIT_KEY=8e852969aa9845acb3c49104b2b7919e  # ✅ HAVE
OPENWEATHER_KEY=your_openweather_key_here

# ===========================================
# 🔑 GEOGRAPHIC APIs (REQUIRE API KEYS)
# ===========================================
HERE_API_KEY=mpzT1HNvxnUGHXTbU8KNzUwqt0dXoNzqsvF33XOEzD0  # ✅ HAVE
MAPBOX_KEY=sk.eyJ1IjoianVzdHR5eSIsImEiOiJjbWZ5ZTVxb3Mwam5zMmpzYjlvYnptN20xIn0.OHw5Hi1ptGqHoPjADCdeNg  # ✅ HAVE
GOOGLE_MAPS_KEY=your_google_maps_key_here

# ===========================================
# 🆓 FREE APIs (NO API KEYS REQUIRED)
# ===========================================
OPEN_METEO_KEY=  # Free
NASA_POWER_KEY=  # Free
PVGIS_KEY=  # Free
SEARX_URL=  # Free
```

---

## 📚 Additional Resources

- [Project README](README.md)
- [Technical Overview](docs/TECHNICAL_OVERVIEW.md)
- [How to Run Guide](HOW_TO_RUN.md)
- [API Documentation](docs/api/)

---

*Last Updated: December 2024*
*Project: Solar Capstone*