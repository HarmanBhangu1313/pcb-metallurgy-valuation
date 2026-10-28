"""
metallurgy_engine.py
====================
Industrial Metallurgical Characterization, Live Market Valuation & Profit Engine
for E-Waste Printed Circuit Boards (PCBs).
Integrated with Phase 1 Computer Vision (YOLOv8) Component Detection.

Features:
  1. Range-Based Elemental Assays (Min, Avg, Max) per Component Class.
  2. Live Market Price Engine (Yahoo Finance Free API / GoldAPI / Web Scraper).
  3. Gross Valuation Range & Stoichiometric Chemical Reagent Model.
  4. Operational Extraction Cost & Net Profit Range (USD $ and INR ₹).
  5. Component-Specific Metallurgy Profiler (e.g., Gold in ICs, Palladium in MLCCs).

Department of Metallurgical Engineering
Indian Institute of Technology (BHU), Varanasi
Course: MT-392 | Authors: Sourav Kumar Agarwal, Rituraj Singh, Harman Singh Bhangu
Supervisor: Dr. Kamalesh Kumar Singh | Mentor: Mr. Sayantan Mukherjee
"""

import os
import time
import json
import ssl
import urllib.request
from collections import Counter
from datetime import datetime

# ==============================================================================
# 1. RANGE-BASED METALLURGICAL ASSAY DATABASE (Literature Benchmarks)
# References: Havlík et al. (2010), Veit et al. (2005), Kaya (2016),
# Charles et al. (2017), Cui & Zhang (2008), Syed (2012).
# ==============================================================================

COMPONENT_METALLURGY = {
    "IC": {
        "avg_mass_g": 1.25,
        "mass_range_g": {"min": 0.80, "avg": 1.25, "max": 2.80},
        "description": "Integrated Circuit (DIP/QFP/BGA/SOP packages with Cu leadframe & Au wire bonds)",
        "elements_percent": {
            "Cu": 28.5,
            "Au": 0.080,    # ~1.00 mg Au per IC (standard dual-wire bonding)
            "Ag": 0.150,    # Silver die-attach adhesive
            "Ni": 2.10,     # Barrier layer
            "Sn": 4.50,     # Solder lead finish
            "Si": 8.00,
            "Resin": 56.67
        },
        "elements_percent_range": {
            "Au": {"min": 0.025, "avg": 0.080, "max": 0.180}, # 0.20 mg to 5.04 mg Au/IC
            "Cu": {"min": 18.0,  "avg": 28.5,  "max": 40.0},
            "Ag": {"min": 0.06,  "avg": 0.15,  "max": 0.35},
            "Ni": {"min": 1.00,  "avg": 2.10,  "max": 3.80},
            "Sn": {"min": 2.50,  "avg": 4.50,  "max": 7.50}
        }
    },
    "capacitor": {
        "avg_mass_g": 0.045,
        "mass_range_g": {"min": 0.015, "avg": 0.045, "max": 0.120},
        "description": "Ceramic Capacitor / MLCC (BaTiO3 dielectric with Pd/Ag sintered inner electrodes)",
        "elements_percent": {
            "Pd": 1.20,     # ~0.54 mg Pd per MLCC
            "Ag": 8.50,     # ~3.82 mg Ag per MLCC
            "Ni": 3.20,
            "Cu": 1.50,
            "Sn": 2.10,
            "BaTiO3": 83.5
        },
        "elements_percent_range": {
            "Pd": {"min": 0.30, "avg": 1.20, "max": 2.80},  # Base-metal vs precious-metal electrode
            "Ag": {"min": 3.50, "avg": 8.50, "max": 16.0},  # High-Ag termination paste
            "Ni": {"min": 1.80, "avg": 3.20, "max": 5.50},
            "Cu": {"min": 0.80, "avg": 1.50, "max": 2.80},
            "Sn": {"min": 1.20, "avg": 2.10, "max": 3.50}
        }
    },
    "connector": {
        "avg_mass_g": 2.80,
        "mass_range_g": {"min": 1.20, "avg": 2.80, "max": 5.50},
        "description": "Gold-Plated Edge Connector / Pin Header (Phosphor bronze / brass substrate)",
        "elements_percent": {
            "Cu": 68.0,
            "Au": 0.045,    # ~1.26 mg Au per connector cluster (30 micro-inch hard gold)
            "Ni": 3.50,     # Nickel underplate barrier
            "Sn": 6.50,
            "Plastic": 21.955
        },
        "elements_percent_range": {
            "Au": {"min": 0.015, "avg": 0.045, "max": 0.120}, # 10 u-in flash up to 50 u-in military
            "Cu": {"min": 52.0,  "avg": 68.0,  "max": 78.0},
            "Ni": {"min": 1.80,  "avg": 3.50,  "max": 5.20},
            "Sn": {"min": 3.50,  "avg": 6.50,  "max": 10.0}
        }
    },
    "transformer": {
        "avg_mass_g": 28.5,
        "mass_range_g": {"min": 16.0, "avg": 28.5, "max": 52.0},
        "description": "Switch-Mode Power Transformer (Heavy pure annealed Cu magnet wire on ferrite core)",
        "elements_percent": {
            "Cu": 36.5,     # ~10.4 g pure Cu per unit
            "Fe_Ferrite": 48.0,
            "Insulation": 15.5
        },
        "elements_percent_range": {
            "Cu": {"min": 28.0, "avg": 36.5, "max": 46.0}
        }
    },
    "transistor": {
        "avg_mass_g": 0.85,
        "mass_range_g": {"min": 0.40, "avg": 0.85, "max": 1.90},
        "description": "Power Transistor / MOSFET (TO-220/DPAK with massive Cu heat-sink tab)",
        "elements_percent": {
            "Cu": 52.0,     # ~0.44 g Cu tab
            "Au": 0.015,    # ~0.13 mg Au wire bond
            "Ni": 2.50,
            "Sn": 3.50,
            "Si": 5.00,
            "Epoxy": 36.985
        },
        "elements_percent_range": {
            "Cu": {"min": 38.0,  "avg": 52.0,  "max": 65.0},
            "Au": {"min": 0.005, "avg": 0.015, "max": 0.035},
            "Ni": {"min": 1.50,  "avg": 2.50,  "max": 4.00},
            "Sn": {"min": 2.00,  "avg": 3.50,  "max": 5.50}
        }
    },
    "resistor": {
        "avg_mass_g": 0.025,
        "mass_range_g": {"min": 0.010, "avg": 0.025, "max": 0.060},
        "description": "Thick-Film Chip Resistor (Al2O3 ceramic substrate with RuO2-Ag glaze)",
        "elements_percent": {
            "Ag": 1.80,     # ~0.45 mg Ag terminal paste
            "Ru": 0.12,     # Ruthenium conductive oxide
            "Ni": 1.50,
            "Cu": 1.20,
            "Sn": 2.00,
            "Al2O3": 93.38
        },
        "elements_percent_range": {
            "Ag": {"min": 0.80, "avg": 1.80, "max": 3.20},
            "Ru": {"min": 0.05, "avg": 0.12, "max": 0.25},
            "Ni": {"min": 0.80, "avg": 1.50, "max": 2.50},
            "Cu": {"min": 0.60, "avg": 1.20, "max": 2.00},
            "Sn": {"min": 1.00, "avg": 2.00, "max": 3.50}
        }
    },
    "power capacitor": {
        "avg_mass_g": 4.50,
        "mass_range_g": {"min": 2.20, "avg": 4.50, "max": 11.0},
        "description": "Aluminum Electrolytic Capacitor (High-purity etched Al anode/cathode foils)",
        "elements_percent": {
            "Al": 38.0,     # ~1.71 g pure Al
            "Cu": 4.50,     # Lead wires
            "Fe": 3.00,
            "Sn": 1.50,
            "Electrolyte_Paper": 53.0
        },
        "elements_percent_range": {
            "Al": {"min": 28.0, "avg": 38.0, "max": 46.0},
            "Cu": {"min": 2.50, "avg": 4.50, "max": 7.00},
            "Sn": {"min": 0.80, "avg": 1.50, "max": 2.50}
        }
    },
    "inductor": {
        "avg_mass_g": 2.10,
        "mass_range_g": {"min": 0.90, "avg": 2.10, "max": 4.80},
        "description": "SMD / Toroidal Power Inductor (Coiled enamelled copper wire)",
        "elements_percent": {
            "Cu": 32.0,     # ~0.67 g Cu
            "Ferrite": 58.0,
            "Resin": 10.0
        },
        "elements_percent_range": {
            "Cu": {"min": 22.0, "avg": 32.0, "max": 44.0}
        }
    },
    "LED": {
        "avg_mass_g": 0.06,
        "mass_range_g": {"min": 0.025, "avg": 0.060, "max": 0.140},
        "description": "Light Emitting Diode (GaAs/GaN optical die with micro Au bond wire)",
        "elements_percent": {
            "Cu": 35.0,
            "Au": 0.060,    # ~0.036 mg Au wire
            "Sn": 5.00,
            "Epoxy": 59.94
        },
        "elements_percent_range": {
            "Au": {"min": 0.020, "avg": 0.060, "max": 0.120},
            "Cu": {"min": 24.0,  "avg": 35.0,  "max": 46.0},
            "Sn": {"min": 2.50,  "avg": 5.00,  "max": 8.00}
        }
    },
    "diode": {
        "avg_mass_g": 0.35,
        "mass_range_g": {"min": 0.18, "avg": 0.35, "max": 0.75},
        "description": "Schottky / Rectifier Diode (Heavy axial/SMD Cu leads)",
        "elements_percent": {
            "Cu": 45.0,
            "Ag": 0.08,
            "Sn": 4.00,
            "Si_Glass": 50.92
        },
        "elements_percent_range": {
            "Cu": {"min": 32.0, "avg": 45.0, "max": 58.0},
            "Ag": {"min": 0.03, "avg": 0.08, "max": 0.16},
            "Sn": {"min": 2.00, "avg": 4.00, "max": 6.50}
        }
    },
    "mov": {
        "avg_mass_g": 1.80,
        "mass_range_g": {"min": 0.90, "avg": 1.80, "max": 3.50},
        "description": "Metal Oxide Varistor (ZnO ceramic disc with silver electrode coatings)",
        "elements_percent": {
            "Ag": 0.95,     # ~17.1 mg Ag wash coating
            "Cu": 12.0,
            "ZnO": 82.0,
            "Polymer": 5.05
        },
        "elements_percent_range": {
            "Ag": {"min": 0.45, "avg": 0.95, "max": 1.75},
            "Cu": {"min": 8.00, "avg": 12.0, "max": 17.0}
        }
    },
    "switch": {
        "avg_mass_g": 1.10,
        "mass_range_g": {"min": 0.50, "avg": 1.10, "max": 2.20},
        "description": "Tactile / Slide Switch (Silver-plated beryllium copper contacts)",
        "elements_percent": {
            "Cu": 38.0,
            "Ag": 0.25,     # Contact plating
            "Ni": 2.00,
            "Plastic": 59.75
        },
        "elements_percent_range": {
            "Cu": {"min": 26.0, "avg": 38.0, "max": 48.0},
            "Ag": {"min": 0.10, "avg": 0.25, "max": 0.55},
            "Ni": {"min": 1.00, "avg": 2.00, "max": 3.50}
        }
    },
    "emi filter": {
        "avg_mass_g": 4.20,
        "mass_range_g": {"min": 2.40, "avg": 4.20, "max": 7.80},
        "description": "EMI Line Suppression Filter (Bifilar coiled Cu windings on ferrite)",
        "elements_percent": {
            "Cu": 28.0,
            "Ferrite": 55.0,
            "Plastic": 17.0
        },
        "elements_percent_range": {
            "Cu": {"min": 18.0, "avg": 28.0, "max": 38.0}
        }
    },
    "ferrite bead": {
        "avg_mass_g": 0.05,
        "mass_range_g": {"min": 0.02, "avg": 0.05, "max": 0.10},
        "description": "Ferrite suppression bead with Ag inner conductor",
        "elements_percent": {
            "Ferrite": 94.0,
            "Ag": 1.20,
            "Ni_Sn": 4.80
        },
        "elements_percent_range": {
            "Ag": {"min": 0.50, "avg": 1.20, "max": 2.20}
        }
    },
    "potentiometer": {
        "avg_mass_g": 3.20,
        "mass_range_g": {"min": 1.60, "avg": 3.20, "max": 5.80},
        "description": "Trimpot / Rotary Potentiometer (Cu wiper & Ag tracking traces)",
        "elements_percent": {
            "Cu": 24.0,
            "Ag": 0.15,
            "C_Ceramic": 45.0,
            "Steel": 30.85
        },
        "elements_percent_range": {
            "Cu": {"min": 16.0, "avg": 24.0, "max": 32.0},
            "Ag": {"min": 0.06, "avg": 0.15, "max": 0.28}
        }
    },
    "relay": {
        "avg_mass_g": 12.5,
        "mass_range_g": {"min": 6.50, "avg": 12.5, "max": 22.0},
        "description": "Electromechanical Relay (Ag-Ni heavy contacts & Cu solenoid coil)",
        "elements_percent": {
            "Cu": 34.0,     # ~4.25 g Cu coil
            "Ag": 0.85,     # ~106 mg Ag solid contact points
            "Fe": 38.0,
            "Plastic": 27.15
        },
        "elements_percent_range": {
            "Cu": {"min": 24.0, "avg": 34.0, "max": 44.0},
            "Ag": {"min": 0.45, "avg": 0.85, "max": 1.55}
        }
    },
    "battery": {
        "avg_mass_g": 3.0,
        "mass_range_g": {"min": 2.50, "avg": 3.00, "max": 3.60},
        "description": "Lithium Coin Cell (CR2032 casing with Ni plating)",
        "elements_percent": {
            "Stainless_Steel": 72.0,
            "Li": 2.50,
            "MnO2": 22.0,
            "Ni": 3.50
        },
        "elements_percent_range": {
            "Ni": {"min": 2.20, "avg": 3.50, "max": 5.00}
        }
    },
    "display": {
        "avg_mass_g": 5.5,
        "mass_range_g": {"min": 3.20, "avg": 5.50, "max": 11.0},
        "description": "LCD/OLED Display Module (Cu ribbon & Indium Tin Oxide ITO)",
        "elements_percent": {
            "Glass": 78.0,
            "Cu": 8.0,
            "In": 0.05,
            "Plastic": 13.95
        },
        "elements_percent_range": {
            "Cu": {"min": 5.00, "avg": 8.00, "max": 13.0},
            "In": {"min": 0.02, "avg": 0.05, "max": 0.10}
        }
    }
}

# Hydrometallurgical recovery efficiencies based on lab flowsheets
RECOVERY_EFFICIENCY = {
    "Cu": 0.94,     # 2.0 M HNO3 / H2SO4 leaching
    "Au": 0.91,     # Aqua Regia / Ammonium Thiosulfate
    "Pd": 0.89,     # HCl + Cl2 / Aqua Regia extraction
    "Ag": 0.88,     # Nitric leaching + NaCl precipitation
    "Al": 0.90,     # NaOH digestion
    "Ni": 0.88,     # Acid leaching + solvent extraction
    "Sn": 0.85,     # Metastannic acid calcination / electrowinning
    "Pt": 0.90,
    "Ru": 0.75,
    "In": 0.80
}

# Baseline Calibrated Benchmark Prices (USD per gram)
# Used as reliable fallback if internet connectivity is absent during presentation
BENCHMARK_PRICES_USD = {
    "Au": 85.00,    # ~$2,640 / troy oz
    "Pd": 34.00,    # ~$1,058 / troy oz
    "Ag": 1.05,     # ~$32.65 / troy oz
    "Pt": 32.50,    # ~$1,010 / troy oz
    "Cu": 0.0095,   # ~$9,500 / metric ton
    "Al": 0.0026,   # ~$2,600 / metric ton
    "Ni": 0.0165,   # ~$16,500 / metric ton
    "Sn": 0.0310,   # ~$31,000 / metric ton
    "Ru": 14.50,    # Ruthenium
    "In": 0.280     # Indium
}

BENCHMARK_USD_INR = 86.20

# Cached live prices to avoid redundant calls within the same run
_LIVE_PRICE_CACHE = {
    "data": None,
    "timestamp": 0.0
}


# ==============================================================================
# 2. LIVE MARKET PRICE ENGINE (Free API / Zero-Key Public Feeds)
# ==============================================================================

def fetch_live_metal_prices(use_live: bool = True, api_key: str = None, force_refresh: bool = False) -> dict:
    """
    Fetches real-time commodity spot prices for gold, silver, copper, palladium, etc.
    
    Data Feeds:
      - Primary: Yahoo Finance public quote endpoint (100% Free, NO API KEY required).
                 Tickers: GC=F (Au), SI=F (Ag), PA=F (Pd), PL=F (Pt), HG=F (Cu), ALI=F (Al), USDINR=X.
      - Secondary: GoldAPI.io (if user passes a free API key).
      - Fallback: Benchmark spot prices (ensures zero crash risk during offline presentation).
    """
    global _LIVE_PRICE_CACHE
    now = time.time()
    
    if not force_refresh and _LIVE_PRICE_CACHE["data"] is not None and (now - _LIVE_PRICE_CACHE["timestamp"]) < 300:
        return _LIVE_PRICE_CACHE["data"]

    prices_usd = BENCHMARK_PRICES_USD.copy()
    usd_inr = BENCHMARK_USD_INR
    source = "IIT BHU Metallurgical Benchmark Spot Prices (Cached)"
    status = "BENCHMARK / OFFLINE"
    ticker_quotes = {}

    if use_live:
        # 1. Try GoldAPI.io if a free key is provided
        if api_key:
            try:
                headers = {"x-access-goldapi-key": api_key.strip(), "Content-Type": "application/json"}
                ctx = ssl._create_unverified_context()
                for metal_code, sym in [("Au", "XAU"), ("Ag", "XAG"), ("Pd", "XPD"), ("Pt", "XPT")]:
                    url = f"https://www.goldapi.io/api/{sym}/USD"
                    req = urllib.request.Request(url, headers=headers)
                    with urllib.request.urlopen(req, context=ctx, timeout=4) as resp:
                        j = json.loads(resp.read().decode())
                        if "price_gram_24k" in j:
                            prices_usd[metal_code] = float(j["price_gram_24k"])
                source = "GoldAPI.io Live Feed"
                status = "LIVE"
            except Exception:
                pass

        # 2. Free Zero-Key Yahoo Finance Endpoint (Institutional Commodity Quotes)
        if status != "LIVE":
            try:
                ctx = ssl._create_unverified_context()
                headers = {
                    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
                }
                
                # Tickers and unit conversion factors to USD per gram
                # 1 troy oz = 31.1034768 grams | 1 lb = 453.59237 grams | 1 metric ton = 1,000,000 grams
                tickers = {
                    "Au": ("GC=F", 31.1035, "troy oz"),
                    "Ag": ("SI=F", 31.1035, "troy oz"),
                    "Pd": ("PA=F", 31.1035, "troy oz"),
                    "Pt": ("PL=F", 31.1035, "troy oz"),
                    "Cu": ("HG=F", 453.592, "lb"),
                    "Al": ("ALI=F", 1000000.0, "tonne")
                }
                
                fetched_count = 0
                for metal, (ticker, divisor, unit) in tickers.items():
                    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}?interval=1d&range=1d"
                    req = urllib.request.Request(url, headers=headers)
                    try:
                        with urllib.request.urlopen(req, context=ctx, timeout=3.5) as resp:
                            data = json.loads(resp.read().decode())
                            quote = data["chart"]["result"][0]["meta"]["regularMarketPrice"]
                            prices_usd[metal] = round(quote / divisor, 4)
                            ticker_quotes[metal] = {"ticker": ticker, "quote": quote, "unit": unit}
                            fetched_count += 1
                    except Exception:
                        continue
                
                # Fetch live USD to INR exchange rate
                url_inr = "https://query1.finance.yahoo.com/v8/finance/chart/USDINR=X?interval=1d&range=1d"
                req_inr = urllib.request.Request(url_inr, headers=headers)
                try:
                    with urllib.request.urlopen(req_inr, context=ctx, timeout=3.0) as resp:
                        data_inr = json.loads(resp.read().decode())
                        usd_inr = round(data_inr["chart"]["result"][0]["meta"]["regularMarketPrice"], 2)
                except Exception:
                    usd_inr = BENCHMARK_USD_INR

                if fetched_count >= 3:
                    source = "Yahoo Finance Live Institutional Commodity Feeds (Zero-Key API)"
                    status = "LIVE"
            except Exception:
                status = "BENCHMARK / OFFLINE"

    # Compute INR prices per gram
    prices_inr = {k: round(v * usd_inr, 2) for k, v in prices_usd.items()}

    result = {
        "status": status,
        "source": source,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S IST"),
        "usd_inr": usd_inr,
        "prices_usd_per_g": prices_usd,
        "prices_inr_per_g": prices_inr,
        "ticker_quotes": ticker_quotes
    }
    _LIVE_PRICE_CACHE["data"] = result
    _LIVE_PRICE_CACHE["timestamp"] = now
    return result


# ==============================================================================
# 3. COMPONENT-SPECIFIC RANGE PROFILER (e.g., Gold in ICs)
# ==============================================================================

def get_component_range_profile(comp_name: str, metal: str = None, count: int = 1, market_data: dict = None) -> dict:
    """
    Calculates the detailed Min, Avg, and Max metal content and corresponding monetary
    values for a specific component class.
    
    Example:
      get_component_range_profile("IC", "Au", count=12)
      -> Computes mg of Gold in 12 ICs (Min, Avg, Max) and financial yield in USD & INR.
    """
    if comp_name not in COMPONENT_METALLURGY:
        raise ValueError(f"Unknown component class '{comp_name}'. Available: {list(COMPONENT_METALLURGY.keys())}")
    
    if market_data is None:
        market_data = fetch_live_metal_prices(use_live=True)

    prices_usd = market_data["prices_usd_per_g"]
    usd_inr = market_data["usd_inr"]
    comp = COMPONENT_METALLURGY[comp_name]
    
    mass_min = comp["mass_range_g"]["min"]
    mass_avg = comp["mass_range_g"]["avg"]
    mass_max = comp["mass_range_g"]["max"]
    
    elem_ranges = comp.get("elements_percent_range", {})
    profiles = {}

    target_metals = [metal] if metal else list(elem_ranges.keys())
    
    for m in target_metals:
        if m not in elem_ranges:
            continue
        p_min = elem_ranges[m]["min"] / 100.0
        p_avg = elem_ranges[m]["avg"] / 100.0
        p_max = elem_ranges[m]["max"] / 100.0
        
        # Unit metal mass in grams and milligrams
        unit_m_min_g = mass_min * p_min
        unit_m_avg_g = mass_avg * p_avg
        unit_m_max_g = mass_max * p_max
        
        tot_m_min_g = unit_m_min_g * count
        tot_m_avg_g = unit_m_avg_g * count
        tot_m_max_g = unit_m_max_g * count
        
        # Chemical recovery yield
        eff = RECOVERY_EFFICIENCY.get(m, 0.85)
        rec_min_g = tot_m_min_g * eff
        rec_avg_g = tot_m_avg_g * eff
        rec_max_g = tot_m_max_g * eff
        
        price_g_usd = prices_usd.get(m, 0.0)
        
        val_min_usd = rec_min_g * price_g_usd
        val_avg_usd = rec_avg_g * price_g_usd
        val_max_usd = rec_max_g * price_g_usd
        
        profiles[m] = {
            "unit_mass_mg": {
                "min": round(unit_m_min_g * 1000.0, 3),
                "avg": round(unit_m_avg_g * 1000.0, 3),
                "max": round(unit_m_max_g * 1000.0, 3)
            },
            "total_mass_mg": {
                "min": round(tot_m_min_g * 1000.0, 3),
                "avg": round(tot_m_avg_g * 1000.0, 3),
                "max": round(tot_m_max_g * 1000.0, 3)
            },
            "recoverable_mass_mg": {
                "min": round(rec_min_g * 1000.0, 3),
                "avg": round(rec_avg_g * 1000.0, 3),
                "max": round(rec_max_g * 1000.0, 3)
            },
            "recoverable_value_usd": {
                "min": round(val_min_usd, 4),
                "avg": round(val_avg_usd, 4),
                "max": round(val_max_usd, 4)
            },
            "recoverable_value_inr": {
                "min": round(val_min_usd * usd_inr, 2),
                "avg": round(val_avg_usd * usd_inr, 2),
                "max": round(val_max_usd * usd_inr, 2)
            },
            "market_price_used_usd_per_g": price_g_usd,
            "recovery_efficiency": eff
        }

    return {
        "component": comp_name,
        "count": count,
        "description": comp["description"],
        "unit_weight_range_g": comp["mass_range_g"],
        "elements": profiles,
        "market_status": market_data["status"]
    }


# ==============================================================================
# 4. QUANTITATIVE YIELD, REAGENT & PROFIT RANGE ENGINE
# ==============================================================================

def calculate_metallurgical_yield(
    component_counts: dict,
    use_live_prices: bool = True,
    custom_prices: dict = None,
    api_key: str = None
) -> dict:
    """
    Computes range-based elemental inventories, recoverable masses, stoichiometric
    reagent requirements, operational processing costs, and NET PROFIT RANGES.
    """
    # 1. Fetch Market Prices
    market_data = fetch_live_metal_prices(use_live=use_live_prices, api_key=api_key)
    prices_usd = market_data["prices_usd_per_g"].copy()
    if custom_prices:
        prices_usd.update(custom_prices)
    usd_inr = market_data["usd_inr"]

    target_metals = ["Au", "Pd", "Ag", "Cu", "Al", "Ni", "Sn", "Pt", "Ru", "In"]
    
    # 2. Compute Elemental Inventory Ranges (Min, Avg, Max)
    total_comp_count = sum(component_counts.values())
    tot_mass_min_g = 0.0
    tot_mass_avg_g = 0.0
    tot_mass_max_g = 0.0
    
    raw_elem_min_g = Counter()
    raw_elem_avg_g = Counter()
    raw_elem_max_g = Counter()
    
    component_breakdown = {}

    for comp_name, count in component_counts.items():
        if comp_name not in COMPONENT_METALLURGY or count <= 0:
            continue
        comp_info = COMPONENT_METALLURGY[comp_name]
        
        # Mass ranges
        m_min = comp_info["mass_range_g"]["min"] * count
        m_avg = comp_info["mass_range_g"]["avg"] * count
        m_max = comp_info["mass_range_g"]["max"] * count
        
        tot_mass_min_g += m_min
        tot_mass_avg_g += m_avg
        tot_mass_max_g += m_max
        
        comp_contrib_usd = {"min": 0.0, "avg": 0.0, "max": 0.0}
        comp_elem_summary = {}

        # Elements range
        ranges = comp_info.get("elements_percent_range", {})
        fixed_pct = comp_info.get("elements_percent", {})
        
        # Consider all elements present in this component
        all_elems = set(ranges.keys()) | set(fixed_pct.keys())
        
        for elem in all_elems:
            if elem in ranges:
                pct_min = ranges[elem]["min"] / 100.0
                pct_avg = ranges[elem]["avg"] / 100.0
                pct_max = ranges[elem]["max"] / 100.0
            else:
                pct = fixed_pct.get(elem, 0.0) / 100.0
                pct_min = pct * 0.85
                pct_avg = pct
                pct_max = pct * 1.15
            
            e_min_g = m_min * pct_min
            e_avg_g = m_avg * pct_avg
            e_max_g = m_max * pct_max
            
            raw_elem_min_g[elem] += e_min_g
            raw_elem_avg_g[elem] += e_avg_g
            raw_elem_max_g[elem] += e_max_g
            
            if elem in target_metals:
                eff = RECOVERY_EFFICIENCY.get(elem, 0.85)
                pr = prices_usd.get(elem, 0.0)
                v_min = e_min_g * eff * pr
                v_avg = e_avg_g * eff * pr
                v_max = e_max_g * eff * pr
                
                comp_contrib_usd["min"] += v_min
                comp_contrib_usd["avg"] += v_avg
                comp_contrib_usd["max"] += v_max
                
                comp_elem_summary[elem] = {
                    "avg_mass_mg": round(e_avg_g * 1000.0, 2),
                    "avg_val_usd": round(v_avg, 3)
                }

        component_breakdown[comp_name] = {
            "count": count,
            "unit_description": comp_info["description"],
            "avg_mass_total_g": round(m_avg, 2),
            "recoverable_value_usd_range": {
                "min": round(comp_contrib_usd["min"], 3),
                "avg": round(comp_contrib_usd["avg"], 3),
                "max": round(comp_contrib_usd["max"], 3)
            },
            "recoverable_value_inr_range": {
                "min": round(comp_contrib_usd["min"] * usd_inr, 2),
                "avg": round(comp_contrib_usd["avg"] * usd_inr, 2),
                "max": round(comp_contrib_usd["max"] * usd_inr, 2)
            },
            "key_metals": comp_elem_summary
        }

    # 3. Calculate Recoverable Yields & Gross Revenues
    metals_summary = {}
    tot_gross_min_usd = 0.0
    tot_gross_avg_usd = 0.0
    tot_gross_max_usd = 0.0

    for metal in target_metals:
        if raw_elem_avg_g[metal] <= 0:
            continue
        eff = RECOVERY_EFFICIENCY.get(metal, 0.85)
        pr_usd = prices_usd.get(metal, 0.0)
        
        # Mass in grams
        rm_min = raw_elem_min_g[metal] * eff
        rm_avg = raw_elem_avg_g[metal] * eff
        rm_max = raw_elem_max_g[metal] * eff
        
        # Value in USD
        gv_min = rm_min * pr_usd
        gv_avg = rm_avg * pr_usd
        gv_max = rm_max * pr_usd
        
        tot_gross_min_usd += gv_min
        tot_gross_avg_usd += gv_avg
        tot_gross_max_usd += gv_max
        
        metals_summary[metal] = {
            "recovery_efficiency": eff,
            "market_price_usd_per_g": pr_usd,
            "market_price_inr_per_g": round(pr_usd * usd_inr, 2),
            "recoverable_mass_mg_range": {
                "min": round(rm_min * 1000.0, 2),
                "avg": round(rm_avg * 1000.0, 2),
                "max": round(rm_max * 1000.0, 2)
            },
            "recoverable_mass_g_range": {
                "min": round(rm_min, 4),
                "avg": round(rm_avg, 4),
                "max": round(rm_max, 4)
            },
            "gross_revenue_usd_range": {
                "min": round(gv_min, 4),
                "avg": round(gv_avg, 4),
                "max": round(gv_max, 4)
            },
            "gross_revenue_inr_range": {
                "min": round(gv_min * usd_inr, 2),
                "avg": round(gv_avg * usd_inr, 2),
                "max": round(gv_max * usd_inr, 2)
            }
        }

    # 4. Stoichiometric Chemical Reagents & Operational Cost Model
    # Reagent 1: Nitric acid for base metals (3 Cu + 8 HNO3 -> 3 Cu(NO3)2 + 2 NO + 4 H2O)
    cu_avg_g = raw_elem_avg_g["Cu"]
    cu_min_g = raw_elem_min_g["Cu"]
    cu_max_g = raw_elem_max_g["Cu"]
    
    # 68% conc HNO3 is ~15.1 M (density 1.41 g/mL)
    def calc_hno3_ml(cu_g):
        moles_cu = cu_g / 63.546
        moles_hno3 = moles_cu * (8.0 / 3.0)
        return (moles_hno3 / 15.1) * 1000.0 * 1.25 # 25% stoichiometric excess

    hno3_min_ml = calc_hno3_ml(cu_min_g)
    hno3_avg_ml = calc_hno3_ml(cu_avg_g)
    hno3_max_ml = calc_hno3_ml(cu_max_g)

    # Reagent 2: Aqua Regia for Au & Pd leaching
    def calc_ar_ml(au_g, pd_g):
        return max(15.0, (au_g + pd_g) * 120.0 + 10.0)

    ar_min_ml = calc_ar_ml(raw_elem_min_g["Au"], raw_elem_min_g["Pd"])
    ar_avg_ml = calc_ar_ml(raw_elem_avg_g["Au"], raw_elem_avg_g["Pd"])
    ar_max_ml = calc_ar_ml(raw_elem_max_g["Au"], raw_elem_max_g["Pd"])

    # Reagent 3: Sodium metabisulfite (Na2S2O5) precipitant for Au powder
    def calc_na2s2o5_g(au_g):
        moles_au = au_g / 196.966
        return moles_au * 1.5 * 190.107 * 1.20

    na2s2o5_avg_g = calc_na2s2o5_g(raw_elem_avg_g["Au"])

    # Industrial Processing Unit Costs (Commercial Chemical & Electricity Benchmarks):
    # - 68% Industrial HNO3: ~$0.45 per Liter ($0.00045/mL)
    # - Aqua Regia (3:1 HCl:HNO3): ~$0.70 per Liter ($0.00070/mL)
    # - Neutralization reagents (Lime/NaOH) + Na2S2O5: ~$0.030 per board
    # - Electricity (Agitation, heating & Cu electrowinning @ 3 kWh/kg Cu, $0.10/kWh): ~$0.015 - $0.035
    # - Selective mechanical/vision disassembly labor amortization: ~$0.10 - $0.20 per board
    def calc_processing_cost_usd(hno3_ml, ar_ml, cu_g, fixed_labor):
        acid_cost = (hno3_ml * 0.00045) + (ar_ml * 0.00070)
        salt_neutral_cost = 0.025
        electricity_cost = (cu_g / 1000.0) * 3.0 * 0.10 + 0.015
        return acid_cost + salt_neutral_cost + electricity_cost + fixed_labor

    proc_cost_min_usd = calc_processing_cost_usd(hno3_min_ml, ar_min_ml, cu_min_g, fixed_labor=0.08)
    proc_cost_avg_usd = calc_processing_cost_usd(hno3_avg_ml, ar_avg_ml, cu_avg_g, fixed_labor=0.14)
    proc_cost_max_usd = calc_processing_cost_usd(hno3_max_ml, ar_max_ml, cu_max_g, fixed_labor=0.22)

    # 5. Net Profit Ranges (Revenue - Processing Cost)
    # Conservative lower bound: Min Revenue - Max Cost
    # Most likely / Expected: Avg Revenue - Avg Cost
    # Optimistic upper bound: Max Revenue - Min Cost
    net_profit_min_usd = max(0.0, tot_gross_min_usd - proc_cost_max_usd)
    net_profit_avg_usd = max(0.0, tot_gross_avg_usd - proc_cost_avg_usd)
    net_profit_max_usd = max(0.0, tot_gross_max_usd - proc_cost_min_usd)

    margin_min = (net_profit_min_usd / tot_gross_min_usd * 100.0) if tot_gross_min_usd > 0 else 0.0
    margin_avg = (net_profit_avg_usd / tot_gross_avg_usd * 100.0) if tot_gross_avg_usd > 0 else 0.0
    margin_max = (net_profit_max_usd / tot_gross_max_usd * 100.0) if tot_gross_max_usd > 0 else 0.0

    report = {
        "summary": {
            "total_detected_components": total_comp_count,
            "component_mass_g_range": {
                "min": round(tot_mass_min_g, 2),
                "avg": round(tot_mass_avg_g, 2),
                "max": round(tot_mass_max_g, 2)
            },
            "market_feed_status": market_data["status"],
            "market_feed_source": market_data["source"],
            "market_feed_timestamp": market_data["timestamp"],
            "usd_to_inr_rate": usd_inr
        },
        "gross_valuation_usd": {
            "min": round(tot_gross_min_usd, 2),
            "avg": round(tot_gross_avg_usd, 2),
            "max": round(tot_gross_max_usd, 2)
        },
        "gross_valuation_inr": {
            "min": round(tot_gross_min_usd * usd_inr, 2),
            "avg": round(tot_gross_avg_usd * usd_inr, 2),
            "max": round(tot_gross_max_usd * usd_inr, 2)
        },
        "extraction_cost_usd": {
            "min": round(proc_cost_min_usd, 2),
            "avg": round(proc_cost_avg_usd, 2),
            "max": round(proc_cost_max_usd, 2)
        },
        "extraction_cost_inr": {
            "min": round(proc_cost_min_usd * usd_inr, 2),
            "avg": round(proc_cost_avg_usd * usd_inr, 2),
            "max": round(proc_cost_max_usd * usd_inr, 2)
        },
        "net_profit_usd": {
            "min": round(net_profit_min_usd, 2),
            "avg": round(net_profit_avg_usd, 2),
            "max": round(net_profit_max_usd, 2)
        },
        "net_profit_inr": {
            "min": round(net_profit_min_usd * usd_inr, 2),
            "avg": round(net_profit_avg_usd * usd_inr, 2),
            "max": round(net_profit_max_usd * usd_inr, 2)
        },
        "net_profit_margin_percent": {
            "min": round(margin_min, 1),
            "avg": round(margin_avg, 1),
            "max": round(margin_max, 1)
        },
        "metals": metals_summary,
        "component_breakdown": component_breakdown,
        "reagents_stoichiometry": {
            "nitric_acid_68pct_ml_range": {
                "min": round(hno3_min_ml, 1),
                "avg": round(hno3_avg_ml, 1),
                "max": round(hno3_max_ml, 1)
            },
            "aqua_regia_ml_range": {
                "min": round(ar_min_ml, 1),
                "avg": round(ar_avg_ml, 1),
                "max": round(ar_max_ml, 1)
            },
            "na2s2o5_precipitant_g_avg": round(na2s2o5_avg_g, 3)
        },
        # Legacy fields for backward compatibility
        "total_components": total_comp_count,
        "total_component_mass_g": round(tot_mass_avg_g, 2),
        "total_gross_value_usd": round(tot_gross_avg_usd, 2),
        "total_net_value_usd": round(net_profit_avg_usd, 2),
        "recoverable_elements_mg": {k: v["recoverable_mass_mg_range"]["avg"] for k, v in metals_summary.items()},
        "recoverable_elements_g": {k: v["recoverable_mass_g_range"]["avg"] for k, v in metals_summary.items()},
        "net_recoverable_value_usd": {k: v["gross_revenue_usd_range"]["avg"] for k, v in metals_summary.items()}
    }
    return report


# ==============================================================================
# 5. END-TO-END INFERENCE: IMAGE -> YOLO -> METALLURGY VALUATION
# ==============================================================================

def analyze_pcb_image(
    image_path: str,
    model_path: str = "models/pcb_best_model.pt",
    conf: float = 0.25,
    use_live_prices: bool = True,
    api_key: str = None
) -> dict:
    """
    Runs YOLO detection on a PCB image and executes the range-based metallurgical
    assay and profit calculation engine.
    """
    from ultralytics import YOLO

    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found at {image_path}")

    # Fallback to yolov8s.pt or base if model_path does not exist
    if not os.path.exists(model_path):
        model_path = "yolov8s.pt"

    model = YOLO(model_path)
    results = model.predict(source=image_path, conf=conf, verbose=False)
    result = results[0]

    counts = Counter()
    for box in result.boxes:
        cls_id = int(box.cls[0].item())
        cls_name = model.names[cls_id]
        counts[cls_name] += 1

    report = calculate_metallurgical_yield(
        counts,
        use_live_prices=use_live_prices,
        api_key=api_key
    )
    report["image_name"] = os.path.basename(image_path)
    report["detected_counts"] = dict(counts)
    return report


# ==============================================================================
# CLI TEST & DEMONSTRATION
# ==============================================================================

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="PCB Metallurgical Characterization & Profit Range Engine")
    parser.add_argument("--image", type=str, default="", help="Path to PCB image")
    parser.add_argument("--model", type=str, default="models/pcb_best_model.pt", help="Path to YOLO weights")
    parser.add_argument("--no-live", action="store_true", help="Disable live price fetching, use offline benchmark")
    parser.add_argument("--gold-key", type=str, default="", help="Optional GoldAPI.io free API key")
    args = parser.parse_args()

    sample_counts = {
        "IC": 12,
        "capacitor": 48,
        "connector": 4,
        "transformer": 2,
        "transistor": 6,
        "resistor": 35,
        "power capacitor": 3,
        "inductor": 2,
        "mov": 2
    }
    
    print("=" * 72)
    print("⚗️  PCB METALLURGICAL ASSAY & PROFIT RANGE VALUATION ENGINE")
    print("    Department of Metallurgical Engineering, IIT (BHU) Varanasi")
    print("=" * 72)
    
    rep = calculate_metallurgical_yield(
        sample_counts,
        use_live_prices=not args.no_live,
        api_key=args.gold_key
    )
    
    sm = rep["summary"]
    print(f"📊 Market Data Feed  : {sm['market_feed_status']} ({sm['market_feed_source']})")
    print(f"⏰ Timestamp         : {sm['market_feed_timestamp']}")
    print(f"💱 Currency Exchange : 1 USD = ₹{sm['usd_to_inr_rate']:.2f} INR")
    print(f"📦 Total Components  : {sm['total_detected_components']}")
    print(f"⚖️ Component Mass    : {sm['component_mass_g_range']['min']}g (Min) | {sm['component_mass_g_range']['avg']}g (Avg) | {sm['component_mass_g_range']['max']}g (Max)\n")
    
    print("─" * 72)
    print("💎 RECOVERABLE PRECIOUS & BASE METALS (₹ INR RANGE ESTIMATES):")
    print(f"{'Metal':<16} {'Mass Range (mg/g)':<28} {'Revenue Range (INR)':<24}")
    print("─" * 72)
    for m, d in rep["metals"].items():
        mass_str = f"{d['recoverable_mass_mg_range']['min']} - {d['recoverable_mass_mg_range']['avg']} - {d['recoverable_mass_mg_range']['max']} mg"
        val_str = f"₹{d['gross_revenue_inr_range']['min']:,.2f} - ₹{d['gross_revenue_inr_range']['avg']:,.2f} - ₹{d['gross_revenue_inr_range']['max']:,.2f}"
        print(f"• {m:<14}: {mass_str:<28} -> {val_str}")
    
    print("─" * 72)
    gv_i = rep["gross_valuation_inr"]
    ec_i = rep["extraction_cost_inr"]
    np_i = rep["net_profit_inr"]
    pm = rep["net_profit_margin_percent"]

    print("💰 FINANCIAL SUMMARY PER PCB (RUPEES ONLY):")
    print(f"   • Gross Metal Value : ₹{gv_i['min']:,.2f} – ₹{gv_i['avg']:,.2f} – ₹{gv_i['max']:,.2f}")
    print(f"   • Extraction Cost   : ₹{ec_i['min']:,.2f} – ₹{ec_i['avg']:,.2f} – ₹{ec_i['max']:,.2f}  (Chemicals, energy & labor)")
    print(f"   --------------------------------------------------------------------")
    print(f"   🚀 NET PROFIT RANGE : ₹{np_i['min']:,.2f} – ₹{np_i['avg']:,.2f} – ₹{np_i['max']:,.2f} INR / Board")
    print(f"   📈 Net Profit Margin: {pm['min']}% – {pm['avg']}% – {pm['max']}%")
    print("─" * 72)

    # Example: IC Gold breakdown
    print("\n🔍 SPOTLIGHT: GOLD (Au) IN INTEGRATED CIRCUITS (ICs):")
    ic_prof = get_component_range_profile("IC", "Au", count=sample_counts["IC"])
    au_prof = ic_prof["elements"]["Au"]
    print(f"   • Gold in 1 IC      : {au_prof['unit_mass_mg']['min']} mg (Min) | {au_prof['unit_mass_mg']['avg']} mg (Avg) | {au_prof['unit_mass_mg']['max']} mg (Max)")
    print(f"   • Gold in {sample_counts['IC']} ICs     : {au_prof['total_mass_mg']['min']} mg (Min) | {au_prof['total_mass_mg']['avg']} mg (Avg) | {au_prof['total_mass_mg']['max']} mg (Max)")
    print(f"   • Net Monetary Yield: ₹{au_prof['recoverable_value_inr']['min']:,.2f} – ₹{au_prof['recoverable_value_inr']['avg']:,.2f} – ₹{au_prof['recoverable_value_inr']['max']:,.2f}")
    print("=" * 72)
