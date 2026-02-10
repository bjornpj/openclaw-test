# W204 Diagnostic Reference

## 1) Intake questions (always gather first)

- VIN or exact model/year (e.g., 2011 C300 4MATIC)
- Engine family if known (M271, M272, M274, M276, M156)
- Mileage and maintenance history
- Fuel type and quality
- Exact symptom timing: cold start, hot idle, acceleration, braking, bumps, highway only
- Any fault codes (P-codes + manufacturer-specific)

## 2) Symptom-to-cause map

### A. Rough idle / check engine / misfire
- Common suspects:
  - Ignition coils/plugs
  - Vacuum leak/PCV leak
  - Injector imbalance
  - Timing/cam correlation issues
- First checks:
  - Scan codes + freeze frame
  - Misfire counters by cylinder
  - Smoke test intake
  - Plug/coil swap test

### B. Rattle on startup / timing faults
- Common suspects:
  - Timing chain stretch / tensioner wear (engine dependent)
- First checks:
  - Cold-start noise recording
  - Cam/crank correlation data
  - Check service bulletins by VIN

### C. Overheating or coolant loss
- Common suspects:
  - Thermostat sticking
  - Water pump leak
  - Expansion tank/cap/hoses
  - Radiator seep
- First checks:
  - Pressure test cooling system
  - Inspect for dried coolant traces
  - Verify fan operation

### D. Harsh shifting / limp mode (7G-Tronic)
- Common suspects:
  - Fluid condition/level issue
  - Valve body / conductor plate faults
  - Adaptation values out of range
- First checks:
  - Transmission module scan
  - Fluid inspection at correct temp procedure
  - Adaptation read/reset only after mechanical baseline verified

### E. Clunks over bumps / uneven tire wear
- Common suspects:
  - Control arm bushings / ball joints
  - Sway bar links
  - Shock mounts
  - Alignment drift from worn arms
- First checks:
  - Pry-bar play test on lift
  - Tire wear pattern inspection
  - Alignment check after component replacement

### F. Random electrical warnings
- Common suspects:
  - Weak battery or charging instability
  - Ground faults / moisture in connectors
  - Steering lock / ESL-related faults
- First checks:
  - Battery test + charging voltage under load
  - Ground strap inspection
  - Module scan for communication faults

## 3) Service planning guidance

- Prefer staged repair plan:
  1. Safety-critical first
  2. Root-cause diagnostics
  3. Reliability fixes
  4. Comfort/cosmetic issues
- After major work, recommend:
  - Fault memory clear and drive cycle
  - Re-scan for returning or pending codes
  - Leak re-check after heat soak

## 4) Cost band template (use ranges)

Use broad ranges unless user provides region/shop rates:
- Low: <$250
- Medium: $250-$900
- High: $900-$2500+

Always explain uncertainty drivers: labor rate, OEM vs aftermarket, corrosion, seized hardware, hidden damage.

## 5) Safety escalation triggers

Recommend tow / no driving if any:
- Active overheating or coolant temp spikes
- Oil pressure warning
- Severe engine knock/misfire with flashing MIL
- Brake pedal sinking or major brake fluid leak
- Steering assist loss with abnormal noise
