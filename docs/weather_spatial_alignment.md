# Phase 1.4: Weather Spatial Alignment and Grid Centroid Report

**Generated:** 2026-11-02 11:43:09
**Source:** ECMWF ERA5-Land Reanalysis via Open-Meteo API
**Grid Resolution:** 0.1° (~9 km spatial resolution)

---

## 1. Study Sites vs. Reanalysis Grid Centroids

| Location | Cohort Association | Study Site Coordinates | ERA5-Land Centroid | Spatial Offset (km) | Alignment Rationale |
|---|---|---|---|---|---|
| **Delhi** | IIIT Delhi | `Okhla Phase III, New Delhi (28.5439 N, 77.2724 E)` | Lat `28.5764`, Lon `77.2893` | **3.96 km** | Closest 0.1° grid cell centroid; within standard microclimate homogeneity boundary (<10 km). |
| **Mathura** | CEEW Mathura | `Mathura District, UP (27.4924 N, 77.6737 E)` | Lat `27.522`, Lon `77.7044` | **4.46 km** | Closest 0.1° grid cell centroid; within standard microclimate homogeneity boundary (<10 km). |
| **Bareilly** | CEEW Bareilly | `Bareilly District, UP (28.3670 N, 79.4304 E)` | Lat `28.3656`, Lon `79.4779` | **4.65 km** | Closest 0.1° grid cell centroid; within standard microclimate homogeneity boundary (<10 km). |

---

## 2. Scientific Spatial Offset Notice

> [!IMPORTANT]
> ERA5-Land does not sample directly at the rooftop of IIIT Delhi or the individual rural households of Mathura and Bareilly. The data represents average atmospheric state over the 0.1° × 0.1° grid cell surrounding each centroid.
> Macro-meteorological variables (surface pressure, temperature, relative humidity, solar radiation) exhibit spatial correlation lengths >> 10 km, making the 3.96–4.65 km offsets scientifically robust for 15-minute electricity demand forecasting.
