# Scientific Experiment Report

## Abstract

This document describes the results of environmental monitoring conducted
in Laboratory A during December 2025. Measurements were captured using
calibrated digital sensors with ±0.1°C accuracy.

## Measurement Data

The following image shows the measurement dashboard:

![Scientific Measurement Dashboard](measurement.png)

The dashboard indicates an alert condition due to temperature deviation.

## Environmental Readings

| Parameter | Value | Unit | Normal Range | Status |
|-----------|-------|------|-------------|--------|
| Temperature | 23.5 | °C | 20-25 | Normal |
| Pressure | 1013.25 | hPa | 1000-1030 | Normal |
| Humidity | 45 | % | 40-60 | Normal |
| CO2 | 412 | ppm | <1000 | Normal |
| Particulates | 15 | µg/m³ | <50 | Normal |

## Heat Transfer Model

The steady-state heat transfer in the lab environment follows Fourier's law:

$$q = -k \nabla T = -k \frac{\partial T}{\partial x}$$

where:
- $q$ is the heat flux (W/m²)
- $k$ is the thermal conductivity (W/m·K)
- $T$ is the temperature (K)

For a multi-layer wall, the overall heat transfer coefficient is:

$$U = \frac{1}{\frac{1}{h_1} + \sum_{i=1}^{n} \frac{L_i}{k_i} + \frac{1}{h_2}}$$

## Statistical Analysis

The temperature variance over the measurement period:

$$\sigma^2 = \frac{1}{N-1} \sum_{i=1}^{N} (T_i - \bar{T})^2 = 0.42 \text{ °C}^2$$

This yields a standard deviation of $\sigma = 0.65$ °C, which is within
acceptable limits for the laboratory classification.

## Conclusion

All environmental parameters remain within normal operating ranges.
The alert condition shown in the dashboard was triggered by a brief
temperature spike that self-corrected within 5 minutes.
