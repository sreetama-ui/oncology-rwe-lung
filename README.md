# 🫁 Lung Cancer Survival Dashboard — R Shiny

**Purpose:**  
Interactive survival-analysis dashboard showing Kaplan–Meier and Cox Proportional Hazards results for real-world lung-cancer data.

**Tech stack:**  
R | Shiny | survival | survminer | dplyr | ggplot2 | readr

---

## 🚀 Run locally
```r
# Install required packages once
install.packages(c("shiny","survival","survminer","dplyr","readr","ggplot2"))

# Then launch directly from GitHub
shiny::runGitHub("sreetama06/oncology-rwe-lung")
