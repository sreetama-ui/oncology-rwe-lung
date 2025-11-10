install.packages(c("shiny", "survival", "survminer", "dplyr", "readr"))
# app.R

library(shiny)
library(survival)
library(survminer)
library(dplyr)
library(readr)
library(ggplot2)

# Load dataset
df <- read_csv("lung_survival.csv") %>%
  mutate(status = ifelse(status == 1, 1, 0))  # ensure binary event

ui <- fluidPage(
  
  titlePanel("Lung Cancer Survival Dashboard"),
  
  sidebarLayout(
    sidebarPanel(
      checkboxGroupInput("stage_sel", "Select Stage:",
                         choices = unique(df$stage),
                         selected = unique(df$stage)),
      br(),
      checkboxInput("show_data", "Show Dataset", FALSE)
    ),
    
    mainPanel(
      tabsetPanel(
        tabPanel("Kaplan-Meier Plot", plotOutput("kmPlot")),
        tabPanel("Cox Model Results",
                 tableOutput("coxTable")),
        tabPanel("Data", tableOutput("dataView"))
      )
    )
  )
)

server <- function(input, output, session) {
  
  filtered_data <- reactive({
    df %>% filter(stage %in% input$stage_sel)
  })
  
  output$kmPlot <- renderPlot({
    req(nrow(filtered_data()) > 0)
    
    fit <- survfit(Surv(time, status) ~ stage, data = filtered_data())
    
    ggsurvplot(
      fit,
      data = filtered_data(),
      pval = TRUE,
      risk.table = TRUE,
      ggtheme = theme_minimal(),
      title = "Kaplan-Meier Survival by Stage"
    )
  })
  
  output$coxTable <- renderTable({
    req(nrow(filtered_data()) > 0)
    
    cox_fit <- coxph(Surv(time, status) ~ stage, data = filtered_data())
    summary(cox_fit)$coefficients %>%
      as.data.frame() %>%
      mutate(HR = exp(coef))
  }, rownames = TRUE)
  
  output$dataView <- renderTable({
    if(input$show_data) filtered_data()
  })
}

shinyApp(ui, server)

rsconnect::setAccountInfo(name='sreetama06', 
                          token='A89E672A3D2A1BB2FF7563A5E9225326', 
                          secret='e8Uim8NqNqtnTVVo8lYTRfYD0+NF3jGMqKVBwQK7')
rsconnect::deployApp("C:/Users/sreet/OneDrive/Desktop/oncology-rwe-lung/data")
