## Mental Health Multi-CSV Dashboard
## Dynamic Shiny app that loads every CSV in the app folder and provides EDA + modelling

library(shiny)
library(tidyverse)
library(caret)
library(randomForest)
library(plotly)
library(DT)
library(pROC)
library(recipes)    # tidy preprocessing pipelines (imputation, scaling, dummies, PCA)
library(rlang)      # for tidy programming helpers used in some recipe steps

options(shiny.maxRequestSize = 30*1024^2)

ui <- fluidPage(
  titlePanel("🧠 Multi-CSV Mental Health Dashboard"),

  sidebarLayout(
    sidebarPanel(
      helpText("Select a dataset (CSV) from the project folder."),
      uiOutput("dataset_ui"),
      tags$hr(),
      h5("Preprocessing (learning options)"),
      helpText("These demonstrate common preprocessing steps: imputation, scaling, dummy encoding and PCA. Use them to learn how preprocessing affects results."),
      checkboxGroupInput("preproc_steps", "Preprocessing steps:",
                         choices = list("Impute median (numeric)" = "impute_median",
                                        "Center & scale (numeric)" = "scale",
                                        "One-hot encode factors" = "dummies",
                                        "PCA on numeric predictors" = "pca"),
                         selected = c()),
      conditionalPanel("input.preproc_steps.includes('pca')",
                       numericInput("pca_comp", "PCA components:", value = 2, min = 1, step = 1)),
      hr(),
      h4("Modeling (if a binary target exists)"),
      uiOutput("target_ui"),
      uiOutput("modelMethod_ui"),
      checkboxInput("compare_models", "Train and compare both GLM and RF (educational)", value = FALSE),
      hr(),
      actionButton("train", "Train Model"),
      hr(),
      conditionalPanel("output.model_trained == true",
                       h4("Make a prediction"),
                       uiOutput("predict_inputs_ui"),
                       actionButton("predict_btn", "Predict"),
                       downloadButton("downloadData", "💾 Download Prediction as CSV")
      ),
      width = 3
    ),

    mainPanel(
      tabsetPanel(
        tabPanel("📄 Data Preview", DTOutput("data_table")),
        tabPanel("📊 EDA",
                 plotlyOutput("ageDist"),
                 plotlyOutput("corrPlot"),
                 plotlyOutput("catVsTarget"),
                 plotlyOutput("boxByTarget"),
                 plotlyOutput("densityByTarget")
        ),
        tabPanel("🧩 Model Output",
                 verbatimTextOutput("model_summary"),
                 verbatimTextOutput("conf_matrix"),
                 plotOutput("featImportance"),
                 plotlyOutput("rocPlot")
                 verbatimTextOutput("preproc_summary"),
                 downloadButton("downloadModel", "💾 Download Trained Model (RDS)")
        )
      ),
      width = 9
    )
  )
)

server <- function(input, output, session) {
  # discover csv files in the app directory
  csv_files <- reactive({
    files <- list.files(path = ".", pattern = "\\.csv$", full.names = FALSE)
    files
  })

  output$dataset_ui <- renderUI({
    files <- csv_files()
    if(length(files) == 0) return(div("No CSV files found in project folder."))
    selectInput("dataset", "Dataset:", choices = files)
  })

  # load the selected dataset
  raw_data <- reactive({
    req(input$dataset)
    df <- tryCatch({
      read.csv(input$dataset, stringsAsFactors = FALSE)
    }, error = function(e) {
      NULL
    })
    df
  })

  # quick cleaning: convert character to factor where appropriate
  data <- reactive({
    df <- raw_data()
    req(df)
    # convert character columns with limited unique values to factors
    df <- df %>% mutate(across(where(is.character), ~ if(n_distinct(.) < (nrow(df) * 0.5)) as.factor(.) else as.character(.)))
    # also turn logical to factor
    df <- df %>% mutate(across(where(is.logical), as.factor))
    df
  })

  output$data_table <- renderDT({
    req(data())
    datatable(head(data(), 50), options = list(scrollX = TRUE))
  })

  # detect binary targets (factors with 2 levels)
  binary_targets <- reactive({
    df <- data()
    req(df)
    cols <- names(df)[sapply(df, function(x) is.factor(x) && length(levels(x)) == 2)]
    cols
  })

  output$target_ui <- renderUI({
    bt <- binary_targets()
    if(length(bt) == 0) return(div("No binary target detected. Only EDA available."))
    selectInput("target", "Binary target (select):", choices = bt)
  })

  output$modelMethod_ui <- renderUI({
    req(binary_targets())
    radioButtons("method", "Method:", choices = c("glm" = "glm", "randomForest" = "rf"), selected = "glm")
  })

  # reactive for model and artifacts
  model_store <- reactiveValues(model = NULL, importance = NULL, conf = NULL, roc = NULL, trained = FALSE, train_data = NULL)

  observeEvent(input$train, {
    req(input$dataset)
    req(input$target)
    df <- data()
    # prepare data: keep target + predictors and drop rows with target NA only; let recipe handle predictor NA if chosen
    df2 <- df %>% select(all_of(c(input$target, names(df)[names(df) != input$target])))
    # ensure target is factor with 2 levels
    df2[[input$target]] <- as.factor(df2[[input$target]])
    # ensure character predictors become factors for modeling where helpful
    df2[] <- lapply(df2, function(x) if(is.character(x)) as.factor(x) else x)

    # --- Build a recipe based on user-chosen preprocessing steps (teaches tidy preprocessing pipelines) ---
    rec <- recipe(as.formula(paste(input$target, "~ .")), data = df2)
    # optional median imputation for numeric predictors
    if("impute_median" %in% input$preproc_steps){
      rec <- rec %>% step_impute_median(all_numeric_predictors())
    }
    # optional center & scale
    if("scale" %in% input$preproc_steps){
      rec <- rec %>% step_normalize(all_numeric_predictors())
    }
    # optional dummy encoding for factors
    if("dummies" %in% input$preproc_steps){
      rec <- rec %>% step_dummy(all_nominal_predictors(), one_hot = TRUE)
    }
    # optional PCA (applied after normalization/dummies where numeric predictors exist)
    if("pca" %in% input$preproc_steps){
      k <- max(1, as.integer(input$pca_comp))
      rec <- rec %>% step_pca(all_numeric_predictors(), num_comp = k)
    }
    # prep the recipe using the full dataset (recipes will estimate needed statistics)
    rec_prep <- tryCatch({prep(rec, training = df2)}, error = function(e){
      rec
    })
    df_trans <- tryCatch({
      bake(rec_prep, new_data = df2)
    }, error = function(e){
      # if bake fails, fallback to df2 and drop rows with NA in target
      df2
    })
    # remove rows with NA target
    df_trans <- df_trans %>% filter(!is.na(.data[[input$target]]))

  set.seed(123)
  trainIndex <- createDataPartition(df_trans[[input$target]], p = .8, list = FALSE)
  train <- df_trans[trainIndex, ]
  test <- df_trans[-trainIndex, ]

    control <- trainControl(method = "cv", number = 5, classProbs = TRUE, summaryFunction = twoClassSummary, savePredictions = TRUE)

    # caret expects the positive class to be the first level in twoClassSummary; enforce levels
    # Ensure levels are named "Class1"/"Class2"? We'll use existing factor levels but ensure caret sees them.
    model_formula <- as.formula(paste(input$target, "~ ."))

    # train based on chosen method; optionally compare both models for learning
    models_trained <- list()
    withProgress(message = 'Training model(s)...', value = 0, {
      if(input$compare_models){
        incProgress(0.2, detail = 'Training GLM')
        m_glm <- tryCatch({train(model_formula, data = train, method = "glm", family = "binomial", trControl = control, metric = "ROC")}, error = function(e) NULL)
        incProgress(0.5, detail = 'Training RF')
        m_rf <- tryCatch({train(model_formula, data = train, method = "rf", trControl = control, metric = "ROC", importance = TRUE)}, error = function(e) NULL)
        models_trained$glm <- m_glm
        models_trained$rf <- m_rf
        model <- if(!is.null(m_rf)) m_rf else m_glm
      } else {
        if(input$method == "glm"){
          model <- train(model_formula, data = train, method = "glm", family = "binomial", trControl = control, metric = "ROC")
        } else {
          model <- train(model_formula, data = train, method = "rf", trControl = control, metric = "ROC", importance = TRUE)
        }
        models_trained[[input$method]] <- model
      }
    })

    # predictions
  pred <- predict(model, newdata = test)
  prob <- tryCatch({predict(model, newdata = test, type = "prob")}, error = function(e) NULL)

    # confusion
    cm <- confusionMatrix(pred, test[[input$target]])

    # ROC
    roc_obj <- NULL
    if(!is.null(prob)){
      # pick second column as positive probability (caret returns columns named after factor levels)
      pos_col <- colnames(prob)[2]
      roc_obj <- roc(response = test[[input$target]], predictor = prob[[pos_col]])
    }

    # importance
    imp <- tryCatch({varImp(model)}, error = function(e) NULL)

  model_store$model <- model
    model_store$importance <- imp
    model_store$conf <- cm
    model_store$roc <- roc_obj
    model_store$trained <- TRUE
    model_store$train_data <- train
  model_store$recipe <- rec_prep
  model_store$all_models <- models_trained

    # expose to UI
    output$model_trained <- reactive({model_store$trained})
    outputOptions(output, "model_trained", suspendWhenHidden = FALSE)
  })

  output$model_summary <- renderPrint({
    req(model_store$trained)
    # Show the primary model and (if present) comparison results
    if(!is.null(model_store$all_models) && length(model_store$all_models) > 1){
      cat("Trained models (comparison):\n")
      lapply(names(model_store$all_models), function(nm){
        cat(paste0("--- ", nm, " ---\n"))
        print(model_store$all_models[[nm]])
        cat('\n')
      })
    } else {
      print(model_store$model)
    }
  })

  output$conf_matrix <- renderPrint({
    req(model_store$trained)
    print(model_store$conf)
  })

  output$featImportance <- renderPlot({
    req(model_store$trained)
    if(!is.null(model_store$importance)){
      plot(model_store$importance, main = "Feature Importance")
    }
  })

  output$rocPlot <- renderPlotly({
    req(model_store$trained)
    roc_obj <- model_store$roc
    if(is.null(roc_obj)) return(NULL)
    df_roc <- data.frame(tpr = rev(roc_obj$sensitivities), fpr = rev(1 - roc_obj$specificities))
    p <- ggplot(df_roc, aes(x = fpr, y = tpr)) + geom_line() + geom_abline(linetype = "dashed") + labs(title = paste0('ROC AUC = ', round(auc(roc_obj), 3)))
    ggplotly(p)
  })

  # EDA plots
  output$ageDist <- renderPlotly({
    df <- data()
    req(df)
    # pick a numeric column if exists named age, otherwise first numeric
    if("age" %in% names(df) && is.numeric(df$age)){
      p <- ggplot(df, aes(x = age)) + geom_histogram(bins = 20, fill = "steelblue") + labs(title = "Age distribution")
      ggplotly(p)
    } else {
      nums <- df %>% select(where(is.numeric))
      if(ncol(nums) == 0) return(NULL)
      coln <- names(nums)[1]
      p <- ggplot(df, aes_string(x = coln)) + geom_histogram(bins = 20, fill = "steelblue") + labs(title = paste0(coln, " distribution"))
      ggplotly(p)
    }
  })

  output$corrPlot <- renderPlotly({
    df <- data()
    req(df)
    nums <- df %>% select(where(is.numeric))
    if(ncol(nums) < 2) return(NULL)
    corr <- round(cor(nums, use = "pairwise.complete.obs"), 2)
    plot_ly(z = corr, x = colnames(corr), y = rownames(corr), type = "heatmap", colorscale = "Viridis") %>% layout(title = "Correlation heatmap")
  })

  output$catVsTarget <- renderPlotly({
    df <- data()
    req(df)
    if(is.null(input$target)) return(NULL)
    # find a categorical variable other than target
    cats <- names(df)[sapply(df, is.factor) & names(df) != input$target]
    if(length(cats) == 0) return(NULL)
    var <- cats[1]
    p <- ggplot(df, aes_string(x = var, fill = input$target)) + geom_bar(position = "fill") + labs(y = "Proportion", title = paste0(var, " vs ", input$target))
    ggplotly(p)
  })

  # Boxplot of first numeric predictor grouped by target (learning: distribution comparisons)
  output$boxByTarget <- renderPlotly({
    df <- data()
    req(df)
    if(is.null(input$target)) return(NULL)
    nums <- df %>% select(where(is.numeric))
    if(ncol(nums) == 0) return(NULL)
    var <- names(nums)[1]
    p <- ggplot(df, aes_string(x = input$target, y = var, fill = input$target)) + geom_boxplot() + labs(title = paste0('Boxplot of ', var, ' by ', input$target))
    ggplotly(p)
  })

  # Density plot of first numeric predictor by target
  output$densityByTarget <- renderPlotly({
    df <- data()
    req(df)
    if(is.null(input$target)) return(NULL)
    nums <- df %>% select(where(is.numeric))
    if(ncol(nums) == 0) return(NULL)
    var <- names(nums)[1]
    p <- ggplot(df, aes_string(x = var, color = input$target, fill = input$target)) + geom_density(alpha = 0.3) + labs(title = paste0('Density of ', var, ' by ', input$target))
    ggplotly(p)
  })

  # dynamic prediction input UI generated from predictors of trained model
  output$predict_inputs_ui <- renderUI({
    req(model_store$trained)
    model <- model_store$model
    train_df <- model_store$train_data
    preds <- predictors(model)
    if(length(preds) == 0) return(div("No predictors found."))
    inputs <- map(preds, function(var){
      if(is.factor(train_df[[var]])){
        selectInput(paste0("pred_", var), var, choices = levels(train_df[[var]]))
      } else if(is.numeric(train_df[[var]])){
        numericInput(paste0("pred_", var), var, value = round(mean(train_df[[var]], na.rm = TRUE), 2))
      } else {
        textInput(paste0("pred_", var), var, value = as.character(train_df[[var]][1]))
      }
    })
    do.call(tagList, inputs)
  })

  # prediction action
  observeEvent(input$predict_btn, {
    req(model_store$trained)
    model <- model_store$model
    preds <- predictors(model)
    newrow <- map(preds, function(var){
      value <- input[[paste0("pred_", var)]]
      # coerce to same type as training
      train_df <- model_store$train_data
      if(is.factor(train_df[[var]])) return(factor(value, levels = levels(train_df[[var]])))
      if(is.numeric(train_df[[var]])) return(as.numeric(value))
      return(value)
    }) %>% set_names(preds) %>% as.data.frame()

    # If a recipe was used, apply the same preprocessing to the new observation before predicting
    if(!is.null(model_store$recipe)){
      newrow_trans <- tryCatch({
        bake(model_store$recipe, new_data = newrow)
      }, error = function(e) {
        newrow
      })
    } else {
      newrow_trans <- newrow
    }

    pred_class <- predict(model, newdata = newrow)
    prob <- tryCatch({predict(model, newdata = newrow, type = "prob")}, error = function(e) NULL)

    output$prediction_result <- renderPrint({
      cat("Predicted class:", as.character(pred_class), "\n")
      if(!is.null(prob)){
        print(prob)
      }
    })

    output$downloadData <- downloadHandler(
      filename = function() { paste0("prediction_", tools::file_path_sans_ext(input$dataset), ".csv") },
      content = function(file){
        out <- bind_cols(newrow, Prediction = as.character(pred_class))
        if(!is.null(prob)) out <- bind_cols(out, prob)
        write.csv(out, file, row.names = FALSE)
      }
    )
  })

  # allow user to download the trained primary model as an RDS
  output$downloadModel <- downloadHandler(
    filename = function(){ paste0('trained_model_', tools::file_path_sans_ext(input$dataset), '.rds') },
    content = function(file){
      req(model_store$trained)
      saveRDS(list(model = model_store$model, recipe = model_store$recipe), file)
    }
  )

  # expose prediction result output so it shows in UI
  outputOptions(output, "prediction_result", suspendWhenHidden = FALSE)

}

shinyApp(ui, server)
