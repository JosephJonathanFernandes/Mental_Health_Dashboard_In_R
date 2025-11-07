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
library(bslib)

options(shiny.maxRequestSize = 30*1024^2)

ui <- fluidPage(
  theme = bs_theme(bootswatch = "flatly", primary = "#2c7fb8"),
  tags$head(
    tags$style(HTML(".sidebar-badge { margin-right: 6px; } .control-card { padding: 10px; margin-bottom: 12px; border-radius:6px; box-shadow: 0 1px 2px rgba(0,0,0,0.05);} .muted-help{color:#6c757d;font-size:0.9em;}"))
  ),
  titlePanel("🧠 Multi-CSV Mental Health Dashboard"),

  sidebarLayout(
  sidebarPanel(
    # top badges and quick actions
    div(class = 'control-card',
      uiOutput("dataset_badges"),
      div(class = 'muted-help', "Choose a CSV, explore with EDA, then train if a binary target is available." )
    ),

    # Dataset selection
    div(class = 'control-card',
      tags$strong("Dataset"),
      br(),
      uiOutput("dataset_ui"),
      tags$hr(),
      uiOutput("binary_datasets_ui"),
      uiOutput("status_panel")
    ),

    # EDA controls
    div(class = 'control-card',
      tags$strong("EDA Controls"),
      uiOutput("eda_controls")
    ),

    # Preprocessing options
    div(class = 'control-card',
      tags$strong("Preprocessing (learning options)"),
      tags$div(class='muted-help', "Imputation, scaling, dummies and PCA — try combinations to see effects."),
      checkboxGroupInput("preproc_steps", "Preprocessing steps:",
               choices = list("Impute median (numeric)" = "impute_median",
                      "Center & scale (numeric)" = "scale",
                      "One-hot encode factors" = "dummies",
                      "PCA on numeric predictors" = "pca"),
               selected = c()),
      conditionalPanel("input.preproc_steps.includes('pca')",
               numericInput("pca_comp", "PCA components:", value = 2, min = 1, step = 1))
    ),

    # Modeling and sampling
    div(class = 'control-card',
      tags$strong("Modeling"),
      uiOutput("target_ui"),
      uiOutput("modelMethod_ui"),
      tags$hr(),
      tags$strong("Sampling"),
      tags$div(class='muted-help', "Limit rows for faster experiments."),
      checkboxInput("use_sampling", "Use sampling for training", value = FALSE),
      conditionalPanel("input.use_sampling == true",
               numericInput("sample_rows", "Rows to sample (approx):", value = 1000, min = 100, step = 100)
      ),
      checkboxInput("compare_models", "Train and compare both GLM and RF (educational)", value = FALSE),
      hr(),
      actionButton("train", "Train Model", class = "btn-primary")
    ),

    # prediction controls (hidden until model trained)
    conditionalPanel("output.model_trained == true",
             div(class = 'control-card',
               h4("Make a prediction"),
               uiOutput("predict_inputs_ui"),
               actionButton("predict_btn", "Predict", class = "btn-success"),
               downloadButton("downloadData", "💾 Download Prediction as CSV", class = "btn-sm")
             )
    ),
    width = 3
  ),

    mainPanel(
      tabsetPanel(
        tabPanel("📄 Data Preview", DTOutput("data_table")),
        tabPanel("📊 EDA",
                 # For most interactive plots we use plotly; Pair Plot uses a static GGally ggpairs when available
                 conditionalPanel("input.plot_type != 'Pair Plot'", plotlyOutput("eda_plot", height = "600px")),
                 conditionalPanel("input.plot_type == 'Pair Plot'", plotOutput("pairPlotStatic", height = "800px")),
                 conditionalPanel("input.plot_type == 'Pair Plot'", downloadButton("downloadPair", "Download Pair Plot (PNG)")),
                 tags$hr(),
                 fluidRow(
                   column(6, plotlyOutput("corrPlot")),
                   column(6, plotlyOutput("catVsTarget"))
                 ),
                 fluidRow(
                   column(6, plotlyOutput("boxByTarget")),
                   column(6, plotlyOutput("densityByTarget"))
                 )
        ),
  tabPanel("🧩 Model Output",
     verbatimTextOutput("model_summary"),
     verbatimTextOutput("conf_matrix"),
     verbatimTextOutput("prediction_result"),
    plotOutput("featImportance"),
    downloadButton("downloadFeat", "💾 Download Feature Importance (PNG)"),
     plotlyOutput("rocPlot"),
     verbatimTextOutput("preproc_summary"),
     downloadButton("downloadModel", "💾 Download Trained Model (RDS)")
  )
      ),
      width = 9
    )
  )
)

server <- function(input, output, session) {
  # helper to safely convert ggplot to plotly and avoid crashing the app
  safe_ggplotly <- function(p){
    if(is.null(p)) return(NULL)
    if(inherits(p, "plotly")) return(p)
    tryCatch({
      ggplotly(p)
    }, error = function(e){
      message("Plotly conversion error: ", e$message)
      plot_ly() %>% layout(title = paste0("Plot error: ", substr(e$message,1,200)))
    })
  }
  # discover csv files in the app directory
  csv_files <- reactive({
    files <- list.files(path = ".", pattern = "\\.csv$", full.names = FALSE)
    files
  })

  # which files in the folder contain at least one binary target (factor 2-level or numeric 0/1)
  files_with_binary <- reactive({
    files <- csv_files()
    res <- c()
    for(f in files){
      ok <- tryCatch({
        df <- read.csv(f, stringsAsFactors = FALSE)
        # quick detection: character 2-unique or numeric 0/1
        facs <- names(df)[vapply(df, function(x) {
          (is.factor(x) && length(levels(x)) == 2) || (is.character(x) && length(unique(na.omit(x))) == 2)
        }, logical(1))]
        nums <- names(df)[vapply(df, function(x) {
          is.numeric(x) && length(unique(na.omit(x))) == 2
        }, logical(1))]
        nums_bin <- character(0)
        if(length(nums) > 0){
          nums_keep <- vapply(nums, function(col){
            vals <- unique(na.omit(df[[col]]))
            all(vals %in% c(0,1))
          }, logical(1))
          nums_bin <- nums[nums_keep]
        }
        length(unique(c(facs, nums_bin))) > 0
      }, error = function(e) FALSE)
      if(isTRUE(ok)) res <- c(res, f)
    }
    res
  })

  output$binary_datasets_ui <- renderUI({
    files <- files_with_binary()
    tagList(
      strong("Datasets with detected binary targets:"),
      if(length(files) == 0) div("None found") else div(paste(files, collapse = ", "))
    )
  })

  output$dataset_badges <- renderUI({
    files <- csv_files()
    files_bin <- files_with_binary()
    tagList(
      span(class = 'badge bg-primary sidebar-badge', paste0(length(files), ' CSVs')),
      span(class = 'badge bg-success sidebar-badge', paste0(length(files_bin), ' binary-ready')),
      actionLink('refresh_files', icon('sync'), title = 'Refresh file list')
    )
  })

  observeEvent(input$refresh_files, {
    # trivial trigger to re-evaluate reactive file listings
    csv_files()
    files_with_binary()
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

  # dataset stats and quick info
  output$dataset_stats <- renderUI({
    df <- data()
    req(df)
    nrows <- nrow(df)
    ncols <- ncol(df)
    bins <- paste0(nrows, " rows × ", ncols, " columns")
    bt <- binary_targets()
    tagList(
      strong("Dataset:"), div(input$dataset),
      br(),
      strong("Size:"), div(bins),
      br(),
      strong("Detected binary targets (factor 2-level or numeric 0/1):"),
      if(length(bt) == 0) div("None detected") else div(paste(bt, collapse = ", "))
    )
  })

  # small status panel showing live counts (rows, cols, sampled rows)
  output$status_panel <- renderUI({
    df <- raw_data()
    req(df)
    total_rows <- nrow(df)
    total_cols <- ncol(df)
    sampled <- if(isTRUE(input$use_sampling)) min(as.integer(input$sample_rows), total_rows) else NA
    tagList(
      tags$hr(),
      strong("Status"),
      div(paste0("Rows: ", total_rows)),
      div(paste0("Columns: ", total_cols)),
      if(!is.na(sampled)) div(paste0("Sampled rows (planned): ", sampled)) else div("Sampled rows: not using sampling")
    )
  })

  # helpers: numeric and categorical variable lists for the selected dataset
  numeric_vars <- reactive({
    df <- data()
    req(df)
    names(df)[sapply(df, is.numeric)]
  })

  categorical_vars <- reactive({
    df <- data()
    req(df)
    names(df)[sapply(df, function(x) is.factor(x) || is.character(x))]
  })

  # EDA controls UI
  output$eda_controls <- renderUI({
    nums <- numeric_vars()
    cats <- categorical_vars()
    plot_types <- c("Histogram", "Density", "Boxplot", "Violin", "Bar", "Stacked Bar", "Scatter", "Pair Plot", "Missing Map", "Correlation Heatmap")
    tagList(
      selectInput("plot_type", "Plot type:", choices = plot_types, selected = "Histogram"),
  # plot appearance controls
  conditionalPanel("input.plot_type == 'Histogram' || input.plot_type == 'Density' || input.plot_type == 'Boxplot' || input.plot_type == 'Violin' || input.plot_type == 'Scatter'",
           numericInput("bins", "Bins (for histogram):", value = 30, min = 5, step = 1),
           selectInput("hist_fill", "Histogram / fill color:", choices = c("steelblue", "salmon", "darkgreen", "purple", "grey"), selected = "steelblue"),
           sliderInput("alpha", "Alpha / transparency:", min = 0.1, max = 1, value = 0.8, step = 0.05),
           sliderInput("point_size", "Point size:", min = 0.5, max = 5, value = 2, step = 0.1),
           checkboxInput("add_jitter", "Add jitter (scatter/box)", value = FALSE),
           conditionalPanel("input.add_jitter == true", sliderInput("jitter_width", "Jitter width:", min = 0, max = 1, value = 0.2, step = 0.01)),
           checkboxInput("add_smooth", "Add smoothing line (scatter)", value = FALSE),
           conditionalPanel("input.add_smooth == true", selectInput("smooth_method", "Smoother:", choices = c("loess", "lm"), selected = "loess")),
           conditionalPanel("input.add_smooth == true && input.smooth_method == 'loess'", sliderInput("smooth_span", "Loess span:", min = 0.1, max = 1, value = 0.75, step = 0.05))
  ),
  selectInput("axis_transform", "Axis transform:", choices = c("none", "log10", "sqrt"), selected = "none"),
      conditionalPanel("input.plot_type == 'Histogram' || input.plot_type == 'Density' || input.plot_type == 'Boxplot' || input.plot_type == 'Violin'",
                       selectInput("eda_x", "Numeric variable:", choices = nums, selected = ifelse(length(nums)>0, nums[1], NA))
      ),
      conditionalPanel("input.plot_type == 'Bar' || input.plot_type == 'Stacked Bar'",
                       selectInput("eda_cat", "Categorical variable:", choices = cats, selected = ifelse(length(cats)>0, cats[1], NA))
      ),
      conditionalPanel("input.plot_type == 'Scatter'",
                       selectInput("eda_x", "X variable:", choices = nums, selected = ifelse(length(nums)>1, nums[1], NA)),
                       selectInput("eda_y", "Y variable:", choices = nums, selected = ifelse(length(nums)>1, nums[2], NA))
      ),
      conditionalPanel("input.plot_type == 'Pair Plot'",
                       helpText("Pair plot uses all numeric variables. If GGally is installed you'll get a richer static ggpairs view.")),
      conditionalPanel("input.plot_type == 'Missing Map'",
                       helpText("Shows missing-value counts per column.")),
      conditionalPanel("input.plot_type == 'Correlation Heatmap'",
                       helpText("Correlation for numeric variables (pairwise complete observations)."))
    )
  })

  # detect binary targets:
  # - factor columns with exactly 2 levels
  # - character columns with exactly 2 unique values (e.g. "Yes"/"No")
  # - numeric columns that contain only 0/1 values (common encoding for binary targets)
  binary_targets <- reactive({
    df <- data()
    req(df)
    # factors or character columns with 2 unique values
    facs <- names(df)[vapply(df, function(x) {
      (is.factor(x) && length(levels(x)) == 2) || (is.character(x) && length(unique(na.omit(x))) == 2)
    }, logical(1))]
    # numeric columns with exactly 2 unique non-NA values
    nums <- names(df)[vapply(df, function(x) {
      is.numeric(x) && length(unique(na.omit(x))) == 2
    }, logical(1))]
    # keep only numeric columns that are truly 0/1
    if(length(nums) > 0){
      nums_keep <- vapply(nums, function(col){
        vals <- unique(na.omit(df[[col]]))
        all(vals %in% c(0,1))
      }, logical(1))
      nums_bin <- nums[nums_keep]
    } else {
      nums_bin <- character(0)
    }
    unique(c(facs, nums_bin))
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

    # If user requested sampling (to limit rows for training), sample here before building recipe
    if(isTRUE(input$use_sampling)){
      n_req <- as.integer(input$sample_rows)
      if(!is.na(n_req) && n_req > 0 && n_req < nrow(df2)){
        set.seed(123)
        df2 <- df2 %>% slice_sample(n = n_req)
      }
    }

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
    # build importance plot as a ggplot object so it can be downloaded
    imp_plot_obj <- local({
      imp_obj <- model_store$importance
      if(is.null(imp_obj)) return(NULL)
      imp_df <- NULL; vals <- NULL; vars <- NULL
      try({
        if(is.list(imp_obj) && !is.null(imp_obj$importance)){
          imp_df <- as.data.frame(imp_obj$importance)
          vars <- rownames(imp_df)
          if("Overall" %in% colnames(imp_df)) vals <- imp_df$Overall else vals <- rowMeans(imp_df, na.rm = TRUE)
        } else if(is.data.frame(imp_obj)){
          imp_df <- imp_obj
          vars <- rownames(imp_df)
          if(ncol(imp_df) == 1) vals <- imp_df[[1]] else vals <- rowMeans(imp_df, na.rm = TRUE)
        } else if(!is.null(model_store$model$finalModel) && "randomForest" %in% class(model_store$model$finalModel)){
          rf_imp <- tryCatch({randomForest::importance(model_store$model$finalModel)}, error = function(e) NULL)
          if(!is.null(rf_imp) && is.matrix(rf_imp)){
            vars <- rownames(rf_imp)
            vals <- if("MeanDecreaseGini" %in% colnames(rf_imp)) rf_imp[, "MeanDecreaseGini"] else rowMeans(rf_imp, na.rm = TRUE)
          }
        }
      })
      if(is.null(vars) || length(vars) == 0) return(NULL)
      df_imp_plot <- data.frame(variable = vars, importance = as.numeric(vals), stringsAsFactors = FALSE)
      df_imp_plot <- df_imp_plot %>% arrange(desc(importance)) %>% mutate(pct = 100 * importance / sum(importance, na.rm = TRUE))
      p <- ggplot(df_imp_plot, aes(x = reorder(variable, pct), y = pct)) +
        geom_col(fill = "#2c7fb8") + coord_flip() +
        labs(x = NULL, y = "Importance (%)", title = "Feature importance (relative %)") +
        geom_text(aes(label = sprintf("%.1f%%", pct)), hjust = -0.1, size = 3) +
        theme_minimal() + theme(plot.margin = margin(5, 40, 5, 5)) +
        scale_y_continuous(expand = expansion(mult = c(0, .15)))
      return(p)
    })
    if(is.null(imp_plot_obj)){
      plot.new(); text(0.5,0.5, "Feature importance could not be extracted")
      return()
    }
    print(imp_plot_obj)
  })

  # download handler for feature importance PNG
  output$downloadFeat <- downloadHandler(
    filename = function(){ paste0('feature_importance_', tools::file_path_sans_ext(input$dataset), '.png') },
    content = function(file){
      # reconstruct the plot object same as above
      imp_obj <- model_store$importance
      if(is.null(imp_obj)){
        png(file); plot.new(); text(0.5,0.5, 'No feature importance'); dev.off(); return()
      }
      # reuse the plotting logic to build p
      p <- tryCatch({
        # build df
        imp_df <- NULL; vals <- NULL; vars <- NULL
        if(is.list(imp_obj) && !is.null(imp_obj$importance)){
          imp_df <- as.data.frame(imp_obj$importance); vars <- rownames(imp_df); vals <- if('Overall' %in% colnames(imp_df)) imp_df$Overall else rowMeans(imp_df, na.rm = TRUE)
        } else if(is.data.frame(imp_obj)){
          imp_df <- imp_obj; vars <- rownames(imp_df); vals <- if(ncol(imp_df)==1) imp_df[[1]] else rowMeans(imp_df, na.rm = TRUE)
        } else if(!is.null(model_store$model$finalModel) && 'randomForest' %in% class(model_store$model$finalModel)){
          rf_imp <- tryCatch({randomForest::importance(model_store$model$finalModel)}, error = function(e) NULL)
          if(!is.null(rf_imp) && is.matrix(rf_imp)){ vars <- rownames(rf_imp); vals <- if('MeanDecreaseGini' %in% colnames(rf_imp)) rf_imp[,'MeanDecreaseGini'] else rowMeans(rf_imp, na.rm = TRUE) }
        }
        if(is.null(vars) || length(vars)==0) stop('no importance')
        df_imp_plot <- data.frame(variable = vars, importance = as.numeric(vals), stringsAsFactors = FALSE)
        df_imp_plot <- df_imp_plot %>% arrange(desc(importance)) %>% mutate(pct = 100 * importance / sum(importance, na.rm = TRUE))
        ggplot(df_imp_plot, aes(x = reorder(variable, pct), y = pct)) + geom_col(fill = '#2c7fb8') + coord_flip() + labs(x = NULL, y = 'Importance (%)', title = 'Feature importance (relative %)') + geom_text(aes(label = sprintf('%.1f%%', pct)), hjust = -0.1, size = 3) + theme_minimal()
      }, error = function(e) NULL)
      if(is.null(p)){
        png(file); plot.new(); text(0.5,0.5, 'Feature importance could not be extracted'); dev.off(); return()
      }
      # save via png device
      png(file, width = 1200, height = 800)
      print(p)
      dev.off()
    }
  )

  output$rocPlot <- renderPlotly({
    req(model_store$trained)
    roc_obj <- model_store$roc
    if(is.null(roc_obj)) return(NULL)
    df_roc <- data.frame(tpr = rev(roc_obj$sensitivities), fpr = rev(1 - roc_obj$specificities))
    p <- ggplot(df_roc, aes(x = fpr, y = tpr)) + geom_line() + geom_abline(linetype = "dashed") + labs(title = paste0('ROC AUC = ', round(auc(roc_obj), 3)))
    safe_ggplotly(p)
  })

  # EDA plots
  output$ageDist <- renderPlotly({
    df <- data()
    req(df)
    # pick a numeric column if exists named age, otherwise first numeric
    if("age" %in% names(df) && is.numeric(df$age)){
        p <- ggplot(df, aes(x = age)) + geom_histogram(bins = 20, fill = "steelblue") + labs(title = "Age distribution")
        safe_ggplotly(p)
    } else {
      nums <- df %>% select(where(is.numeric))
      if(ncol(nums) == 0) return(NULL)
      coln <- names(nums)[1]
  p <- ggplot(df, aes_string(x = coln)) + geom_histogram(bins = 20, fill = "steelblue") + labs(title = paste0(coln, " distribution"))
  safe_ggplotly(p)
    }
  })

  output$corrPlot <- renderPlotly({
    df <- data()
    req(df)
    nums <- df %>% select(where(is.numeric))
    if(ncol(nums) < 2) return(NULL)
    # remove constant columns (sd == 0) to avoid cor() warnings/errors
    sds <- sapply(nums, function(x) sd(x, na.rm = TRUE))
    keep <- names(sds)[which(!is.na(sds) & sds > 0)]
    if(length(keep) < 2) return(NULL)
    nums2 <- nums %>% select(all_of(keep))
    corr <- round(cor(nums2, use = "pairwise.complete.obs"), 2)
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
  safe_ggplotly(p)
  })

  # Main interactive EDA plot (many plot types)
  output$eda_plot <- renderPlotly({
    req(data())
    df <- data()
    pt <- input$plot_type
    if(is.null(pt)) pt <- "Histogram"
    # safe selection helpers
    xvar <- input$eda_x
    yvar <- input$eda_y
    catvar <- input$eda_cat

    # use target if available for coloring
    color_by <- if(!is.null(input$target) && input$target %in% names(df)) input$target else NULL

    p <- NULL
    # read appearance controls with safe defaults
    bins_in <- if(!is.null(input$bins)) as.integer(input$bins) else 30
    point_size <- if(!is.null(input$point_size)) input$point_size else 2
    add_jitter <- isTRUE(input$add_jitter)
    jitter_width <- if(!is.null(input$jitter_width)) input$jitter_width else 0.2
    add_smooth <- isTRUE(input$add_smooth)
    axis_transform <- if(!is.null(input$axis_transform)) input$axis_transform else "none"

    try({
      if(pt == "Histogram"){
        req(xvar)
        p <- ggplot(df, aes_string(x = xvar, fill = color_by)) + geom_histogram(alpha = 0.7, bins = bins_in) + labs(title = paste0("Histogram of ", xvar))
        if(axis_transform != "none") p <- p + scale_x_continuous(trans = axis_transform)
      } else if(pt == "Density"){
        req(xvar)
        if(!is.null(color_by)){
          p <- ggplot(df, aes_string(x = xvar, color = color_by, fill = color_by)) + geom_density(alpha = 0.3) + labs(title = paste0("Density of ", xvar))
        } else {
          p <- ggplot(df, aes_string(x = xvar)) + geom_density(fill = "steelblue", alpha = 0.5) + labs(title = paste0("Density of ", xvar))
        }
        if(axis_transform != "none") p <- p + scale_x_continuous(trans = axis_transform)
      } else if(pt == "Boxplot"){
        req(xvar)
        if(!is.null(color_by)){
          p <- ggplot(df, aes_string(x = color_by, y = xvar, fill = color_by)) + geom_boxplot()
          if(add_jitter) p <- p + geom_jitter(position = position_jitter(width = jitter_width), size = point_size, alpha = 0.7)
          p <- p + labs(title = paste0("Boxplot of ", xvar, " by ", color_by))
        } else {
          p <- ggplot(df, aes_string(y = xvar)) + geom_boxplot(fill = "steelblue") + labs(title = paste0("Boxplot of ", xvar))
          if(add_jitter) p <- p + geom_jitter(width = jitter_width, size = point_size, alpha = 0.7)
        }
        if(axis_transform != "none") p <- p + scale_y_continuous(trans = axis_transform)
      } else if(pt == "Violin"){
        req(xvar)
        if(!is.null(color_by)){
          p <- ggplot(df, aes_string(x = color_by, y = xvar, fill = color_by)) + geom_violin(alpha = 0.7)
          if(add_jitter) p <- p + geom_jitter(position = position_jitter(width = jitter_width), size = point_size, alpha = 0.6)
          p <- p + labs(title = paste0("Violin of ", xvar, " by ", color_by))
        } else {
          p <- ggplot(df, aes_string(y = xvar)) + geom_violin(fill = "steelblue", alpha = 0.7) + labs(title = paste0("Violin of ", xvar))
          if(add_jitter) p <- p + geom_jitter(width = jitter_width, size = point_size, alpha = 0.6)
        }
        if(axis_transform != "none") p <- p + scale_y_continuous(trans = axis_transform)
      } else if(pt == "Bar"){
        req(catvar)
        p <- ggplot(df, aes_string(x = catvar)) + geom_bar(fill = "steelblue") + labs(title = paste0("Count of ", catvar))
      } else if(pt == "Stacked Bar"){
        req(catvar)
        req(color_by)
        p <- ggplot(df, aes_string(x = catvar, fill = color_by)) + geom_bar(position = "fill") + labs(y = "Proportion", title = paste0(catvar, " by ", color_by))
      } else if(pt == "Scatter"){
        req(xvar, yvar)
        if(!is.null(color_by)){
          if(add_jitter) p <- ggplot(df, aes_string(x = xvar, y = yvar, color = color_by)) + geom_jitter(width = jitter_width, height = jitter_width, size = point_size, alpha = 0.8)
          else p <- ggplot(df, aes_string(x = xvar, y = yvar, color = color_by)) + geom_point(size = point_size, alpha = 0.8)
          if(add_smooth) p <- p + geom_smooth(method = "loess", se = FALSE)
        } else {
          if(add_jitter) p <- ggplot(df, aes_string(x = xvar, y = yvar)) + geom_jitter(width = jitter_width, height = jitter_width, size = point_size, alpha = 0.8)
          else p <- ggplot(df, aes_string(x = xvar, y = yvar)) + geom_point(size = point_size, alpha = 0.8)
          if(add_smooth) p <- p + geom_smooth(method = "loess", se = FALSE)
        }
        if(axis_transform != "none"){
          p <- p + scale_x_continuous(trans = axis_transform) + scale_y_continuous(trans = axis_transform)
        }
      } else if(pt == "Pair Plot"){
        # Pair plot handled by static GGally::ggpairs in a separate output; return NULL here.
        return(NULL)
      } else if(pt == "Missing Map"){
        miss_counts <- sapply(df, function(x) sum(is.na(x)))
        miss_df <- data.frame(variable = names(miss_counts), missing = as.integer(miss_counts))
        p <- ggplot(miss_df, aes(x = reorder(variable, -missing), y = missing)) + geom_bar(stat = 'identity', fill = 'salmon') + coord_flip() + labs(title = 'Missing values per column', x = '', y = 'Missing count')
      } else if(pt == "Correlation Heatmap"){
        nums <- numeric_vars()
        if(length(nums) < 2) return(NULL)
        corr <- round(cor(df %>% select(all_of(nums)), use = 'pairwise.complete.obs'), 2)
        p <- plot_ly(z = corr, x = colnames(corr), y = rownames(corr), type = 'heatmap', colorscale = 'Blues') %>% layout(title = 'Correlation Heatmap')
        return(p)
      }
    })
    if(is.null(p)) return(NULL)
    # If p is already a plotly object, return it directly
    if(inherits(p, "plotly")) return(p)
    # Convert ggplot to plotly safely and return a friendly message on failure
    safe_ggplotly(p)
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
    safe_ggplotly(p)
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
    safe_ggplotly(p)
  })

  # static GGally pair plot (rich ggpairs view) - falls back gracefully if GGally not installed
  # Build pair plot object so it can be rendered and downloaded
  pair_plot_obj <- reactive({
    df <- data()
    req(df)
    nums <- numeric_vars()
    if(length(nums) < 2) return(list(type = 'none', plot = NULL, note = 'Not enough numeric variables'))
    # if too many numerics, choose top variables by variance to keep plot readable
    if(length(nums) > 12){
      vars_sd <- sapply(df %>% select(all_of(nums)), function(x) sd(x, na.rm = TRUE))
      topn <- names(sort(vars_sd, decreasing = TRUE))[1:12]
      nums_sel <- topn
      note <- paste0('Selected top ', length(nums_sel), ' numeric vars by variance for plotting')
    } else {
      nums_sel <- nums
      note <- NULL
    }
    # prefer GGally::ggpairs if available
    if(requireNamespace('GGally', quietly = TRUE)){
      g <- tryCatch({
        GGally::ggpairs(df %>% select(all_of(nums_sel)), progress = FALSE, upper = list(continuous = GGally::wrap('cor', size = 3)))
      }, error = function(e){
        NULL
      })
      if(!is.null(g)) return(list(type = 'ggpairs', plot = g, note = note))
    }
    # fallback to base pairs
    return(list(type = 'pairs', plot = df %>% select(all_of(nums_sel)), note = note))
  })

  output$pairPlotStatic <- renderPlot({
    obj <- pair_plot_obj()
    if(is.null(obj) || obj$type == 'none'){
      plot.new(); text(0.5,0.5, 'Not enough numeric variables for pair plot')
      return()
    }
    if(obj$type == 'ggpairs'){
      print(obj$plot)
    } else if(obj$type == 'pairs'){
      pairs(obj$plot, main = 'Pairwise scatter (fallback)')
    }
  })

  # download handler for pair plot (PNG)
  output$downloadPair <- downloadHandler(
    filename = function(){ paste0('pairplot_', tools::file_path_sans_ext(input$dataset), '.png') },
    content = function(file){
      obj <- pair_plot_obj()
      if(is.null(obj) || obj$type == 'none'){
        png(file); plot.new(); text(0.5,0.5,'Not enough numeric variables'); dev.off(); return()
      }
      # create PNG device and print plot
      png(file, width = 1600, height = 1600)
      try({
        if(obj$type == 'ggpairs') print(obj$plot) else pairs(obj$plot, main = 'Pairwise scatter (fallback)')
      }, silent = TRUE)
      dev.off()
    }
  )

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

  # placeholder for prediction output so outputOptions can be set safely
  output$prediction_result <- renderPrint({
    cat("No prediction yet. Use the prediction UI after training a model.")
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
